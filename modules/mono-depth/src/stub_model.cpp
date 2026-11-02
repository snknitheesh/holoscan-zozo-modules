// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The model is written as ONNX protobuf directly: the few message types it
// needs are encoded here, so the module does not depend on ONNX or
// protobuf libraries.

#include <unistd.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <holoscan/mono_depth/image_fit.hpp>
#include <holoscan/mono_depth/stub_model.hpp>

namespace holoscan::mono_depth {
namespace {

static_assert(std::endian::native == std::endian::little,
              "ONNX raw tensor data is little-endian; this writer copies host bytes");

// ONNX enumerations.
constexpr int kFloat = 1;  // TensorProto.DataType.FLOAT
constexpr int kInt64 = 7;  // TensorProto.DataType.INT64

// One protobuf message under construction.
class Message {
 public:
  Message& varint(int field, std::uint64_t value) {
    tag(field, 0);
    put_varint(value);
    return *this;
  }
  Message& integer(int field, std::int64_t value) {
    return varint(field, static_cast<std::uint64_t>(value));
  }
  Message& bytes(int field, std::string_view value) {
    tag(field, 2);
    put_varint(value.size());
    out_.append(value);
    return *this;
  }
  Message& message(int field, const Message& value) { return bytes(field, value.out_); }
  [[nodiscard]] const std::string& str() const { return out_; }

 private:
  void tag(int field, int wire_type) {
    put_varint((static_cast<std::uint64_t>(field) << 3U) | static_cast<std::uint64_t>(wire_type));
  }
  void put_varint(std::uint64_t value) {
    while (value >= 0x80U) {
      out_.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
      value >>= 7U;
    }
    out_.push_back(static_cast<char>(value));
  }
  std::string out_;
};

template <typename T>
std::string raw(const std::vector<T>& values) {
  std::string bytes(values.size() * sizeof(T), '\0');
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

// TensorProto: dims = 1, data_type = 2, name = 8, raw_data = 9.
Message tensor(std::string_view name, const std::vector<std::int64_t>& dims, int type,
               std::string data) {
  Message t;
  for (const std::int64_t dim : dims) {
    t.integer(1, dim);
  }
  t.integer(2, type);
  t.bytes(8, name);
  t.bytes(9, data);
  return t;
}

Message int64_tensor(std::string_view name, std::vector<std::int64_t> values) {
  return tensor(name, {static_cast<std::int64_t>(values.size())}, kInt64, raw(values));
}

Message float_scalar(std::string_view name, float value) {
  return tensor(name, {}, kFloat, raw(std::vector<float>{value}));
}

// NodeProto: input = 1, output = 2, name = 3, op_type = 4.
Message node(std::string_view op, std::initializer_list<std::string_view> inputs,
             std::string_view output) {
  Message n;
  for (const std::string_view input : inputs) {
    n.bytes(1, input);
  }
  n.bytes(2, output);
  n.bytes(3, output);
  n.bytes(4, op);
  return n;
}

// ValueInfoProto for a float tensor: name = 1, type = 2 { tensor_type = 1 {
// elem_type = 1, shape = 2 { dim = 1 { dim_value = 1 } } } }.
Message float_value(std::string_view name, std::initializer_list<std::int64_t> dims) {
  Message shape;
  for (const std::int64_t dim : dims) {
    Message d;
    d.integer(1, dim);
    shape.message(1, d);
  }
  Message tensor_type;
  tensor_type.integer(1, kFloat).message(2, shape);
  Message type;
  type.message(1, tensor_type);
  Message value;
  value.bytes(1, name).message(2, type);
  return value;
}

}  // namespace

std::string stub_model_onnx(int width, int height) {
  if (width < 1 || height < 1 || width > kMaxModelSide || height > kMaxModelSide) {
    throw std::invalid_argument("stub model sides must be 1.." + std::to_string(kMaxModelSide));
  }
  Message graph;
  // GraphProto: node = 1, name = 2, initializer = 5, input = 11, output = 12.
  graph.bytes(2, "mono_depth_stub");
  const std::vector<Message> initializers{
      int64_tensor("channel_axis", {1}),
      int64_tensor("red_start", {0}),
      int64_tensor("red_end", {1}),
      int64_tensor("green_start", {1}),
      int64_tensor("green_end", {2}),
      int64_tensor("blue_start", {2}),
      int64_tensor("blue_end", {3}),
      int64_tensor("output_shape", {1, height, width}),
      float_scalar("red_weight", kStubWeights[0]),
      float_scalar("green_weight", kStubWeights[1]),
      float_scalar("blue_weight", kStubWeights[2]),
      float_scalar("bias", kStubBias),
  };
  for (const Message& initializer : initializers) {
    graph.message(5, initializer);
  }
  // ((red * w0 + green * w1) + blue * w2) + bias, as stub_response().
  const std::vector<Message> nodes{
      node("Slice", {"image", "red_start", "red_end", "channel_axis"}, "red"),
      node("Slice", {"image", "green_start", "green_end", "channel_axis"}, "green"),
      node("Slice", {"image", "blue_start", "blue_end", "channel_axis"}, "blue"),
      node("Mul", {"red", "red_weight"}, "red_term"),
      node("Mul", {"green", "green_weight"}, "green_term"),
      node("Mul", {"blue", "blue_weight"}, "blue_term"),
      node("Add", {"red_term", "green_term"}, "red_green"),
      node("Add", {"red_green", "blue_term"}, "weighted"),
      node("Add", {"weighted", "bias"}, "planar"),
      node("Reshape", {"planar", "output_shape"}, "depth"),
  };
  for (const Message& n : nodes) {
    graph.message(1, n);
  }
  graph.message(11, float_value("image", {1, 3, height, width}));
  graph.message(12, float_value("depth", {1, height, width}));

  // ModelProto: ir_version = 1, producer_name = 2, graph = 7, opset_import =
  // 8 { domain = 1, version = 2 }.
  Message opset;
  opset.bytes(1, "").integer(2, 13);
  Message model;
  model.integer(1, 7).bytes(2, "holoscan-mono-depth stub").message(7, graph).message(8, opset);
  return model.str();
}

std::filesystem::path write_stub_model(const std::filesystem::path& directory, int width,
                                       int height) {
  const std::string bytes = stub_model_onnx(width, height);
  const std::filesystem::path path = directory / ("mono_depth_stub_" + std::to_string(width) + "x" +
                                                  std::to_string(height) + ".onnx");
  {
    std::ifstream existing(path, std::ios::binary);
    if (existing) {
      const std::string current{std::istreambuf_iterator<char>(existing),
                                std::istreambuf_iterator<char>()};
      if (current == bytes) {
        return path;
      }
    }
  }
  static std::atomic<unsigned> sequence{0};
  const std::filesystem::path temporary =
      directory / (path.filename().string() + ".tmp." + std::to_string(::getpid()) + "." +
                   std::to_string(sequence.fetch_add(1)));
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out.flush()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw std::runtime_error("cannot write the stub model to " + temporary.string());
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    throw std::runtime_error("cannot write the stub model to " + path.string());
  }
  return path;
}

}  // namespace holoscan::mono_depth
