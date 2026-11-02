// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The models are written as ONNX protobuf directly: the few message types they
// need are encoded here, so the module does not depend on ONNX or protobuf
// libraries.

#include <unistd.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/sam3/pixel_math.hpp>
#include <holoscan/sam3/stub_model.hpp>

namespace holoscan::sam3 {
namespace {

static_assert(std::endian::native == std::endian::little,
              "ONNX raw tensor data is little-endian; this writer copies host bytes");

// ONNX enumerations.
constexpr int kFloat = 1;          // TensorProto.DataType.FLOAT
constexpr int kInt64 = 7;          // TensorProto.DataType.INT64
constexpr int kAttributeInt = 2;   // AttributeProto.AttributeType.INT
constexpr int kAttributeInts = 7;  // AttributeProto.AttributeType.INTS

// Where a token's fields sit in its text features.
constexpr int kWeights = 0;
constexpr int kBias = 3;
constexpr int kScore = 4;
constexpr int kBox = 5;
constexpr int kZero = 9;
constexpr int kFields = 10;

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
Message int64_tensor(std::string_view name, std::vector<std::int64_t> values) {
  Message t;
  t.integer(1, static_cast<std::int64_t>(values.size()));
  t.integer(2, kInt64);
  t.bytes(8, name);
  t.bytes(9, raw(values));
  return t;
}

// AttributeProto: name = 1, i = 3, ints = 8, type = 20.
Message ints_attribute(std::string_view name, std::initializer_list<std::int64_t> values) {
  Message a;
  a.bytes(1, name);
  for (const std::int64_t value : values) {
    a.integer(8, value);
  }
  a.integer(20, kAttributeInts);
  return a;
}

Message int_attribute(std::string_view name, std::int64_t value) {
  Message a;
  a.bytes(1, name).integer(3, value).integer(20, kAttributeInt);
  return a;
}

// NodeProto: input = 1, output = 2, name = 3, op_type = 4, attribute = 5.
Message node(std::string_view op, std::initializer_list<std::string_view> inputs,
             std::string_view output, std::initializer_list<Message> attributes = {}) {
  Message n;
  for (const std::string_view input : inputs) {
    n.bytes(1, input);
  }
  n.bytes(2, output);
  n.bytes(3, output);
  n.bytes(4, op);
  for (const Message& attribute : attributes) {
    n.message(5, attribute);
  }
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

// GraphProto: node = 1, name = 2, initializer = 5, input = 11, output = 12;
// ModelProto: ir_version = 1, producer_name = 2, graph = 7, opset_import = 8
// { domain = 1, version = 2 }.
std::string model(std::string_view name, const std::vector<Message>& initializers,
                  const std::vector<Message>& nodes, const std::vector<Message>& inputs,
                  const std::vector<Message>& outputs) {
  Message graph;
  graph.bytes(2, name);
  for (const Message& initializer : initializers) {
    graph.message(5, initializer);
  }
  for (const Message& n : nodes) {
    graph.message(1, n);
  }
  for (const Message& input : inputs) {
    graph.message(11, input);
  }
  for (const Message& output : outputs) {
    graph.message(12, output);
  }
  Message opset;
  opset.bytes(1, "").integer(2, 13);
  Message result;
  result.integer(1, 7).bytes(2, "holoscan-sam3 stub").message(7, graph).message(8, opset);
  return result.str();
}

// Writes `bytes` to `path` unless it already holds them, atomically.
void write_if_changed(const std::filesystem::path& path, const std::string& bytes) {
  {
    std::ifstream existing(path, std::ios::binary);
    if (existing) {
      const std::string current{std::istreambuf_iterator<char>(existing),
                                std::istreambuf_iterator<char>()};
      if (current == bytes) {
        return;
      }
    }
  }
  static std::atomic<unsigned> sequence{0};
  const std::filesystem::path temporary =
      path.parent_path() / (path.filename().string() + ".tmp." + std::to_string(::getpid()) + "." +
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
}

std::string shape_tag(const StubShape& s) {
  return std::to_string(s.input_size) + "_m" + std::to_string(s.mask_size) + "_q" +
         std::to_string(s.queries) + "_t" + std::to_string(s.tokens) + "_d" +
         std::to_string(s.dim) + "_b" + std::to_string(s.boxes);
}

}  // namespace

void validate_stub_shape(const StubShape& s) {
  if (s.mask_size < 4 || s.mask_size % 4 != 0 || s.mask_size > kMaxModelSide ||
      s.input_size < s.mask_size || s.input_size % s.mask_size != 0 ||
      s.input_size > kMaxModelSide || s.queries < 1 || s.queries > 1024 || s.tokens < s.queries ||
      s.tokens > 4096 || s.dim < kFields || s.dim > 4096 || s.boxes < 1 || s.boxes > 64) {
    throw std::invalid_argument(
        "stub shape: mask_size a multiple of 4, input_size a multiple of it, 1..1024 queries, "
        "at least as many tokens, dim at least 10, 1..64 boxes");
  }
}

TextEmbedding stub_embedding(std::string text, const StubShape& shape,
                             std::span<const StubQuery> queries) {
  validate_stub_shape(shape);
  if (queries.size() > static_cast<std::size_t>(shape.queries)) {
    throw std::invalid_argument("the stub has " + std::to_string(shape.queries) + " queries");
  }
  TextEmbedding entry;
  entry.text = std::move(text);
  entry.mask.assign(static_cast<std::size_t>(shape.tokens), 0.0F);
  entry.features.assign(static_cast<std::size_t>(shape.tokens) * shape.dim, 0.0F);
  for (std::size_t q = 0; q < static_cast<std::size_t>(shape.queries); ++q) {
    entry.mask[q] = 1.0F;
  }
  for (std::size_t q = 0; q < queries.size(); ++q) {
    float* token = entry.features.data() + q * static_cast<std::size_t>(shape.dim);
    const StubQuery& query = queries[q];
    for (int c = 0; c < 3; ++c) {
      token[kWeights + c] = query.weights[static_cast<std::size_t>(c)];
    }
    token[kBias] = query.bias;
    token[kScore] = query.score;
    for (int e = 0; e < 4; ++e) {
      token[kBox + e] = query.box[static_cast<std::size_t>(e)];
    }
  }
  return entry;
}

PromptTable stub_prompt_table(const StubShape& shape) {
  // Normalized channels lie in [-1, 1]. Each prompt's first query is
  // positive on one of the synthetic scenes' colors only: red (220, 30, 30),
  // blue (30, 60, 220) and yellow (240, 220, 30), on a mid-gray background.
  const StubQuery never{
      .weights = {0.0F, 0.0F, 0.0F}, .bias = 1.0F, .score = 0.25F, .box = {0.0F, 0.0F, 1.0F, 1.0F}};
  const std::vector<StubQuery> red{{.weights = {2.0F, -2.0F, -1.0F},
                                    .bias = -1.5F,
                                    .score = 0.9F,
                                    .box = {0.1F, 0.1F, 0.4F, 0.5F}},
                                   never};
  const std::vector<StubQuery> blue{{.weights = {-1.0F, -1.0F, 2.0F},
                                     .bias = -1.5F,
                                     .score = 0.8F,
                                     .box = {0.5F, 0.1F, 0.9F, 0.4F}},
                                    never};
  const std::vector<StubQuery> visual{{.weights = {1.0F, 1.0F, -1.0F},
                                       .bias = -1.5F,
                                       .score = 0.7F,
                                       .box = {0.0F, 0.0F, 0.0F, 0.0F}},
                                      never};
  PromptTable table{.tokens = shape.tokens, .dim = shape.dim, .entries = {}};
  table.entries.push_back(stub_embedding("red", shape, red));
  table.entries.push_back(stub_embedding("blue", shape, blue));
  table.entries.push_back(stub_embedding(std::string(kVisualPrompt), shape, visual));
  return table;
}

std::string stub_encoder_onnx(const StubShape& s) {
  validate_stub_shape(s);
  const std::int64_t cell = s.input_size / s.mask_size;
  const std::int64_t m = s.mask_size;
  const std::vector<Message> nodes{
      node("AveragePool", {"image"}, "fpn0",
           {ints_attribute("kernel_shape", {cell, cell}), ints_attribute("strides", {cell, cell})}),
      node("AveragePool", {"fpn0"}, "fpn1",
           {ints_attribute("kernel_shape", {2, 2}), ints_attribute("strides", {2, 2})}),
      node("AveragePool", {"fpn1"}, "fpn2",
           {ints_attribute("kernel_shape", {2, 2}), ints_attribute("strides", {2, 2})}),
  };
  return model("sam3_stub_encoder", {}, nodes,
               {float_value("image", {1, 3, s.input_size, s.input_size})},
               {float_value("fpn0", {1, 3, m, m}), float_value("fpn1", {1, 3, m / 2, m / 2}),
                float_value("fpn2", {1, 3, m / 4, m / 4})});
}

std::string stub_decoder_onnx(const StubShape& s) {
  validate_stub_shape(s);
  const std::int64_t m = s.mask_size;
  const std::int64_t q = s.queries;
  const std::vector<Message> initializers{
      int64_tensor("zero", {0}),
      int64_tensor("query_end", {q}),
      int64_tensor("axis1", {1}),
      int64_tensor("axis2", {2}),
      int64_tensor("weights_end", {kBias}),
      int64_tensor("bias_start", {kBias}),
      int64_tensor("bias_end", {kScore}),
      int64_tensor("score_start", {kScore}),
      int64_tensor("score_end", {kBox}),
      int64_tensor("box_start", {kBox}),
      int64_tensor("box_end", {kZero}),
      int64_tensor("zero_start", {kZero}),
      int64_tensor("zero_end", {kFields}),
      int64_tensor("pixels_shape", {1, 3, m * m}),
      int64_tensor("scalar_shape", {1, 1, 1}),
      int64_tensor("masks_shape", {1, q, m, m}),
      int64_tensor("scores_shape", {1, q}),
      int64_tensor("box_weight_shape", {1, s.boxes, 1}),
  };
  const std::vector<Message> nodes{
      node("Slice", {"text_features", "zero", "query_end", "axis1"}, "tokens"),
      node("Slice", {"tokens", "zero", "weights_end", "axis2"}, "weights"),
      node("Slice", {"tokens", "bias_start", "bias_end", "axis2"}, "bias"),
      node("Slice", {"tokens", "score_start", "score_end", "axis2"}, "score_column"),
      node("Slice", {"tokens", "box_start", "box_end", "axis2"}, "token_boxes"),
      node("Slice", {"tokens", "zero_start", "zero_end", "axis2"}, "zero_column"),
      // Masks: weights . fpn0 + bias + zero * (mean(fpn1) - mean(fpn2)).
      node("Reshape", {"fpn0", "pixels_shape"}, "pixels"),
      node("MatMul", {"weights", "pixels"}, "weighted"),
      node("Add", {"weighted", "bias"}, "biased"),
      node("ReduceMean", {"fpn1"}, "fpn1_mean", {int_attribute("keepdims", 1)}),
      node("ReduceMean", {"fpn2"}, "fpn2_mean", {int_attribute("keepdims", 1)}),
      node("Sub", {"fpn1_mean", "fpn2_mean"}, "mean_difference"),
      node("Reshape", {"mean_difference", "scalar_shape"}, "difference"),
      node("Mul", {"zero_column", "difference"}, "nothing"),
      node("Add", {"biased", "nothing"}, "logits"),
      node("Reshape", {"logits", "masks_shape"}, "masks"),
      // Scores: the score times the token's mask value.
      node("Slice", {"text_mask", "zero", "query_end", "axis1"}, "token_mask"),
      node("Reshape", {"score_column", "scores_shape"}, "token_scores"),
      node("Mul", {"token_scores", "token_mask"}, "scores"),
      // Boxes: the token's box plus the sum of the positive prompt boxes.
      node("Mul", {"box_valid", "box_labels"}, "box_weight"),
      node("Reshape", {"box_weight", "box_weight_shape"}, "box_weight_column"),
      node("Mul", {"boxes", "box_weight_column"}, "weighted_boxes"),
      node("ReduceSum", {"weighted_boxes", "axis1"}, "box_sum", {int_attribute("keepdims", 1)}),
      node("Add", {"token_boxes", "box_sum"}, "boxes_xyxy"),
  };
  return model("sam3_stub_decoder", initializers, nodes,
               {float_value("fpn0", {1, 3, m, m}), float_value("fpn1", {1, 3, m / 2, m / 2}),
                float_value("fpn2", {1, 3, m / 4, m / 4}),
                float_value("text_features", {1, s.tokens, s.dim}),
                float_value("text_mask", {1, s.tokens}), float_value("boxes", {1, s.boxes, 4}),
                float_value("box_labels", {1, s.boxes}), float_value("box_valid", {1, s.boxes})},
               {float_value("masks", {1, q, m, m}), float_value("boxes_xyxy", {1, q, 4}),
                float_value("scores", {1, q})});
}

StubModel write_stub_model(const std::filesystem::path& directory, const StubShape& shape) {
  const std::string tag = shape_tag(shape);
  StubModel result{.encoder = directory / ("sam3_stub_encoder_" + tag + ".onnx"),
                   .decoder = directory / ("sam3_stub_decoder_" + tag + ".onnx"),
                   .prompt_table = directory / ("sam3_stub_prompts_" + tag + ".bin")};
  write_if_changed(result.encoder, stub_encoder_onnx(shape));
  write_if_changed(result.decoder, stub_decoder_onnx(shape));
  write_prompt_table(result.prompt_table, stub_prompt_table(shape));
  return result;
}

std::vector<float> reference_stub_encoder(std::span<const float> input, const StubShape& s) {
  validate_stub_shape(s);
  const auto side = static_cast<std::size_t>(s.input_size);
  const auto m = static_cast<std::size_t>(s.mask_size);
  const std::size_t cell = side / m;
  if (input.size() != 3 * side * side) {
    throw std::invalid_argument("reference_stub_encoder: the input is not [3, S, S]");
  }
  std::vector<float> fpn0(3 * m * m);
  const float count = static_cast<float>(cell * cell);
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t y = 0; y < m; ++y) {
      for (std::size_t x = 0; x < m; ++x) {
        float sum = 0.0F;
        for (std::size_t v = 0; v < cell; ++v) {
          for (std::size_t u = 0; u < cell; ++u) {
            sum += input[c * side * side + (y * cell + v) * side + x * cell + u];
          }
        }
        fpn0[c * m * m + y * m + x] = sum / count;
      }
    }
  }
  return fpn0;
}

DecoderOutput reference_stub_decoder(std::span<const float> fpn0, const DecoderPrompt& prompt,
                                     const StubShape& s) {
  validate_stub_shape(s);
  const auto m = static_cast<std::size_t>(s.mask_size);
  const auto dim = static_cast<std::size_t>(s.dim);
  if (fpn0.size() != 3 * m * m ||
      prompt.text_features.size() != static_cast<std::size_t>(s.tokens) * dim ||
      prompt.text_mask.size() != static_cast<std::size_t>(s.tokens) ||
      prompt.boxes.size() != 4 * static_cast<std::size_t>(s.boxes)) {
    throw std::invalid_argument("reference_stub_decoder: inputs of the wrong size");
  }
  DecoderOutput out;
  out.queries = s.queries;
  out.mask_width = s.mask_size;
  out.mask_height = s.mask_size;
  out.scores.resize(static_cast<std::size_t>(s.queries));
  out.boxes.resize(4 * static_cast<std::size_t>(s.queries));
  out.masks.resize(static_cast<std::size_t>(s.queries) * m * m);
  float box_sum[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (std::size_t b = 0; b < static_cast<std::size_t>(s.boxes); ++b) {
    const float weight = prompt.box_valid[b] * prompt.box_labels[b];
    for (std::size_t e = 0; e < 4; ++e) {
      box_sum[e] += prompt.boxes[4 * b + e] * weight;
    }
  }
  for (std::size_t q = 0; q < static_cast<std::size_t>(s.queries); ++q) {
    const float* token = prompt.text_features.data() + q * dim;
    out.scores[q] = token[kScore] * prompt.text_mask[q];
    for (std::size_t e = 0; e < 4; ++e) {
      out.boxes[4 * q + e] = token[kBox + e] + box_sum[e];
    }
    for (std::size_t i = 0; i < m * m; ++i) {
      const float weighted = token[kWeights] * fpn0[i] + token[kWeights + 1] * fpn0[m * m + i] +
                             token[kWeights + 2] * fpn0[2 * m * m + i];
      out.masks[q * m * m + i] = weighted + token[kBias];
    }
  }
  return out;
}

}  // namespace holoscan::sam3
