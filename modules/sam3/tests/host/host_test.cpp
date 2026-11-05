// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The module's host-only pieces, without the SDK, CUDA or a GPU: prompts and
// the prompt table file; the decoder inputs of a prompt; the mask sampling,
// ranking and box arithmetic, against independent evaluations; the stub
// model files and their references, which find each synthetic shape; the
// synthetic scene's geometry and truth; and the overlay reference.
//
// Usage: sam3_host_test DIR (a scratch folder for the files)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/sam3/pixel_math.hpp>
#include <holoscan/sam3/prompts.hpp>
#include <holoscan/sam3/reference.hpp>
#include <holoscan/sam3/stub_model.hpp>
#include <holoscan/sam3/synthetic_scene.hpp>

#include "test_support.hpp"

namespace holoscan::sam3 {
namespace {

std::filesystem::path g_directory;

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
}

void check_prompts() {
  VERIFY(table_text(Sam3Prompt{.text = "apple"}) == "apple");
  VERIFY(table_text(Sam3Prompt{.boxes = {PromptBox{.xyxy = {0, 0, 1, 1}}}}) == kVisualPrompt);
  VERIFY(display_name(Sam3Prompt{.boxes = {PromptBox{.xyxy = {0, 0, 1, 1}}}}) == "box");
  const auto rejects = [](Sam3Prompt prompt) {
    return test::throws<std::invalid_argument>([&] { validate_prompt(prompt); });
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  VERIFY(rejects(Sam3Prompt{}));
  VERIFY(rejects(Sam3Prompt{.text = std::string(kMaxPromptTextBytes + 1, 'a')}));
  VERIFY(rejects(Sam3Prompt{.boxes = {PromptBox{.xyxy = {0.5F, 0, 0.5F, 1}}}}));
  VERIFY(rejects(Sam3Prompt{.boxes = {PromptBox{.xyxy = {0, 0.6F, 1, 0.5F}}}}));
  VERIFY(rejects(Sam3Prompt{.boxes = {PromptBox{.xyxy = {-0.1F, 0, 1, 1}}}}));
  VERIFY(rejects(Sam3Prompt{.boxes = {PromptBox{.xyxy = {0, 0, 1.1F, 1}}}}));
  VERIFY(rejects(Sam3Prompt{.boxes = {PromptBox{.xyxy = {nan, 0, 1, 1}}}}));
  VERIFY(!rejects(Sam3Prompt{.text = "a", .boxes = {PromptBox{.xyxy = {0, 0, 1, 1}}}}));
}

PromptTable small_table() {
  PromptTable table{.tokens = 3, .dim = 2, .entries = {}};
  table.entries.push_back({.text = "apple", .mask = {1, 1, 0}, .features = {1, 2, 3, 4, 5, 6}});
  table.entries.push_back(
      {.text = std::string(kVisualPrompt), .mask = {1, 0, 0}, .features = {-1, 0, 0, 0, 0, 0}});
  return table;
}

void check_table_file() {
  const std::filesystem::path path = g_directory / "table.bin";
  const PromptTable table = small_table();
  write_prompt_table(path, table);
  // The documented layout, byte for byte: a header and two entries.
  const std::string bytes = read_file(path);
  VERIFY(bytes.size() == 8 + 12 + 2 * (4 + 3 * 4 + 6 * 4) + 5 + 6);
  VERIFY(bytes.compare(0, 8, "SAM3PRM1") == 0);
  const PromptTable read = read_prompt_table(path);
  VERIFY(read.tokens == 3 && read.dim == 2 && read.entries.size() == 2);
  VERIFY(read.entries[0].text == "apple" && read.entries[0].features == table.entries[0].features);
  VERIFY(read.entries[1].mask == table.entries[1].mask);
  VERIFY(read.find("apple") == &read.entries[0] && read.find("pear") == nullptr);

  const auto unreadable = [&](const std::string& damaged) {
    write_file(g_directory / "damaged.bin", damaged);
    return test::throws<std::runtime_error>(
        [&] { (void)read_prompt_table(g_directory / "damaged.bin"); });
  };
  VERIFY(unreadable(""));
  VERIFY(unreadable("SAM3PRM2" + bytes.substr(8)));
  VERIFY(unreadable(bytes.substr(0, bytes.size() - 1)));
  VERIFY(unreadable(bytes + "x"));
  VERIFY(test::throws<std::runtime_error>(
      [&] { (void)read_prompt_table(g_directory / "missing.bin"); }));
  // The same text twice.
  PromptTable twice = table;
  twice.entries.push_back(table.entries[0]);
  VERIFY(test::throws<std::invalid_argument>([&] { write_prompt_table(path, twice); }));
  PromptTable short_entry = table;
  short_entry.entries[0].features.pop_back();
  VERIFY(test::throws<std::invalid_argument>([&] { write_prompt_table(path, short_entry); }));
}

void check_decoder_prompt() {
  const PromptTable table = small_table();
  const DecoderPrompt text = make_decoder_prompt(Sam3Prompt{.text = "apple"}, table, 2);
  VERIFY(text.text_features == table.entries[0].features);
  VERIFY(text.text_mask == table.entries[0].mask);
  VERIFY(text.boxes == std::vector<float>(8, 0.0F));
  VERIFY(text.box_labels == std::vector<float>(2, 0.0F) && text.box_valid == text.box_labels);
  // Boxes alone use "visual"; boxes become center, size; labels and validity.
  const DecoderPrompt boxes = make_decoder_prompt(
      Sam3Prompt{.boxes = {PromptBox{.xyxy = {0.25F, 0.5F, 0.75F, 1.0F}},
                           PromptBox{.xyxy = {0.0F, 0.0F, 0.5F, 0.25F}, .positive = false}}},
      table, 3);
  VERIFY(boxes.text_features == table.entries[1].features);
  VERIFY((boxes.boxes ==
          std::vector<float>{0.5F, 0.75F, 0.5F, 0.5F, 0.25F, 0.125F, 0.5F, 0.25F, 0, 0, 0, 0}));
  VERIFY((boxes.box_labels == std::vector<float>{1, 0, 0}));
  VERIFY((boxes.box_valid == std::vector<float>{1, 1, 0}));
  VERIFY(test::throws<std::invalid_argument>(
      [&] { (void)make_decoder_prompt(Sam3Prompt{.text = "pear"}, table, 2); }));
  VERIFY(test::throws<std::invalid_argument>([&] {
    (void)make_decoder_prompt(
        Sam3Prompt{.text = "apple", .boxes = {PromptBox{.xyxy = {0, 0, 1, 1}}}}, table, 0);
  }));
}

// The mask sampling against a direct evaluation of PyTorch's bilinear
// interpolation with align_corners=False.
void check_mask_sampling() {
  const std::vector<float> logits{0, 1, 2, 3, 4, 5};  // 3 wide, 2 high.
  const MaskSampling s = make_mask_sampling(3, 2, 6, 4);
  // Pixel (0, 0): source (0.5 * 0.5 - 0.5, ...) clamped to 0: logits[0].
  VERIFY(sample_mask(s, logits.data(), 0, 0) == 0.0F);
  // Pixel (3, 1): source x = 3.5 * 0.5 - 0.5 = 1.25, y = 1.5 * 0.5 - 0.5 =
  // 0.25: 0.75 * (0.75 * 1 + 0.25 * 2) + 0.25 * (0.75 * 4 + 0.25 * 5) = 2.
  VERIFY(std::fabs(sample_mask(s, logits.data(), 3, 1) - 2.0F) < 1e-6F);
  // The last pixel clamps to the last cell.
  VERIFY(sample_mask(s, logits.data(), 5, 3) == 5.0F);
  VERIFY(std::fabs(logit_of(0.5F)) < 1e-7F);
  VERIFY(std::fabs(logit_of(0.75F) - std::log(3.0F)) < 1e-6F);
}

void check_ranking_and_boxes() {
  VERIFY(ranks_before(0.9F, 5, 0.8F, 1));
  VERIFY(ranks_before(0.8F, 1, 0.8F, 5));
  VERIFY(!ranks_before(0.8F, 5, 0.8F, 1));
  VERIFY(!ranks_before(0.8F, 1, 0.8F, 1));
  const float fraction[4] = {-0.1F, 0.25F, 0.5F, 1.5F};
  float pixels[4];
  box_to_pixels(fraction, 200, 100, pixels);
  VERIFY(pixels[0] == 0.0F && pixels[1] == 25.0F && pixels[2] == 100.0F && pixels[3] == 100.0F);
}

void check_references() {
  // Two prompts of two queries on a 2x2 mask grid over a 4x2 image.
  DecoderOutput a{.queries = 2,
                  .mask_width = 2,
                  .mask_height = 2,
                  .scores = {0.6F, 0.9F},
                  .boxes = {0, 0, 0.5F, 1, 0.5F, 0, 1, 1},
                  .masks = {1, -1, 1, -1, -1, 1, -1, 1}};
  DecoderOutput b{.queries = 2,
                  .mask_width = 2,
                  .mask_height = 2,
                  .scores = {0.9F, 0.2F},
                  .boxes = {0, 0, 1, 1, 0, 0, 1, 1},
                  .masks = {1, 1, 1, 1, 1, 1, 1, 1}};
  const std::vector<DecoderOutput> both{a, b};
  const ReferenceInstances r = reference_instances(both, 4, 0.5F, 0.0F, 4, 2);
  // 0.9 of prompt 0 (query 1), 0.9 of prompt 1 (query 0), 0.6 of prompt 0.
  VERIFY(r.count == 3);
  VERIFY((r.scores == std::vector<float>{0.9F, 0.9F, 0.6F, 0.0F}));
  VERIFY((r.prompt_ids == std::vector<std::int32_t>{0, 1, 0, -1}));
  VERIFY((r.boxes == std::vector<float>{2, 0, 4, 2, 0, 0, 4, 2, 0, 0, 2, 2, 0, 0, 0, 0}));
  VERIFY((r.passed == std::vector<int>{2, 1}));
  // Instance 1 (prompt 0, query 1) covers the right half, instance 2
  // (everything) the rest, instance 3 the left half, under instance 2.
  VERIFY((r.labels == std::vector<std::uint16_t>{2, 2, 1, 1, 2, 2, 1, 1}));
  VERIFY((std::vector<std::uint8_t>(r.masks.begin() + 16, r.masks.begin() + 24) ==
          std::vector<std::uint8_t>{1, 1, 0, 0, 1, 1, 0, 0}));
  // A capacity of one keeps the best, the lower slot first.
  const ReferenceInstances one = reference_instances(both, 1, 0.5F, 0.0F, 4, 2);
  VERIFY(one.count == 1 && one.prompt_ids[0] == 0 && one.scores[0] == 0.9F);
}

void check_overlay_reference() {
  const std::vector<std::uint16_t> labels{0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2};
  const std::vector<std::uint8_t> rgba = reference_overlay(labels, 4, 4, 77);
  std::uint8_t first[3];
  overlay_color(0, first);
  VERIFY(rgba[3] == 0);                                       // Background.
  VERIFY(rgba[4 * 1 + 3] == 255);                             // On the image's edge.
  VERIFY(rgba[4 * 10 + 3] == 255);                            // Above label 2.
  VERIFY(rgba[4 * 6 + 3] == 77);                              // Inside.
  VERIFY(rgba[4 * 5 + 3] == 255 && rgba[4 * 5] == first[0]);  // Beside the background.
  std::uint8_t second[3];
  overlay_color(1, second);
  VERIFY(rgba[4 * 12] == second[0] && rgba[4 * 12 + 1] == second[1]);
  // An inside pixel is translucent.
  const std::vector<std::uint16_t> block(25, 3);
  VERIFY(reference_overlay(block, 5, 5, 77)[4 * 12 + 3] == 77);
}

void check_scene() {
  const ShapeScene scene = make_shape_scene(640, 480, 0);
  std::vector<std::uint8_t> pixels(640 * 480 * 4);
  std::vector<SceneShape> shapes(640 * 480);
  render_shape_scene(scene, PixelFormat::kBgra8, pixels.data(), 640 * 4, shapes.data());
  std::array<std::size_t, 4> area{};
  for (const SceneShape shape : shapes) {
    ++area[static_cast<std::size_t>(shape)];
  }
  // Every shape is there, and covers a fair share of the image.
  for (std::size_t s = 1; s < 4; ++s) {
    VERIFY_MSG(area[s] > 640 * 480 / 30, "shape " + std::to_string(s) + " too small");
  }
  // The circle's center is red (BGRA), its box holds it.
  const int cx = static_cast<int>(scene.circle_center[0]);
  const int cy = static_cast<int>(scene.circle_center[1]);
  const std::uint8_t* center = pixels.data() + (static_cast<std::size_t>(cy) * 640 + cx) * 4;
  VERIFY(center[2] == 220 && center[1] == 30 && center[0] == 30 && center[3] == 255);
  VERIFY(shapes[static_cast<std::size_t>(cy) * 640 + cx] == SceneShape::kCircle);
  for (const SceneShape shape :
       {SceneShape::kCircle, SceneShape::kRectangle, SceneShape::kTriangle}) {
    const std::array<float, 4> box = shape_box(scene, shape);
    for (int y = 0; y < 480; ++y) {
      for (int x = 0; x < 640; ++x) {
        if (shapes[static_cast<std::size_t>(y) * 640 + x] == shape) {
          VERIFY_MSG((x + 0.5F) / 640 >= box[0] - 1e-6F && (x + 0.5F) / 640 <= box[2] + 1e-6F &&
                         (y + 0.5F) / 480 >= box[1] - 1e-6F && (y + 0.5F) / 480 <= box[3] + 1e-6F,
                     "a shape's pixel lies outside its box");
          x = 640;
          y = 480;
        }
      }
    }
  }
  // The scene moves.
  VERIFY(make_shape_scene(640, 480, 30).circle_center != scene.circle_center);
  // The IoU of the truth with itself, of nothing, and of half.
  std::vector<std::uint8_t> mask(shapes.size());
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    mask[i] = shapes[i] == SceneShape::kCircle ? 1 : 0;
  }
  VERIFY(mask_iou(mask, shapes, SceneShape::kCircle) == 1.0);
  VERIFY(mask_iou(mask, shapes, SceneShape::kRectangle) == 0.0);
  VERIFY(test::throws<std::invalid_argument>([] { (void)make_shape_scene(16, 480, 0); }));
}

void check_stub() {
  const StubShape shape{};
  const StubModel files = write_stub_model(g_directory, shape);
  const std::string encoder = read_file(files.encoder);
  const std::string decoder = read_file(files.decoder);
  VERIFY(encoder == stub_encoder_onnx(shape) && decoder == stub_decoder_onnx(shape));
  VERIFY(encoder.find("AveragePool") != std::string::npos);
  VERIFY(decoder.find("text_features") != std::string::npos);
  // Written again, the files stay as they are.
  const auto written = std::filesystem::last_write_time(files.encoder);
  (void)write_stub_model(g_directory, shape);
  VERIFY(std::filesystem::last_write_time(files.encoder) == written);
  const PromptTable table = read_prompt_table(files.prompt_table);
  VERIFY(table.tokens == shape.tokens && table.dim == shape.dim);
  VERIFY(table.find("red") && table.find("blue") && table.find(kVisualPrompt));
  VERIFY(test::throws<std::invalid_argument>(
      [] { validate_stub_shape(StubShape{.input_size = 60, .mask_size = 16}); }));
  VERIFY(test::throws<std::invalid_argument>(
      [] { validate_stub_shape(StubShape{.queries = 9, .tokens = 8}); }));

  // Through the references, each stub prompt finds its shape in the
  // synthetic scene, as the ci mode checks.
  const int width = 640;
  const int height = 480;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
  std::vector<SceneShape> shapes(static_cast<std::size_t>(width) * height);
  const ShapeScene scene = make_shape_scene(width, height, 0);
  render_shape_scene(scene, PixelFormat::kRgb8, pixels.data(), width * 3U, shapes.data());
  const std::vector<float> input =
      reference_preprocess(pixels.data(), width * 3U, PixelFormat::kRgb8, width, height,
                           shape.input_size, shape.input_size, Normalization{});
  const std::vector<float> fpn0 = reference_stub_encoder(input, shape);
  const std::vector<Sam3Prompt> prompts{
      {.text = "red"},
      {.text = "blue"},
      {.boxes = {PromptBox{.xyxy = shape_box(scene, SceneShape::kTriangle)}}}};
  std::vector<DecoderOutput> outputs;
  for (const Sam3Prompt& prompt : prompts) {
    outputs.push_back(
        reference_stub_decoder(fpn0, make_decoder_prompt(prompt, table, shape.boxes), shape));
  }
  const ReferenceInstances r = reference_instances(outputs, 8, 0.5F, 0.0F, width, height);
  VERIFY_MSG(r.count == 3, "stub instances: " + std::to_string(r.count));
  VERIFY((r.prompt_ids == std::vector<std::int32_t>{0, 1, 2, -1, -1, -1, -1, -1}));
  VERIFY((std::vector<float>(r.scores.begin(), r.scores.begin() + 3) ==
          std::vector<float>{0.9F, 0.8F, 0.7F}));
  const std::size_t plane = static_cast<std::size_t>(width) * height;
  const std::array<SceneShape, 3> expected{SceneShape::kCircle, SceneShape::kRectangle,
                                           SceneShape::kTriangle};
  for (std::size_t k = 0; k < 3; ++k) {
    const std::span<const std::uint8_t> mask(r.masks.data() + k * plane, plane);
    const double iou = mask_iou(mask, shapes, expected[k]);
    VERIFY_MSG(iou > 0.7, "stub prompt " + std::to_string(k) + " IoU " + std::to_string(iou));
  }
  // The box prompt's box is added to the stub's own (zero) box.
  const auto& b = prompts[2].boxes[0].xyxy;
  VERIFY(std::fabs(r.boxes[8] - (b[0] + b[2]) * 0.5F * width) < 1e-3F);
}

}  // namespace
}  // namespace holoscan::sam3

int main(int argc, char** argv) {
  namespace s3 = holoscan::sam3;
  if (argc != 2) {
    return 2;
  }
  s3::g_directory = argv[1];
  std::filesystem::create_directories(s3::g_directory);
  return s3::test::run({
      {"prompts", s3::check_prompts},
      {"prompt table file", s3::check_table_file},
      {"decoder inputs of a prompt", s3::check_decoder_prompt},
      {"mask sampling", s3::check_mask_sampling},
      {"ranking and boxes", s3::check_ranking_and_boxes},
      {"instance references", s3::check_references},
      {"overlay reference", s3::check_overlay_reference},
      {"synthetic scene", s3::check_scene},
      {"stub model", s3::check_stub},
  });
}
