# holoscan-sam3

A Holoscan SDK 5 Module for promptable concept segmentation with Meta's
[SAM 3](https://github.com/facebookresearch/sam3) on the GPU. `Sam3SegmentationOp` takes an image
and a list of prompts, short noun phrases ("apple", "red mug") or exemplar boxes, and emits every
instance of every prompt's concept, best first: a mask at the image's size, a score, a box and the
index of its prompt, with a label image of them all. SAM 3 runs as two TensorRT engines through
the SDK's inference library (HoloInfer): its image encoder once per image, and its prompt decoder
once per prompt, so further prompts are cheap. `Sam3OverlayOp` draws the instances for HoloViz.
The [SAM 3 Segmentation demo](../../applications/sam3_segmentation/) runs both end to end.

> **The model is not part of this module, and is licensed separately.** SAM 3's weights are
> Meta's, under the **SAM License** (see [NOTICE](NOTICE)), and their Hugging Face repository is
> **gated**: request access at <https://huggingface.co/facebook/sam3> and accept the license
> there. [`tools/prepare_model.py`](tools/prepare_model.py) then downloads a pinned revision with
> your own access token, and exports it to ONNX with PyTorch and Hugging Face Transformers, which
> you install separately. Nothing is redistributed. The tests need no model: they use a stand-in
> that the module writes itself.
>
> **Text prompts are precomputed.** SAM 3's text encoder runs in the export tool, once per
> phrase, not in the graph: the tool writes a prompt table with the phrases' embeddings, and the
> operator can use any phrase in it. Add new phrases with `prepare_model.py --prompts`.
>
> **One image at a time.** Video tracking with SAM 3's memory is not part of this module; see
> [Toward tracking](#toward-tracking).

| | |
| --- | --- |
| CMake package | `find_package(holoscan-sam3 [COMPONENTS overlay])` |
| Targets | `holoscan::sam3`, `holoscan::sam3_overlay` |
| Headers | `<holoscan/sam3/...>` |
| C++ namespace | `holoscan::sam3` |
| Debian package | `holoscan-sam3` |
| Model tool | `tools/prepare_model.py`, installed as `share/holoscan-sam3/prepare_model.py` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7.2) |

## Operators

### `Sam3SegmentationOp`

| Port | Direction | Type |
| --- | --- | --- |
| `image` | in | `holoscan::schema::ImageT`: RGB8, BGR8, RGBA8 or BGRA8, uint8 `[H, W, C]`, pixels packed within each row; GPU memory on device 0, or host or pinned memory |
| `masks` | out | uint8 `[max_instances, H, W]`: 1 inside each instance, 0 outside, zero beyond `count`. Declared only with `output_masks` |
| `labels` | out | uint16 `[H, W]`: 1 + the index of the best instance covering each pixel, 0 for none |
| `scores` | out | float32 `[max_instances]`: each instance's score, best first |
| `boxes` | out | float32 `[max_instances, 4]`: `x1, y1, x2, y2` in image pixels, clamped to the image |
| `prompt_ids` | out | int32 `[max_instances]`: the index of each instance's prompt; -1 beyond `count` |
| `count` | out | int32 `[1]`: the number of instances |

All outputs are in GPU memory. Connect every output the configuration declares. They carry the
image's capture time and frame id, and go out in the order of the table: should one fail to go
out, those before it have been delivered alone and the image fails. Pair them by capture time, with
an `OnAligned` contract, as `Sam3OverlayOp` does. Instances of all prompts are ranked together by
score, so `max_instances` bounds an image's instances, not each prompt's; when more pass the
threshold, the lowest are dropped and the operator warns once per run. Images may change size from
frame to frame, up to `max_width` x `max_height`; images in host memory are copied to the GPU
through a staging buffer, which the operator says once.

| `Sam3SegmentationConfig` | Default | Meaning |
| --- | --- | --- |
| `encoder`, `decoder` | none | `sam3_encoder.onnx` and `sam3_decoder.onnx` from the tool, or TensorRT engines built from them for this GPU |
| `prompt_table` | none | `sam3_prompts.bin` from the same export |
| `prompts` | none | 1 to 16 `Sam3Prompt`s: `{.text = "cup"}`, `{.boxes = {{.xyxy = {x1, y1, x2, y2}}}}` in fractions of the image, or both |
| `prompt_control` | none | A `Sam3PromptControl` that changes the prompts while the graph runs |
| `score_threshold` | `0.5` | Instances need a score above this, as in SAM 3's own predictor |
| `mask_threshold` | `0.5` | Masks keep the pixels whose probability exceeds this |
| `max_instances` | `16` | The most instances per image, 1 to 256; it sizes the outputs |
| `output_masks` | `true` | Declare the `masks` output; the label image alone holds every instance where it is the best one |
| `max_width`, `max_height` | `1920`, `1080` | The largest image, which sizes the output buffers |
| `fp16` | `true` | Build the engines with FP16 kernels allowed |
| `cuda_graphs` | `true` | Let HoloInfer replay the inference as CUDA graphs |
| `build_workspace_bytes` | 1 GiB | TensorRT's workspace limit while it builds an engine; SAM 3's FP32 encoder needs 4 GiB |

The constructor reads the prompt table and refuses a prompt whose text is not in it. Given ONNX
files, HoloInfer builds TensorRT engines and caches them beside the files, named after the GPU, the
TensorRT version and the settings, so later runs load them in seconds. Building SAM 3's engines
takes under a minute on an RTX 5090 and about 8 minutes on a Jetson AGX Orin. The cache is not
keyed on the files' contents: after exporting again, delete the `<name>.*.engine.*` files
(`prepare_model.py` does). `start()` fails, naming the problem, when the encoder and decoder do
not fit each other, the prompt table or the expected shapes.

**Call `holoscan::sam3::prepare_engines(config)` before running the graph.** It builds and caches
the engines. Some TensorRT releases change the SIGINT and SIGTERM handlers while they build an
engine, which the SDK's managed signal handling accepts only before a run starts; TensorRT also
starts threads of its own, after which the SDK no longer takes over SIGINT and SIGTERM by itself.
Create the process's `holoscan::ManagedShutdownSignals` at the start of `main()`, while it is the
only thread, as below.

The operator uses HoloInfer directly rather than the SDK's `InferenceOp`: it runs one model once
and the other once per prompt, lets the decoder read the encoder's outputs where they are, and
keeps each image's capture time and frame id on every output.

### `Sam3OverlayOp`

In the `holoscan::sam3_overlay` target, which needs the SDK's Visualization component.

| Port | Direction | Type |
| --- | --- | --- |
| `labels`, `scores`, `boxes`, `prompt_ids`, `count` | in | `Sam3SegmentationOp`'s outputs; the label image in GPU memory |
| `text` | out | For a HoloViz text layer: per instance, a shadow row and a `"<prompt> 0.87"` row at its box. Host memory |
| `overlay` | out | RGBA8 `[H, W, 4]` for a HoloViz image layer: each instance in its own color, translucent inside and opaque on its outline, over a transparent background. GPU memory |

| `Sam3OverlayConfig` | Default | Meaning |
| --- | --- | --- |
| `max_width`, `max_height` | `1920`, `1080` | The largest image |
| `max_instances` | `16` | The segmenter's `max_instances` |
| `prompt_names` | none | The prompts' names by index (`display_name()`); others show as `prompt N` |
| `prompt_control` | none | The segmenter's control, whose prompts then name the instances |
| `fill_alpha` | `110` | The opacity of an instance's inside |

## Prompts

A prompt names a concept by a phrase, marks instances of it with exemplar boxes, or both; SAM 3
then finds every instance of the concept, not only those marked. A negative box
(`.positive = false`) marks something that is not an instance. A prompt of boxes alone uses the
phrase "visual", as SAM 3's own predictor does. The decoder takes up to four boxes per prompt.

SAM 3 also takes point prompts, but through its tracker's interactive decoder, a separate head that
segments one object rather than a concept. That head belongs with video tracking, and is not
exported here.

The prompt table (`<holoscan/sam3/prompts.hpp>`) holds each phrase's text embedding: SAM 3's text
encoder output, projected to the decoder's width, for 32 tokens. `read_prompt_table()` and
`write_prompt_table()` read and write it; the header describes the format. To change prompts while
the graph runs, give the operator a `Sam3PromptControl` and keep it:

```cpp
auto control = std::make_shared<s3::Sam3PromptControl>(s3::read_prompt_table(table));
config.prompt_control = control;
// ... while it runs, from any thread:
control->set({{.text = "cup"}, {.text = "bottle"}});   // Throws for a phrase not in the table.
```

The operator takes the new prompts from its next image on; `prompt_ids` then index the new list.

## What it computes

SAM 3 is split where an image's features meet a prompt. The **encoder**, SAM 3's vision backbone
(a ViT) and its feature pyramid, about 450 M parameters, takes the image and gives three feature
maps. The **decoder**, about 32 M parameters, takes the feature maps and one prompt: the prompt's
text embedding and its boxes go through SAM 3's geometry encoder, then its DETR-style encoder and
decoder with 200 queries and a presence token, its scoring and its mask head. It gives each query
mask logits at 288x288, a box, and a score: the query's probability times the presence
probability. The decoder reads the encoder's outputs in place, with no copy.

Per image, a CUDA kernel stretches the image to the encoder's 1008x1008 input, as SAM 3's own
processor does: each input pixel is a bilinear sample of the image with pixel centers aligned and
edges clamped, in RGB order, divided by 255 and normalized to [-1, 1]; alpha is ignored. (SAM 3's
processor also antialiases when it downscales; [Accuracy](#accuracy) has what the model achieves
without.) After each prompt's decoder run, a kernel keeps the queries scoring above the threshold,
at most `max_instances`, with their low-resolution mask logits. A last pass ranks every prompt's
instances together, best first (ties by prompt, then query), keeps `max_instances`, scales their
boxes to the image, and upsamples their masks to the image bilinearly with half-pixel centers
(PyTorch's `interpolate(mode="bilinear", align_corners=False)`, as SAM 3's postprocessing), a
pixel being inside where the logit exceeds the threshold's. Each label image pixel names the best
instance covering it. Nothing is copied to the host.

The per-pixel arithmetic is in [`pixel_math.hpp`](include/holoscan/sam3/pixel_math.hpp), shared by
the kernels and the host references in [`reference.hpp`](include/holoscan/sam3/reference.hpp). The
kernels are compiled without fused multiply-adds, so the tests compare them with the references
bit for bit.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-sam3 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::sam3)
```

```cpp
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/sam3/sam3_segmentation_op.hpp>

namespace s3 = holoscan::sam3;

// First thing in main(), while it is the only thread.
auto signals = holoscan::ManagedShutdownSignals::create();

const s3::Sam3SegmentationConfig config{
    .encoder = "/data/sam3/sam3_encoder.onnx",
    .decoder = "/data/sam3/sam3_decoder.onnx",
    .prompt_table = "/data/sam3/sam3_prompts.bin",
    .prompts = {{.text = "cup"}, {.boxes = {{.xyxy = {0.1F, 0.2F, 0.3F, 0.5F}}}}}};
auto sam3 = graph.op<s3::Sam3SegmentationOp>("sam3", config);
graph.add_flow(camera->image, sam3->image);             // Output<holoscan::schema::ImageT>
graph.add_flow(sam3->masks, consumer->masks);           // Input<holoscan::Tensor>
graph.add_flow(sam3->labels, consumer->labels);
graph.add_flow(sam3->scores, consumer->scores);
graph.add_flow(sam3->boxes, consumer->boxes);
graph.add_flow(sam3->prompt_ids, consumer->prompt_ids);
graph.add_flow(sam3->count, consumer->count);

// The outputs are in GPU memory: bind their device when compiling.
holoscan::CompileOptions options;
for (const char* port : {"masks", "labels", "scores", "boxes", "prompt_ids", "count"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "sam3", .output_port = port, .device = holoscan::DeviceId{0}});
}
const auto plan = holoscan::compile(graph, std::move(options));
s3::prepare_engines(config);   // Before the run: see above.
const int status = holoscan::run(plan);   // Adopts `signals`.
```

### Lossless or real time

`add_flow()` without options gives a blocking edge: every image is segmented and its instances
delivered, in order, at the queue depth the input declares (two). When a consumer is still busy
with earlier images, `emit()` returns `kBackpressured` with nothing delivered, and the operator
cannot send that image's outputs again, so the run fails: a lossless graph paces its source to
what its consumers keep up with, as the demo's test modes do. For a live source, connect the input
and every output as a depth-1 drop-oldest edge
(`{.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U}`): the operator then
never waits for a slower consumer, and takes the newest image. The demo shows the HoloViz wiring.

The kernels need no Holoscan graph: `<holoscan/sam3/kernels.hpp>` has `launch_preprocess()`,
`launch_select()`, `launch_finalize()` and `launch_overlay()`. For tests,
`<holoscan/sam3/stub_model.hpp>` writes a stand-in encoder, decoder and prompt table with the
tensors of the export and a known response, and `<holoscan/sam3/synthetic_scene.hpp>` renders a
scene with known shapes and measures a mask's IoU with them.

## Prepare the model

First request access to <https://huggingface.co/facebook/sam3> and accept the SAM License there,
then make an access token (or log in with `hf auth login`). `prepare_model.py` needs
`--accept-sam-license`, to record that you accept the license for what it writes. It downloads the
checkpoint's files from Hugging Face at a pinned revision, checks each against its pinned SHA-256
digest, exports the encoder and the decoder to ONNX (opset 18, FP32) with PyTorch and Transformers,
embeds the prompts with SAM 3's text encoder, checks the export's inputs and outputs, and writes:

| File | |
| --- | --- |
| `sam3_encoder.onnx`, `sam3_encoder.onnx.data` | The encoder: image float32 `[1, 3, 1008, 1008]`; `fpn0`, `fpn1`, `fpn2` float32 `[1, 256, 288, 288]`, `[1, 256, 144, 144]`, `[1, 256, 72, 72]`. 1.8 GB |
| `sam3_decoder.onnx` | The decoder: `fpn0..fpn2`, `text_features [1, 32, 256]`, `text_mask [1, 32]`, `boxes [1, 4, 4]`, `box_labels [1, 4]`, `box_valid [1, 4]`; `masks [1, 200, 288, 288]`, `boxes_xyxy [1, 200, 4]`, `scores [1, 200]`. 130 MB |
| `sam3_prompts.bin` | The prompt table: "visual", a few defaults (the demo's shapes; person, face, hand, cup, bottle, phone, keyboard, chair, car), and those you add |
| `SAM_LICENSE` | The SAM License, which applies to all of the above |

The tool itself needs only the standard library: the export runs in the interpreter given by
`--python`, which needs `torch`, `torchvision`, `transformers`, `onnx` and `onnxscript`. Install
them in their own virtual environment, not in the CI image, and review what they pull in; the CPU
build of PyTorch is enough. The export takes about 3 minutes and 8 GB of memory. Tested with
`torch` 2.15.0, `torchvision` 0.29.2, `transformers` 5.22.1, `onnx` 1.23.2 and `onnxscript`:

```bash
python3 -m venv ~/sam3-export-venv
~/sam3-export-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
~/sam3-export-venv/bin/pip install transformers onnx onnxscript
export HF_TOKEN=hf_...   # your own token, with access to facebook/sam3
python3 modules/sam3/tools/prepare_model.py --output data/sam3 --accept-sam-license \
  --python ~/sam3-export-venv/bin/python --prompts "apple,red mug" --cache ~/.cache/sam3
```

To add phrases later without exporting again, run it with `--prompts-only --prompts "..."` (or
`--prompts-file` with one phrase per line); it still loads the checkpoint, for the text encoder, so
keep `--cache`. `--force` exports again over an existing model. From the installed module, run
`<prefix>/share/holoscan-sam3/prepare_model.py`; CMake projects find its path in
`holoscan-sam3_PREPARE_MODEL`.

The export follows Hugging Face Transformers' SAM 3 port (Apache-2.0 code) with one change: the
geometry encoder always runs, also for a prompt without boxes, where it contributes its class
token, as Meta's own predictor does. The input size is fixed at 1008x1008, the size SAM 3's
position embeddings are for.

## Security

Treat model files and prompt tables as trusted input. The operator hands the models it is given to
TensorRT, which parses an ONNX model or deserializes an engine; load only models and engines that
you exported or built yourself, or otherwise trust. The export loads the checkpoint with
Transformers from `safetensors` files only, whose digests are pinned.

## Requirements

- Holoscan SDK 5.0 (Core, Sensor I/O and Inference, which brings HoloInfer and TensorRT 10;
  Visualization for the overlay; Testing for the tests), CUDA 13, CMake 4.0 or later and a C++20
  compiler.
- For SAM 3: about 2 GiB of free GPU memory with FP16 engines (7.5 GiB with FP32), access to the
  gated checkpoint, and, for the export, Python 3.12 and the packages above.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/sam3 -B build/sam3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/sam3
ctest --test-dir build/sam3 --output-on-failure
cmake --install build/sam3 --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_SAM3_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module |
| `HOLOSCAN_SAM3_BUILD_OVERLAY` | `ON` (`OFF` when the including project asks only for `Sam3SegmentationOp`) | Build `Sam3OverlayOp`, which needs Visualization |
| `HOLOSCAN_SAM3_MODEL_DIR` | `data/sam3` | The export the real-model test uses |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

| Test | Needs | Checks |
| --- | --- | --- |
| `sam3_kernels` | GPU | The kernels against the host references, bit for bit: the preprocessing in every encoding, down- and upscaled, with padded rows; the selection and finalization of made-up decoder outputs at SAM 3's sizes and others, with tied and NaN scores, more instances than fit, prompts with none, overlapping masks, with and without the masks output; the overlay |
| `sam3_operators` | GPU | Both operators through the SDK's operator test harness, with the stub model and TensorRT, every output against the host reference of the whole pipeline: every encoding, from GPU, host and pinned memory, text and box prompts, more instances than fit, other thresholds, without masks, changing image sizes, prompts from a control, FP16 engines, provenance on every output, unusable images, settings, prompts and models; the overlay's image and labels |
| `sam3_real_model` | GPU, the export; skipped without it | SAM 3 in FP16 on the synthetic scene: "circle", "blue rectangle" and "triangle" each find their shape, and an exemplar box around the circle finds it, with a mask IoU of at least 0.9 |
| `sam3_host` | Nothing | Prompts and the prompt table file; a prompt's decoder inputs; the mask sampling, ranking and box arithmetic against direct evaluations; the instance and overlay references; the synthetic scene; the stub model files, and each stub prompt finding its shape |
| `sam3_prepare_model` | Python | `prepare_model.py` without PyTorch or the network: the license and token requirements, pinned digests, a wrong digest, the cache, exports of another shape, a failed or missing exporter, prompts added to an existing export, and reuse of prepared files |

`sam3_host` and `sam3_prepare_model` are also a project of their own in [`tests/host/`](tests/host/),
which the hosted CI job builds and runs. [`tests/installed_consumer/`](tests/installed_consumer/)
is a separate project that uses the installed package: a graph that runs both operators with the
stub model on synthetic frames and checks every overlay against the reference, and a run of the
installed `prepare_model.py`.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/sam3 package holoscan-sam3 --language cpp --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes `holoscan-sam3_1.0.0_<arch>.deb`. It
installs to `/usr` and depends on `holoscan (>= 5.0.0)`. It holds no model.

## Accuracy

SAM 3 through `Sam3SegmentationOp`, on the synthetic scene at 1280x720 (a red circle, a blue
rectangle and a yellow triangle on gray, drifting), mask IoU of each prompt's best instance with the
true shape:

| Prompts | Engines | IoU, 200 frames: smallest (mean) |
| --- | --- | --- |
| "circle", "blue rectangle", "triangle" | FP16 | 0.987 (0.994) |
| The same | FP32 | 0.988 (0.994) |
| "circle" alone | FP16 | 0.993 (0.995) |
| An exemplar box around the circle, no text | FP16 | 0.993 |

The scores are 0.92 to 0.97; "shape", which names none of them, finds nothing. FP16 and FP32 agree
to three places, so FP16 is the default. The synthetic scene is easy by design: these numbers show
that the export, the TensorRT engines and the pre- and postprocessing reproduce SAM 3, not how well
SAM 3 segments real scenes, for which see the SAM 3 paper.

## Performance

SAM 3 at 1008x1008, measured with SDK 5.0.0, CUDA 13.2 and TensorRT 10.16 in the repository's CI
image:

| | RTX 5090 (x86_64) | Jetson AGX Orin (JetPack 7.2) |
| --- | --- | --- |
| Encoder, TensorRT alone, FP16 (`trtexec`, median) | 19.7 ms | 279 ms |
| Decoder, per prompt, FP16 | 3.2 ms | 33 ms |
| Encoder, FP32 | 84 ms | 920 ms |
| Capture to instances on the host, 1280x720, FP16, one prompt, mean (p99) | 22.8 ms (25.0) | 339 ms (384) |
| The same, three prompts | 29.5 ms (33.8) | 407 ms (455) |
| The same, three prompts, FP32 | 142 ms | not measured |
| GPU memory of a run, FP16 (FP32) | 1.6 GiB (7.3 GiB) | 1.0 to 2.3 GiB |

**On a Jetson AGX Orin, SAM 3 runs at about 3 images per second.** The encoder, a ViT of about
450 M parameters at 1008x1008, takes 279 ms in FP16, and each prompt 33 ms more; that suits
occasional or offline segmentation there, not a live camera. On the RTX 5090 it runs at over 30
images per second with a few prompts.

The RTX 5090 was shared with an inference server while measured; the Orin was otherwise idle. The
GPU memory is the drop in free memory during a run: the engines' weights (about 0.9 GiB in FP16),
their activations (about 170 MiB each in FP16, 3.5 GiB for the FP32 encoder), and the operators'
buffers. On the Orin, whose GPU shares the system's memory, it moves with everything else running,
hence the range. Building the FP16 engines takes under a minute on the RTX 5090 and about 8 minutes
on the Orin. On Jetson Thor, CI runs the stub model only, as no model is downloaded there; SAM 3's
latency on Thor is not measured.

## Toward tracking

Video tracking (carrying instances from frame to frame with SAM 3's memory bank and its tracker's
mask decoder) is a separate operator's job. This one leaves room for it: the encoder is exported on
its own, so a tracker can share its features; the outputs are what a tracker's per-frame
detections are, ranked instances with masks, boxes and scores, to which a tracker adds object ids;
and the prompts are already a run-time control.

## Licenses

Apache-2.0. SAM 3's code and weights are not covered by it: they are under the SAM License; see
the notes at the top and [NOTICE](NOTICE). The export follows Hugging Face Transformers' SAM 3 port,
which is Apache-2.0. HoloInfer and the rest of the SDK are Apache-2.0; TensorRT and CUDA are under
NVIDIA's own terms.
