# YOLO26 Detection

The demo of the [holoscan-yolo26](../../modules/yolo26/) module. A source (synthetic frames with
known detections, JPEG and PPM files, or a live V4L2 webcam) feeds `Yolo26DetectionOp`, or with
`--task segment` `Yolo26SegmentationOp`, whose detections (and masks) a reporter checks, prints and
times, and can draw onto the images, which it writes as PPM files. A HoloViz window can show the
frames with `Yolo26OverlayOp`'s boxes and labels, or `Yolo26SegmentationOverlayOp`'s masks, boxes
and labels, drawn over them, and HoloViz can render the same offscreen, where every render is
checked. It uses the installed module the way any other project would:
`find_package(holoscan-yolo26)`.

> **The models are AGPL-3.0.** YOLO26 and YOLO26-seg weights and the Ultralytics software that exports them are
> licensed under the
> [GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
> (or an Ultralytics Enterprise License). Neither is part of this repository, which is Apache-2.0
> and contains no Ultralytics code or weights. They are downloaded and exported separately, on your
> machine, by the module's `prepare_model.py` (see "Prepare the model" below); the terms that apply
> to using and distributing the model and anything built with it are yours to review. The tests and
> the `ci` mode need neither: they use the module's stand-in models.

```mermaid
flowchart LR
    S[Synthetic frames, JPEG and PPM files, or a webcam] -->|image| D[Yolo26DetectionOp, or Yolo26SegmentationOp]
    D -->|boxes, scores, class_ids, count; masks, label_image| R[Report: truth, expectations, latency, PPM files]
    S -->|image, with --image-dir| R
    D -->|with --display| O[Yolo26OverlayOp, or Yolo26SegmentationOverlayOp] --> V[HoloViz window, or offscreen]
    S -->|with --display| V
    V -->|offscreen| C[Render check]
```

## Demo

Videos of the demo on synthetic input and on real input go in [tests/video](tests/video).

## Requirements

- holoscan-yolo26, installed (see [its README](../../modules/yolo26/README.md)), and what it needs:
  Holoscan SDK 5.0 with Core, Sensor I/O, Inference and Visualization (Testing for the tests),
  CUDA 13 with nvJPEG, CMake 4.0 or later and a C++20 compiler.
- For the `camera` mode: a V4L2 webcam offering YUYV, and a display.
- For the real model only: the model prepared as below.

## Prepare the model

The `demo`, `files` and `camera` modes and the real-model tests need YOLO26n exported to ONNX with
its end-to-end head (and the `segmentation` modes YOLO26n-seg), and the COCO validation image `000000039769.jpg` (subject to the COCO terms of
use, on the Terms of Use page at [cocodataset.org](https://cocodataset.org/)). The module's
[`prepare_model.py`](../../modules/yolo26/tools/prepare_model.py) downloads and checks both, and
exports the model with the Ultralytics `yolo` command, which you install in a virtual environment of
its own; its README explains the steps and their risks. From the repository root:

```bash
python3 -m venv ~/ultralytics-venv
~/ultralytics-venv/bin/pip install ultralytics onnx onnxslim onnxruntime
python3 modules/yolo26/tools/prepare_model.py --output data/yolo26_detection \
  --yolo ~/ultralytics-venv/bin/yolo
python3 modules/yolo26/tools/prepare_model.py --output data/yolo26_detection --task segment \
  --yolo ~/ultralytics-venv/bin/yolo
```

This writes `data/yolo26_detection/yolo26n.onnx`, `yolo26n.labels.txt`, `yolo26n-seg.onnx`,
`yolo26n-seg.labels.txt` and `images/000000039769.jpg`, where the modes and the tests look for them. Copy your own JPEG or PPM
images into `data/yolo26_detection/images` for the `files` mode.

## Build and run

From the repository root, after installing the module into `install/`:

```bash
cmake -S applications/yolo26_detection -B build/yolo26_detection -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/yolo26_detection
```

The modes in [metadata.json](metadata.json) are these commands, where `<holohub_data_dir>` is the
repository's `data` folder:

| Mode | Behavior |
| --- | --- |
| `demo` (default) | YOLO26n on the prepared images, 300 frames looped, printing each image's detections once and the latency. |
| `files` | YOLO26n on each image in `data/yolo26_detection/images` once, printing the detections and writing each image with its boxes drawn. |
| `camera` | YOLO26n on a live webcam, with the detections drawn over it in a HoloViz window. |
| `segmentation` | YOLO26n-seg on each image in `data/yolo26_detection/images` once, printing the detections with each mask's size and writing each image with its masks and boxes drawn. |
| `segmentation_camera` | YOLO26n-seg on a live webcam, with the masks and detections drawn over it in a HoloViz window. |
| `ci` | 30 synthetic BGR8 frames through TensorRT with the stub segmentation model, every detection checked against the frame's known ones, every mask against its rectangle and the label image against the masks. Needs no download; CI runs it. |

```bash
build/yolo26_detection/yolo26_detection --model data/yolo26_detection/yolo26n.onnx \
  --labels data/yolo26_detection/yolo26n.labels.txt --source files \
  --images data/yolo26_detection/images --loop true --count 300 --period-ms 10   # demo mode
build/yolo26_detection/yolo26_detection --task segment --model stub --count 30 --period-ms 5 \
  --encoding bgr8                                                                      # ci mode
```

The `demo` mode prints, for the COCO image:

```text
yolo26_detection: 000000039769.jpg: 4 detections
  cat (15) 0.912 [344.0, 24.6, 640.0, 374.9]
  cat (15) 0.898 [6.9, 55.1, 317.2, 466.1]
  remote (65) 0.857 [40.4, 74.0, 175.9, 118.6]
  couch (57) 0.297 [0.8, 0.6, 640.0, 480.0]
yolo26_detection: frames=300/300 detections=1200 latency_ms: mean=1.390 p50=1.268 p99=2.043 max=25.490
yolo26_detection: verdict=PASS
```

The latency is the time from a frame's capture by the source to its detections on the host,
including the JPEG decode; the first frame's includes the warm-up. The first run builds the TensorRT
engine from the ONNX file, which takes 10 to 30 s, and caches it beside the file; later runs start
in seconds.

`--model stub` writes and uses the module's stand-in model: an ONNX network with the inputs and
outputs of an end-to-end YOLO26 export that "detects", from the pixels, red (class 0) and blue
(class 1) regions, so the synthetic frames' red and blue rectangles. It is written to the working
directory, where its engine is cached too. With it, each synthetic frame's detections are checked
against the rectangles: each box edge within a model input pixel or so (`2 / r + 1` image pixels),
the exact score, and nothing extra.

Each run prints the frame count, the detections, the latency and, as they apply, the truth check and
the expected detections found. A gap in the frame ids fails the run (but for the camera, see
[Lossless or real time](#lossless-or-real-time)), as does a run over files with
nothing expected and no detection at all, which most likely means a wrong model, input size or
threshold. Then comes a `verdict`: `PASS`, `FAIL`, or `STOPPED` when an interrupt or a closed window
ends the run early. It exits 0 on `PASS`, or on `STOPPED` for a camera run without `--count`; 2 for
invalid options, a missing camera among them; else 1.

### Segmentation

`--task segment` runs `Yolo26SegmentationOp` with a YOLO26 segmentation model, or with
`--model stub` the module's stand-in segmentation model, whose masks are the silhouettes of the
synthetic frames' red and blue rectangles. It keeps 32 detections per image unless
`--max-detections` says otherwise, and `--mask-threshold` sets the mask probability a pixel needs
(default 0.5). On the COCO image, the `segmentation` mode prints:

```text
yolo26_detection: 000000039769.jpg: 5 detections
  cat (15) 0.904 [10.7, 54.1, 318.1, 466.1] mask 52727 px
  cat (15) 0.845 [340.7, 24.4, 640.0, 373.8] mask 60664 px
  remote (65) 0.835 [40.1, 72.8, 176.7, 118.0] mask 4518 px
  couch (57) 0.435 [0.0, 1.8, 640.0, 477.4] mask 183221 px
  remote (65) 0.266 [334.7, 77.3, 370.2, 184.2] mask 2165 px
```

With the stub model, each synthetic frame's masks are checked too: each must overlap its
rectangle by an IoU of at least 0.9 (0.8 for a rectangle under 24 pixels a side, whose edges the
letterbox blurs more), and the label image must name, at each pixel, the first detection whose mask
covers it. The run prints `masks: missed=... min_iou=... label_errors=...`. Written images
(`--image-dir`) show the masks half-transparent under the boxes.

`--display offscreen` has HoloViz render the frames with the overlay into GPU memory instead of a
window, for synthetic RGB8 frames with the stub model, in either task. Each render is checked: it
must show some frame of the run, found from where its boxes are, with each rectangle's box in its
class's color at its left edge and, at its center, the frame's color, blended with the class color
at the masks' opacity when segmenting. The run prints `renders=N renders_matched=N`, and fails
unless every render matched. With `--image-dir`, the renders are written there too, as
`render_NNNN.ppm`.

`--expect FILE` lists detections that named image files must yield, as lines of
`image,class_id,x1,y1,x2,y2`, each matched by a detection of the class with an IoU of at least
`--expect-iou` (default 0.8). `--image-dir DIR` writes each frame with its boxes drawn as
`DIR/detections_NNNNNN.ppm`. `--encoding bgr8` decodes the files as BGR8, `--fp16 true` builds an
FP16 engine, and `--score-threshold` sets the threshold. Run the executable with `--help` for every
option. `-DYOLO26_DETECTION_ENABLE_VISUALIZATION=OFF` builds it without HoloViz.

### Lossless or real time

Synthetic frames and files are lossless: every frame is detected and reported, in order, over
blocking edges. They go one at a time: the source sends a frame only once the reporter is done with
the one before, so none is lost however long the reporter takes, to write an image say, and
`--period-ms` is the shortest time between frames. The reporter's count of frames done is the
signal; it cannot reach the source as data, over an edge, because the SDK refuses a graph with a
cycle.

The camera runs in real time. The camera to the detector, and the detector to the reporter, the
overlay and the window, are depth-1 drop-oldest edges: a full queue drops its oldest sample rather
than refusing the new one. The detector never waits for a slower consumer, takes the newest frame
when the camera is faster, and a slow consumer finds the newest detections waiting when it is done.
Frames passed over are counted as `skipped`, not as gaps, and a run with `--count N` ends with frame
`N`'s detections. The reporter's images (with `--image-dir`) and the window's video keep the four
newest frames instead of one, so that a frame is still queued when its detections arrive, a
detector's latency later, to be paired with them. The window, with `--display true`, is real time
in either mode.

### Webcam

Allow the window to open, then run the `camera` mode:

```bash
xhost +local:
build/yolo26_detection/yolo26_detection --model data/yolo26_detection/yolo26n.onnx \
  --labels data/yolo26_detection/yolo26n.labels.txt --source camera --device /dev/video0
```

The camera is opened when the run starts and closed when it stops; frames that arrive while the
detector is busy are dropped, so each detection is of the newest frame (see
[Lossless or real time](#lossless-or-real-time)). The window shows each frame
with its boxes, a color per class, and a `<class> 0.87` label above each box. The camera must offer
YUYV at the requested size (default 640x480, or for example `--width 1280 --height 720`); otherwise
the run stops with the sizes it offers. To use another camera, list them with
`v4l2-ctl --list-devices` and pass one with `--device /dev/videoN`. The run lasts until Ctrl+C or the
window is closed. `--display true` shows synthetic frames too, and `--display false` runs the camera
without a window.

## Performance

On an RTX 5090 (driver 595.91.07, shared with another process) with SDK 5.0.0, CUDA 13.2 and
TensorRT 10.16, for YOLO26n at 640 on the 640x480 COCO image, it takes a median of 1.27 ms from
capture to detections on the host with the FP32 engine, and 0.96 ms with FP16, the source's JPEG
decode included. The module's README compares detection and segmentation (`--task segment` with
YOLO26n-seg) on the RTX 5090 and on a Jetson AGX Orin. The module's README breaks down the GPU work per frame, measured with Nsight
Systems over 300 frames, tracing CUDA graph nodes, from:

```bash
nsys profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o yolo26 \
  build/yolo26_detection/yolo26_detection --model data/yolo26_detection/yolo26n.onnx \
  --source files --images data/yolo26_detection/images --loop true --count 300 --period-ms 10
```

## Tests

With `-DBUILD_TESTING=ON`, `ctest` checks:

- The demo with the stub model on synthetic frames and on JPEG (baseline and progressive) and PPM
  files that a fixture writes: every encoding, GPU and host memory, wide, tall and small frames,
  another input size, FP16 engines, engine files, image output, expected detections, and the exit
  status and verdict of failed, stopped and invalid runs, a run without managed signal handling, a
  missing camera, and the display graph.
- Segmentation with the stub segmentation model, in the same encodings, memory, shapes and input
  sizes, with every mask and the label image checked; the masks drawn into files and printed for
  image files; too small a capacity and too high a mask threshold failing the run; and the
  segmentation display graph.
- Offscreen HoloViz renders, detecting and segmenting, each render checked against the frame it
  shows.
- The real-time edges with a consumer ten times slower than the source: the run completes without a
  failure, the detector keeps the source's pace, and the consumer gets the newest detections,
  ending with the last frame's.
- With the real model, the two cats and the remote that the Ultralytics predictor finds in the COCO
  image, in RGB8 and BGR8, with an FP16 engine and looped; and with YOLO26n-seg, the cats, the
  remote and the couch that its predictor finds, with masks of the sizes it gives, with FP32 and
  FP16 engines. These run when the models have been prepared in `data/yolo26_detection`, and are
  reported as skipped otherwise; point them elsewhere with `-DYOLO26_DATA_DIR=/path/to/prepared`.
- Without a GPU: the expectation and label files, image headers, PPM files and box drawing; and,
  also built on its own from `tests/host/` by the hosted CI job, the webcam's YUYV conversion.

The module's own suite tests the kernels and the operators.

## Licenses

Apache-2.0. The YOLO26 models are not covered by it; see the note at the top. The module's README
lists the licenses of what it uses.
