# Standalone SAM3 + Crop1.2 Tactile Inference

This directory is an inference-only deployment for the current TouchAnything
crop1.2 model. It accepts an image, video, or ordered image directory and runs:

```text
input frames
  -> SAM3 anonymous hand tracking
  -> exact crop1.2 affine and left-hand canonical flip
  -> frozen DINOv3 H+/16 blocks 8/16/24/32
  -> ReZero + FullGrid32 tactile head
  -> 13,614 canonical pressure vertices
```

It does not import the training module, dataset loader, HaMeR detector, MANO
decoder, or evaluation code. SAM3 is deliberately launched in a separate
Python environment and communicates through `bboxes.jsonl`.

## Configure

Edit `config.server.json`, or copy it elsewhere and set:

```bash
export TACTILE_INFERENCE_CONFIG=/path/to/config.json
```

All external artifacts are explicit:

- compact crop1.2 `best_loss.ckpt`
- local DINOv3 H+/16 weights
- canonical subdiv OBJ and palm-face JSON
- SAM3 Python executable, tracker script, and checkpoint

Relative paths in the JSON are resolved relative to that JSON file. Therefore
this directory can be copied to another location without preserving the source
repository layout. The SAM3 code and weights are intentionally not duplicated:
set `sam3.tracker`, `sam3.python`, and `sam3.checkpoint` to their locations in
the target installation.

The tactile environment needs a CUDA-compatible PyTorch plus the packages in
`requirements.txt`. The configured SAM3 environment must already be able to
run the configured `track_video.py` and import its local `sam3` package.

Validate both environments and all paths before inference:

```bash
./run.sh --doctor
```

## Image

For a known right hand:

```bash
./run.sh \
  --input /path/to/image.jpg \
  --output /path/to/output/image_run \
  --handedness right
```

For a bare-hand SAM prompt:

```bash
./run.sh \
  --input /path/to/image.jpg \
  --output /path/to/output/bare_image \
  --prompt-preset bare \
  --handedness right
```

The default `gloved` preset resolves to your existing primary prompt
`gloved hand`; `bare` resolves to `bare human hand`. The configured tracker
loads the complete candidate/verifier lists from the `prompt_presets.json`
beside `track_video.py`. Override only the primary text for one run with:

```bash
./run.sh \
  --input /path/to/image.jpg \
  --output /path/to/output/custom_prompt \
  --sam-prompt "hand wearing a black tactile sensing glove" \
  --handedness right
```

## Video

```bash
./run.sh \
  --input /path/to/video.mp4 \
  --output /path/to/output/video_run \
  --handedness auto
```

`auto` uses egocentric screen order for two queries: the screen-left track is
canonicalized as the right hand and the screen-right track as the left hand.
For one anonymous track it uses `tactile.single_hand_default` from the config.
This is only crop canonicalization, not an identity input to the model.

When handedness is unknown, emit both canonical orientations:

```bash
./run.sh \
  --input /path/to/video.mp4 \
  --output /path/to/output/video_both \
  --handedness both
```

For terminal-assisted query selection, first let SAM3 finish and then assign
each numbered track as left, right, both, or skip:

```bash
./run.sh \
  --input /path/to/video.mp4 \
  --output /path/to/output/video_interactive \
  --handedness interactive
```

The script writes `sam3/handedness_preview.jpg` before prompting. Open that
image from the shared filesystem and answer `l`, `r`, `b`, or `s` in the
terminal. Interactive mode requires an attached terminal and is unsuitable
for `nohup`; non-interactive jobs should use the explicit modes instead.

Reuse an existing SAM result without loading SAM3 again:

```bash
./run.sh \
  --input /path/to/video.mp4 \
  --output /path/to/output/reused_boxes \
  --sam-bboxes /path/to/bboxes.jsonl \
  --handedness right
```

Use `--overwrite` to replace an existing output directory. Use `--no-render`
when only numeric pressure arrays are needed.

## Outputs

The root output contains:

- `input_frames/`: exact frames shared by SAM3 and tactile inference
- `sam3/`: masks, track audit, previews, and `bboxes.jsonl`
- `inference_manifest.json`: resolved model/query provenance

Each `query_*` directory contains:

- `pressure_raw.npy`: raw `[frames, 13614]` predictions
- `pressure_palm_masked.npy`: deployment pressure output
- `bbox_tight.npy` and `bbox_crop12.npy` (the actual rectangular model FOV)
- `detected.npy`
- `tactile.png` / `combined.jpg` for an image
- `combined.mp4` and rendered frame directories for a video

Missing SAM3 boxes produce zero tactile pressure. Temporal smoothing is off by
default and, when enabled in `render.temporal_alpha`, affects visualization
only; saved pressure arrays always contain unsmoothed model predictions.
