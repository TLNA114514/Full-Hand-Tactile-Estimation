<div align="center">

# EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video

<a href="https://egotactile.github.io/"><img src="https://img.shields.io/badge/Project-Page-brightgreen" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/icml-2026-submission/EgoTactile/blob/main/README.md"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow" alt="EgoTactile Dataset"></a>

</div>

<p align="center">
  <img src="assets/visual_1.gif" width="360" alt="Dynamic hand pressure generation 1">
  <img src="assets/visual_2.gif" width="360" alt="Dynamic hand pressure generation 2">
  <br/>
  <img src="assets/visual_3.gif" width="360" alt="Dynamic hand pressure generation 3">
  <img src="assets/visual_4.gif" width="360" alt="Dynamic hand pressure generation 4">
  <br/>
  <em>EgoPressureDiff generates dynamic hand pressure from a single egocentric RGB video.</em>
</p>

---

## Overview

Estimating dynamic full-hand pressure from egocentric video is crucial for VR/AR, human-robot interaction, and dexterous manipulation. Prior pixel-aligned regression methods struggle with severe self-occlusion and physical ambiguity (e.g., visually identical objects with different weights), often yielding conservative or blurred predictions.

We present EgoTactile, a benchmark covering 63 everyday 3D objects with synchronized pressure, and introduce EgoPressureDiff, a latent video diffusion framework. Our key design leverages a pre-trained "world model" prior augmented by a multi-modal conditioning pipeline:

- SVD Backbone: adapts Stable Video Diffusion to generate temporally coherent pressure sequences from video input.
- PIFR Layer: a Physically-Informed Feature Rectification module that explicitly calibrates feature statistics using text metadata (e.g., object weight, material) to resolve force ambiguity.
- Prototype & Mask Conditioning: injects anatomical topology priors and spatial hint masks to guide denoising under occlusion.

On the EgoTactile benchmark, EgoPressureDiff achieves state-of-the-art performance, boosting Volumetric IoU by +12.4% and reducing Center-of-Pressure (CoP) Error by >50% over transformer baselines. It also demonstrates robust zero-shot generalization to bare-hand scenarios.

<p align="center">
  <img src="assets/egopressdiff_pipeline.png" width="360" alt="Dynamic hand pressure generation 1">
  <img src="assets/PIFR_layer.png" width="360" alt="Dynamic hand pressure generation 2">
  <br/>
  <em>Framework overview of EgoPressureDiff and PIFR layer.</em>
</p>


---

## Setup

### Environment

```bash
conda create -n egoPressurediff python=3.10 -y
conda activate egoPressurediff

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.5.1+cu124 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Pretrained Weights

To run **EgoPressureDiff**, you need to download the weights for both the video generation backbone (SVD) and the text encoder (OpenCLIP) used for the PIFR layer.

### 1. Stable Video Diffusion (Backbone)
**Repository:** `stabilityai/stable-video-diffusion-img2vid-xt`

*Description: The foundational generative backbone adapted for pressure estimation.*

#### Option A: Hugging Face
```bash
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
```

*Tip: If you have connection issues, set the mirror endpoint:*

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### Option B: ModelScope

```bash
modelscope download --model stabilityai/stable-video-diffusion-img2vid-xt
```

---

### 2. CLIP Text Encoder (For PIFR Layer)

**Repository:** `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

*Description: Required for encoding text metadata (e.g., object weight, material) in the PIFR layer to resolve physical ambiguity.*

#### Option A: Hugging Face

```bash
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

#### Option B: ModelScope

```bash
# Using OpenGVLab mirror for the LAION-2B ViT-H-14 model
modelscope download --model OpenGVLab/CLIP-ViT-H-14-laion2B-s32B-b79K
```

**Note:** After downloading, move the model folder to `checkpoints/stable-video-diffusion-img2vid-xt/text_encoder/` (create the directory if it does not exist).



### Expected Directory Layout

```text
EgoPressureDiff/
├── core
├── scripts
├── evaluation
├── checkpoints
│   └── stable-video-diffusion-img2vid-xt
│       ├── feature_extractor
│       ├── image_encoder
│       ├── text_encoder
│       ├── scheduler
│       ├── unet
│       ├── vae
│       ├── model_index.json
│       ├── svd_xt.safetensors
│       └── svd_xt_image_decoder.safetensors
├── command_infer.sh
├── command_train.sh
├── inference_hand.py
├── train_hand.py
└── requirement.txt
```

---

## Inference

A runnable example is provided in `command_infer.sh`:

```bash
bash command_infer.sh
```

Notes:
- `--output_dir`: directory to save results  
- `--validation_dir`: **TestSet** directory  
- `--pretrained_model_name_or_path`: path to **SVD** weights  
- `masknet_model_name_or_path`, `unet_model_name_or_path`: paths to EgoPressureDiff checkpoints  
- Increase `--decode_chunk_size` (e.g., `4 -> 8 -> 16`) for smoother temporal results (requires more GPU memory)

The command outputs:
- `result_images/`
- `result_video.mp4`

Optional: re-encode frames into a high-quality MP4:
```bash
cd result_images
ffmpeg -framerate 20 -i frame_%d.png -c:v libx264 -crf 10 -pix_fmt yuv420p /path/result_video.mp4
```
- `-framerate`: FPS
- `-crf`: smaller is higher quality

---

## Training


### Data Preparation

Before training, you need to process the raw collected data to align RGB frames with pressure signals, generate ground-truth heatmaps, and create text description files. Use the provided `raw_to_training.py` script for this conversion.

```bash
python raw_to_training.py \
  --root_dir ./path/to/raw_dataset \
  --output_dir ./path/to/training_dataset \
  --object_info_json object_info.json \
  --subject_info_json subject_info.json
```

**Arguments:**

* `--root_dir`: The root directory of the raw collected data. The script will recursively search for leaf directories containing `video.mp4` and `data.json`.
* `--output_dir`: The destination directory where the processed training data (aligned frames, pressure heatmaps, and text prompts) will be saved.
* `--object_info_json`: Path to the JSON file containing physical attributes of the objects (e.g., weight, surface material), used to generate text descriptions.
* `--subject_info_json`: Path to the JSON file containing participant demographics (e.g., gender, hand length, body fat), used to inject subject-specific details into the text descriptions.


### Generate Training List

After processing the raw data, you need to generate a `motion_list.txt` file containing the absolute paths of all training sequences. This file is required by the dataloader to index the dataset.

Run the provided `get_motion_list.py` script:

```bash
python get_motion_list.py \
  ./path/to/training_dataset \
  ./path/to/training_dataset/motion_list.txt \
  --recursive
```

**Arguments:**
* `input_dir` (Positional): The directory containing your processed training data (this should match the `--output_dir` from the previous step).
* `output_txt` (Positional): The path where the generated list file will be saved.
* `--recursive` (Optional): Add this flag if your training data is organized into subfolders (e.g., grouped by participant ID like `p_001/sequence_name`). If your data is flat, this can be omitted.



### Generate Hint Masks

To train the model with spatial guidance, you need to generate segmentation masks for the hands and objects in your dataset. We utilize [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to automatically label these masks.

#### 1. Installation & Weights

First, clone the Grounded-Segment-Anything repository and install its dependencies. We also need to download the pretrained checkpoints for GroundingDINO and SAM.

```bash
# 1. Clone the repository
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything

# 2. Install dependencies (ensure you have CUDA available)
pip install -q -e segment_anything
pip install -q -e GroundingDINO

# 3. Download Pretrained Weights
# GroundingDINO (Swin-T)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# SAM (ViT-H)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Return to the project root
cd ..
```

#### 2. Run Generation Script

Use the provided `grounded_sam.py` to process your dataset. This script supports multi-GPU processing to accelerate inference.

```bash
python grounded_sam.py \
  --input_dir ./path/to/training_dataset \
  --classes "hand, object" \
  --gpu_ids "0,1,2,3" \
  --box_threshold 0.25 \
  --text_threshold 0.25
```

**Arguments:**
* `--input_dir`: The root directory containing your training images (rgb).
* `--classes`: Comma-separated list of objects to detect. Default is `"hand, object"`.
* `--gpu_ids`: A list of GPU IDs to use for parallel processing (e.g., `"0"` or `"0,1,2,3"`).
* `--grounding_dino_config`: Path to the config file (Default assumes `./Grounded-Segment-Anything/...`).
* `--grounding_dino_ckpt`: Path to the checkpoint (Default assumes `./Grounded-Segment-Anything/groundingdino_swint_ogc.pth`).
* `--sam_ckpt`: Path to the SAM checkpoint (Default assumes `./Grounded-Segment-Anything/sam_vit_h_4b8939.pth`).

**Output:**
The script will generate `_mask.png` files adjacent to the original RGB images in your dataset folders.


### Data Format

```text
training_data/
├── p001-Apple-repeat0000
│   ├── ...
│   ├── 00005_rgb.png
│   ├── 00005_pressure.png
│   ├── 00005_mask.png
│   ├── 00006_rgb.png
│   ├── 00006_pressure.png
│   ├── 00006_mask.png
│   └── ...
├── p001-Banana-repeat0000
│   └── ...
├── p001-BellPepper-repeat0000
│   └── ...
├── ...
└── motion_list.txt
```

- Example `motion_list.txt`:
```text
path/p001-Apple-repeat0000
path/p001-Banana-repeat0000
path/p001-BellPepper-repeat0000
...
```

### Run Training

```bash
bash command_train.sh
```

Key arguments:
- `CUDA_VISIBLE_DEVICES`: GPU IDs (example: `3,2,1,0`)
- `--pretrained_model_name_or_path`: SVD path
- `--output_dir`: checkpoint output directory
- `--data_root_path`: dataset root
- `--data_path`: path to `motion_list.txt`
- `--sample_n_frames`: frames per batch
- `--num_train_epochs`: default is effectively infinite; stop manually when converged

GPU memory note: training typically requires ~**44GB VRAM**.

