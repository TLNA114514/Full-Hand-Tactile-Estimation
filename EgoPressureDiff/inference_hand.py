# -*- coding: utf-8 -*-
"""
inference_hand_v2.py

Changes vs inference_hand_v1.py:
1) Read conditioning data the same way as LargeScaleEgoPressVideos in v2p_dataset.py:
   - per-sample folder contains *_rgb.png, *_mask.png (mask values {0,1,2}), and (optionally) *_pressure.png (GT, not needed for inference)
   - text prompt is read from "<sample_dir>.txt"
   - also loads prototype_{left/right}.png and mask_{left/right}.png to build pressure_mask
2) Replace hand_encoder with text_encoder (CLIPTextModelWithProjection) + tokenizer.
3) Keep the rest of the inference flow compatible with your existing InferencePipeline as much as possible.

NOTE:
- This script assumes your repo's `InferencePipeline` has been updated to accept text conditioning
  (either by accepting `text_prompt` / `text_encoder` / `tokenizer`, or by accepting precomputed embeddings).
  If not yet updated, see the comments near `call_kwargs` for the minimal pipeline-side changes.
"""

import argparse
import os
import random
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from einops import rearrange

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.models.attention_processor import XFormersAttnProcessor
from diffusers.utils.import_utils import is_xformers_available

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from core.modules.mask_net import MaskNet
from core.modules.unet import UNetSpatioTemporalConditionModel
from core.modules.attention_processor import AnimationAttnProcessor
from core.modules.attention_processor_normalized import PIFRAttnProcessor
from core.pipelines.inference_pipeline import InferencePipeline

from core.utils.infer_utils import concat_external_rgb_pressure_with_generated


# -----------------------------
# Utility IO / export
# -----------------------------
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
def export_to_mp4(frames, output_mp4_path, fps: int):
    frames = [np.array(frame) if isinstance(frame, Image.Image) else frame for frame in frames]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_mp4_path, codec="libx264")


def save_frames_as_png(frames, output_path: str, name_fmt: str = "frame_{:05d}.png"):
    os.makedirs(output_path, exist_ok=True)
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    for i, pil_frame in enumerate(pil_frames):
        pil_frame.save(os.path.join(output_path, name_fmt.format(i)))


def filter_files_by_string(directory: str, search_string: str) -> List[str]:
    try:
        files = os.listdir(directory)
        return [f for f in files if search_string in f]
    except Exception as e:
        print(f"[WARN] Error accessing directory {directory}: {e}")
        return []


def preprocess_image_for_model(image_path: str, width: int, height: int, mode: str = "RGB") -> torch.Tensor:
    """
      - RGB: resize (bilinear), float in [-1, 1], shape [H, W, 3]
      - L (mask): NEAREST resize, clamp to [0,2], map to {-1,0,1}, expand to 3ch, shape [H, W, 3]
    """
    try:
        pil_image = Image.open(image_path).convert(mode)

        if mode == "RGB":
            pil_image = pil_image.resize((width, height))
            img = torch.from_numpy(np.array(pil_image)).float()  # [H,W,3]
            img = img / 127.5 - 1.0
            return img

        if mode == "L":
            pil_image = pil_image.resize((width, height), resample=Image.NEAREST)
            mask = torch.from_numpy(np.array(pil_image)).float()  # [H,W], values ideally {0,1,2}
            mask = torch.clamp(mask, 0.0, 2.0)
            mask = mask - 1.0  # 0->-1,1->0,2->1
            mask3 = mask.unsqueeze(-1).repeat(1, 1, 3)  # [H,W,3]
            return mask3

        raise ValueError(f"Unsupported mode={mode}")

    except Exception as e:
        print(f"[WARN] Fail loading the image: {image_path} ({e})")
        if mode in ("RGB", "L"):
            return torch.zeros((height, width, 3), dtype=torch.float32)
        raise


def load_egopress_conditioning(
    sample_dir: str,
    width: int,
    height: int,
    n_sample_frames: int,
    sample_frame_rate: int,
    clip_start: Optional[int],
    prototype_left_path: str,
    prototype_right_path: str,
    mask_left_path: str,
    mask_right_path: str,
    text_prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    rgb_list = sorted(filter_files_by_string(sample_dir, "rgb.png"))
    mask_list = sorted(filter_files_by_string(sample_dir, "mask.png"))
    _ = filter_files_by_string(sample_dir, "pressure.png")  # optional

    if len(rgb_list) == 0 or len(mask_list) == 0:
        raise FileNotFoundError(
            f"No rgb/mask frames found in {sample_dir}. Need at least '*rgb.png' and '*mask.png'."
        )
    if len(rgb_list) != len(mask_list):
        raise ValueError(f"rgb/mask length mismatch: {len(rgb_list)} vs {len(mask_list)} in {sample_dir}")

    video_length = len(rgb_list)

    if n_sample_frames <= 0:
        batch_index = list(range(video_length))
    else:
        clip_length = min(video_length, (n_sample_frames - 1) * sample_frame_rate + 1)
        max_start = max(video_length - clip_length, 0)
        start_idx = 0 if clip_start is None else int(clip_start)
        start_idx = max(0, min(start_idx, max_start))
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, n_sample_frames, dtype=int).tolist()

    hand_side = "left" if "left" in os.path.basename(sample_dir) else "right"

    proto_path = prototype_left_path if hand_side == "left" else prototype_right_path
    pmask_path = mask_left_path if hand_side == "left" else mask_right_path

    prototype = preprocess_image_for_model(proto_path, width, height, mode="RGB")  # [H,W,3] in [-1,1]
    prototype = rearrange(prototype, "h w c -> c h w")  # [3,H,W]

    pmask_pil = Image.open(pmask_path).convert("L").resize((width, height))
    pressure_mask = torch.from_numpy(np.array(pmask_pil)).float() / 255.0
    pressure_mask = pressure_mask.unsqueeze(0).repeat(len(batch_index), 1, 1)  # [F,H,W]
    pressure_mask = rearrange(pressure_mask, "f h w -> f 1 h w")

    if text_prompt_override is not None:
        text_prompt = text_prompt_override
    else:
        tp_path = sample_dir + ".txt"
        try:
            with open(tp_path, "r", encoding="utf-8") as f:
                text_prompt = f.read().strip()
        except Exception as e:
            print(f"[WARN] Fail loading text prompt: {tp_path} ({e})")
            text_prompt = ""

    rgb_frames_list = []
    mask_frames_list = []

    for idx in batch_index:
        rgb_path = os.path.join(sample_dir, rgb_list[idx])
        msk_path = os.path.join(sample_dir, mask_list[idx])

        rgb_t = preprocess_image_for_model(rgb_path, width, height, mode="RGB")  # [H,W,3]
        msk_t = preprocess_image_for_model(msk_path, width, height, mode="L")    # [H,W,3] in {-1,0,1}
        rgb_frames_list.append(rgb_t)
        mask_frames_list.append(msk_t)

    rgb_frames = torch.stack(rgb_frames_list, dim=0)           # [F,H,W,3]
    mask_frames = torch.stack(mask_frames_list, dim=0) # [F,H,W,3]
    rgb_frames = rearrange(rgb_frames, "f h w c -> f c h w")            # [F,3,H,W]
    mask_frames = rearrange(mask_frames, "f h w c -> f c h w")  # [F,3,H,W]

    return dict(
        hand_side=hand_side,
        batch_index=batch_index,
        num_frames=len(batch_index),
        rgb_frames=rgb_frames.unsqueeze(0),               # [1,F,3,H,W]
        mask_frames=mask_frames.unsqueeze(0),     # [1,F,3,H,W]
        prototype=prototype.unsqueeze(0),                 # [1,3,H,W]
        pressure_mask=pressure_mask.unsqueeze(0),         # [1,F,1,H,W]
        text_prompt=text_prompt,
    )


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EgoTactile inference (text-conditioned).")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--validation_dir", type=str, required=True, help="Path to ONE EgoPress sample folder.")

    parser.add_argument("--masknet_model_name_or_path", type=str, required=True)
    parser.add_argument("--unet_model_name_or_path", type=str, required=True)

    parser.add_argument("--sample_n_frames", type=int, default=8)
    parser.add_argument("--sample_frame_rate", type=int, default=4)
    parser.add_argument("--clip_start", type=int, default=None)

    parser.add_argument("--prototype_left_path", type=str, default="./V2P_data/prototype_left.png")
    parser.add_argument("--prototype_right_path", type=str, default="./V2P_data/prototype_right.png")
    parser.add_argument("--mask_left_path", type=str, default="./V2P_data/mask_left.png")
    parser.add_argument("--mask_right_path", type=str, default="./V2P_data/mask_right.png")

    parser.add_argument("--text_prompt", type=str, default=None, help="Override prompt; else read from '<validation_dir>.txt'")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--noise_aug_strength", type=float, default=0.0)

    parser.add_argument("--decode_chunk_size", type=int, default=8)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--frames_overlap", type=int, default=4)

    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # If --text_prompt is empty/None, fall back to reading "<validation_dir>.txt"
    if args.text_prompt is None or str(args.text_prompt).strip() == "":
        tp_path = args.validation_dir + ".txt"
        try:
            with open(tp_path, "r", encoding="utf-8") as f:
                args.text_prompt = f.read().strip()
        except Exception as e:
            print(f"[WARN] Fail loading text prompt: {tp_path} ({e})")
            args.text_prompt = ""
    
    if os.path.exists(args.output_dir):
        print(f"[WARN] output_dir already exists: {args.output_dir}. Exiting.")
        raise SystemExit(1)
    os.makedirs(args.output_dir, exist_ok=True)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cond = load_egopress_conditioning(
        sample_dir=args.validation_dir,
        width=args.width,
        height=args.height,
        n_sample_frames=args.sample_n_frames,
        sample_frame_rate=args.sample_frame_rate,
        clip_start=args.clip_start,
        prototype_left_path=args.prototype_left_path,
        prototype_right_path=args.prototype_right_path,
        mask_left_path=args.mask_left_path,
        mask_right_path=args.mask_right_path,
        text_prompt_override=args.text_prompt,
    )
    num_frames = cond["num_frames"]
    print(f"[INFO] Loaded sample={args.validation_dir}")
    print(f"[INFO] hand_side={cond['hand_side']} num_frames={num_frames} batch_index={cond['batch_index']}")
    print(f"[INFO] text_prompt='{cond['text_prompt']}'")

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    mask_net = MaskNet(noise_latent_channels=unet.config.block_out_channels[0])

    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        import xformers  # noqa: F401

    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                # print(f"This is AnimationAttnProcessor: {name}")
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                # print(f"This is AnimationIDAttnProcessor: {name}")
                layer_name = name.split(".processor")[0]
                weights = {
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"],
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = PIFRAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = XFormersAttnProcessor()
            else:
                attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    unet.load_state_dict(torch.load(args.unet_model_name_or_path, map_location="cpu"), strict=True)
    mask_net.load_state_dict(torch.load(args.masknet_model_name_or_path, map_location="cpu"), strict=True)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    mask_net.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    vae.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    mask_net.to(device, dtype=weight_dtype)

    pipe_init_kwargs = dict(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        mask_net=mask_net,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Filter kwargs based on pipeline __init__ signature
    try:
        sig = inspect.signature(InferencePipeline.__init__)
        allowed = set(sig.parameters.keys())
        pipe_init_kwargs = {k: v for k, v in pipe_init_kwargs.items() if k in allowed}
    except Exception:
        pass

    pipeline = InferencePipeline(**pipe_init_kwargs).to(device=device, dtype=weight_dtype)

    if not hasattr(pipeline, "tokenizer"):
        pipeline.tokenizer = tokenizer
    if not hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder = text_encoder

    generator = torch.Generator(device=device).manual_seed(args.seed)

    call_kwargs = dict(
        rgb_frames=cond["rgb_frames"].to(device=device, dtype=weight_dtype),
        mask_frames=cond["mask_frames"].to(device=device, dtype=weight_dtype),
        prototype=cond["prototype"].to(device=device, dtype=weight_dtype),
        pressure_mask=cond["pressure_mask"].to(device=device, dtype=weight_dtype),
        text_prompt=cond["text_prompt"],

        height=args.height,
        width=args.width,
        num_frames=num_frames,
        tile_size=args.tile_size,
        tile_overlap=args.frames_overlap,
        decode_chunk_size=args.decode_chunk_size,
        motion_bucket_id=127.0,
        fps=7,
        min_guidance_scale=args.guidance_scale,
        max_guidance_scale=args.guidance_scale,
        noise_aug_strength=args.noise_aug_strength,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pil",
    )

    # Filter kwargs based on pipeline.__call__ signature
    try:
        call_sig = inspect.signature(pipeline.__call__)
        allowed = set(call_sig.parameters.keys())
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}
    except Exception:
        pass

    with torch.no_grad():
        result = pipeline(**call_kwargs)



    video_frames = result.frames[0]
    
    video_frames = concat_external_rgb_pressure_with_generated(
    video_frames=video_frames,
    batch_index=cond["batch_index"],
    validation_dir=args.validation_dir,
    )
    
    out_mp4 = os.path.join(args.output_dir, "animation_video.mp4")
    out_png_dir = os.path.join(args.output_dir, "animated_images")
    export_to_mp4(video_frames, out_mp4, fps=5)
    save_frames_as_png(video_frames, out_png_dir)

    print(f"[OK] Saved: {out_mp4}")
    print(f"[OK] Saved frames: {out_png_dir}")
    


if __name__ == "__main__":
    main()