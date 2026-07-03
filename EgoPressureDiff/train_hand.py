import argparse
import random
import logging
import math
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention_processor import XFormersAttnProcessor

from core.dataset.v2p_dataset import LargeScaleEgoTactileVideos
from core.modules.attention_processor import AnimationAttnProcessor
from core.modules.attention_processor_normalized import PIFRAttnProcessor
from core.modules.mask_net import MaskNet
from core.modules.unet import UNetSpatioTemporalConditionModel

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from einops import rearrange

import datetime
import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
import warnings
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def load_images_from_folder(folder):
    images = []

    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith(".png")]
    png_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        images.append(img)

    return images


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(
    shape, image_d, noise_d_low, noise_d_high, sigma_data=1.0, min_value=1e-3, max_value=1e3, device="cpu", dtype=torch.float32
):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=125,
        loop=0,
    )


def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)
    if len(t.shape) == 5:
        video_length = t.shape[1]

        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4:
        latents = vae.encode(t).latent_dist.sample()
    if scale:
        latents = latents * vae.config.scaling_factor
    return latents


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train Stable Diffusion XL for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help=(
            "Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training"
            " InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--log_trainable_parameters",
        action="store_true",
        help="Whether to write the trainable parameters.",
    )
    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=("path to the dataset csv"),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=("path to the video folder"),
    )
    parser.add_argument(
        "--condition_folder",
        type=str,
        default=None,
        help=("path to the depth folder"),
    )

    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=14,
        help=("the sample_n_frames"),
    )
    parser.add_argument(
        "--sample_frame_rate",
        type=int,
        default=4,
        help="Frame sampling rate (take 1 frame every N frames).",
    )
    parser.add_argument(
        "--masknet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained masknet model",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model",
    )

    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        help="Path to the data root path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data path",
    )

    parser.add_argument(
        "--finetune_mode",
        type=bool,
        default=False,
        help="Enable or disable the finetune mode (True/False).",
    )
    parser.add_argument(
        "--masknet_model_finetune_path",
        type=str,
        default=None,
        help="Path to the pretrained masknet model",
    )
    parser.add_argument(
        "--unet_model_finetune_path",
        type=str,
        default=None,
        help="Path to the pretrained unet model",
    )

    parser.add_argument(
        "--dataset_width",
        type=int,
        default=512,
        help="video dataset width",
    )
    parser.add_argument(
        "--dataset_height",
        type=int,
        default=512,
        help="video dataset height",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.multiprocessing.set_start_method("spawn")

    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16"
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    mask_net = MaskNet(noise_latent_channels=unet.config.block_out_channels[0])

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # init adapter modules
    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    # Save the UNet architecture to a text file
    unet_architecture_path = os.path.join(args.output_dir, "unet_architecture.txt")
    with open(unet_architecture_path, "w") as f:
        for name, module in unet.named_modules():
            f.write(f"{name}: {module.__class__.__name__}\n")
    logger.info(f"UNet architecture saved to {unet_architecture_path}")

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
                attn_procs[name] = AnimationAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
                )
            else:
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

    # triggering the finetune mode
    if (
        args.finetune_mode is True
        and args.masknet_model_finetune_path is not None
        and args.unet_model_finetune_path is not None
    ):
        print("Loading existing masknet weights and unet weights.")
        if args.masknet_model_finetune_path.endswith(".pth"):
            mask_net_state_dict = torch.load(args.masknet_model_finetune_path, map_location="cpu")
            mask_net.load_state_dict(mask_net_state_dict, strict=True)
        else:
            print("masknet weights loading fail")
            print(1 / 0)
        if args.unet_model_finetune_path.endswith(".pth"):
            unet_state_dict = torch.load(args.unet_model_finetune_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1 / 0)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Freeze vae and encoders
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    mask_net.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training,"
                    " please update xFormers to at least 0.0.17. See"
                    " https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    mask_net.requires_grad_(True)

    parameters_list = []

    for name, para in mask_net.named_parameters():
        para.requires_grad = True
        parameters_list.append({"params": para, "lr": args.learning_rate})

    for name, para in unet.named_parameters():
        if "attentions" in name:
            para.requires_grad = True
            parameters_list.append({"params": para})
        else:
            para.requires_grad = False

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process and args.log_trainable_parameters:
        rec_txt1 = open("rec_para.txt", "w")
        rec_txt2 = open("rec_para_train.txt", "w")
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f"{name}\n")
            else:
                rec_txt2.write(f"{name}\n")
        rec_txt1.close()
        rec_txt2.close()

    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    root_path = args.data_root_path
    txt_path = args.data_path

    train_dataset = LargeScaleEgoTactileVideos(
        root_path=root_path,
        txt_path=txt_path,
        width=args.dataset_width,
        height=args.dataset_height,
        n_sample_frames=args.sample_n_frames,
        sample_frame_rate=args.sample_frame_rate,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, mask_net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, mask_net, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("StableAnimator", config=vars(args))

    total_batch_size = (
        args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        pixel_values = pixel_values.to(torch.float32)
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(device=accelerator.device, dtype=image_encoder.dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        unet=None,
        device=None,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def encode_text(text_prompts):
        if isinstance(text_prompts, tuple):
            text_prompts = list(text_prompts)

        text_inputs = tokenizer(
            text=text_prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device=accelerator.device)
        attention_mask = text_inputs.attention_mask.to(device=accelerator.device)

        text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.text_embeds
        return text_embeddings

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        mask_net.train()
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch:
                resume_step = (global_step * args.gradient_accumulation_steps) % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
                if step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

            with accelerator.accumulate(mask_net, unet):
                with accelerator.autocast():
                    tgt_pressure = batch["tgt_pressure"].to(weight_dtype).to(accelerator.device, non_blocking=True)

                    train_noise_aug = 0.02
                    rgb_frames = batch["rgb_frames"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                    rgb_frames = rgb_frames + train_noise_aug * randn_tensor(
                        rgb_frames.shape,
                        generator=generator,
                        device=rgb_frames.device,
                        dtype=rgb_frames.dtype,
                    )

                    prototype = batch["prototype"].to(weight_dtype).to(accelerator.device, non_blocking=True)

                    latents = tensor_to_vae_latent(tgt_pressure, vae).to(dtype=weight_dtype)
                    rgb_latents = tensor_to_vae_latent(rgb_frames, vae, scale=False).to(dtype=weight_dtype)

                    # Get the image embedding for conditioning.
                    # prototype: [B, 3, H, W], encode once then repeat F (num frames) times
                    # encode single prototype image to get [B, 1, C]
                    encoder_hidden_states = encode_image(prototype) # [B, 1, 1024]
                    # frame_num = rgb_frames.shape[1]
                    # repeat along frame dimension to match video frames: [B, frame_num, C]
                    # encoder_hidden_states = encoder_hidden_states.repeat(1, frame_num, 1)

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    sigmas = rand_cosine_interpolated(
                        shape=[bsz],
                        image_d=image_d,
                        noise_d_low=noise_d_low,
                        noise_d_high=noise_d_high,
                        sigma_data=sigma_data,
                        min_value=min_value,
                        max_value=max_value,
                    ).to(latents.device, dtype=weight_dtype)

                    sigmas_reshaped = sigmas.clone()
                    while len(sigmas_reshaped.shape) < len(latents.shape):
                        sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                    noisy_latents = latents + noise * sigmas_reshaped

                    timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
                        latents.device, dtype=weight_dtype
                    )

                    inp_noisy_latents = noisy_latents / ((sigmas_reshaped**2 + 1) ** 0.5)

                    added_time_ids = _get_add_time_ids(
                        fps=6,
                        motion_bucket_id=127.0,
                        noise_aug_strength=train_noise_aug,
                        dtype=encoder_hidden_states.dtype,
                        batch_size=bsz,
                        unet=unet,
                        device=latents.device,
                    )

                    added_time_ids = added_time_ids.to(latents.device)

                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(bsz, device=latents.device, generator=generator)
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states
                        )

                        image_mask_dtype = rgb_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1, 1)
                        rgb_latents = image_mask * rgb_latents

                    inp_noisy_latents = torch.cat([inp_noisy_latents, rgb_latents], dim=2)

                    mask_pixels = batch["mask_frames"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    mask_latents = mask_net(mask_pixels)

                    text_prompts = batch["text_prompt"]  # list[str]
                    text_embeddings = encode_text(text_prompts)  # [B, 1024]
                    text_latents = text_embeddings.unsqueeze(1) # [B, 1, 1024]
                    
                    target = latents

                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, text_latents], dim=1
                    )  # [B, 2, 1024]

                    encoder_hidden_states = encoder_hidden_states.to(latents.dtype)
                    inp_noisy_latents = inp_noisy_latents.to(latents.dtype)
                    mask_latents = mask_latents.to(latents.dtype)

                    model_pred = unet(
                        inp_noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_time_ids=added_time_ids,
                        mask_latents=mask_latents,
                    ).sample

                    sigmas = sigmas_reshaped
                    c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents

                    weighing = (1 + sigmas**2) * (sigmas**-2.0)

                    pressure_mask = batch["pressure_mask"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    pressure_mask = rearrange(
                        pressure_mask, "b f c h w -> (b f) c h w"
                    )
                    pressure_mask = F.interpolate(
                        pressure_mask,
                        size=(target.size()[-2], target.size()[-1]),
                        mode="nearest",
                    )
                    pressure_mask = rearrange(
                        pressure_mask, "(b f) c h w -> b f c h w", f=args.sample_n_frames
                    )

                    loss = torch.mean(
                        (
                            weighing.float()
                            * (denoised_latents.float() - target.float()) ** 2
                            * (1 + pressure_mask)
                        ).reshape(target.shape[0], -1),
                        dim=1,
                    )

                    loss = loss.mean()

                    avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    with torch.cuda.device(latents.device):
                        torch.cuda.empty_cache()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    unwrap_unet = accelerator.unwrap_model(unet)
                    unwrap_mask_net = accelerator.unwrap_model(mask_net)

                    unwrap_unet_state_dict = unwrap_unet.state_dict()
                    torch.save(
                        unwrap_unet_state_dict,
                        os.path.join(
                            args.output_dir, f"checkpoint-{global_step}", f"unet-{global_step}.pth"
                        ),
                    )
                    unwrap_mask_net_state_dict = unwrap_mask_net.state_dict()
                    torch.save(
                        unwrap_mask_net_state_dict,
                        os.path.join(
                            args.output_dir,
                            f"checkpoint-{global_step}",
                            f"mask_net-{global_step}.pth",
                        ),
                    )

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-last")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main()
