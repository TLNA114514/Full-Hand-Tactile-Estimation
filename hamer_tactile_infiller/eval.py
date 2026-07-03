#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

cv2.setNumThreads(0)

INFILLER_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = INFILLER_DIR.parent
sys.path.append(str(WORKSPACE_DIR / "hamer"))
sys.path.append(str(WORKSPACE_DIR / "hamer_tactile_ft"))
sys.path.append(str(INFILLER_DIR))

try:
    from .opengl_guard import install_opengl_guard
except ImportError:
    from opengl_guard import install_opengl_guard

install_opengl_guard()

try:
    from .dataset import TactileSequenceDataset
    from .model import metrics
    from .train import InfillerLightningModule, manifest_paths_for_split, prepare_model_cfg
except ImportError:
    from dataset import TactileSequenceDataset
    from model import metrics
    from train import InfillerLightningModule, manifest_paths_for_split, prepare_model_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tactile infiller checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None, help="Manifest jsonl path(s), comma-separated list, or glob.")
    parser.add_argument("--manifest_dir", default=str(INFILLER_DIR / "manifests"))
    parser.add_argument("--manifest_prefix", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seq_stride", type=int, default=16)
    parser.add_argument("--sample_frame_rate", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = prepare_model_cfg()
    manifest = args.manifest
    if manifest is None:
        manifests = manifest_paths_for_split(args.manifest_dir, args.split, args.manifest_prefix)
        if not manifests:
            raise FileNotFoundError(
                f"No manifest found for split={args.split!r}, "
                f"manifest_dir={args.manifest_dir!r}, manifest_prefix={args.manifest_prefix!r}"
            )
        manifest = manifests
    dataset = TactileSequenceDataset(
        cfg=cfg,
        manifest_path=manifest,
        split=args.split,
        train=False,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        sample_frame_rate=args.sample_frame_rate,
        mask_prob=0.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    module = InfillerLightningModule.load_from_checkpoint(args.checkpoint, cfg=cfg, checkpoint=None)
    module = module.to(device).eval()

    sums = {}
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            output = module.model(batch)
            batch_metrics = metrics(batch, output)
            for key, value in batch_metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
            count += 1
    print("Infiller evaluation:")
    for key in sorted(sums):
        print(f"  {key}: {sums[key] / max(1, count):.6f}")


if __name__ == "__main__":
    main()
