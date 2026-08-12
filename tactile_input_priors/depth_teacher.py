#!/usr/bin/env python3
"""Offline MoGe teacher used by the depth-sidecar builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class MogeTeacher:
    """Load MoGe lazily and return CPU NumPy geometry maps."""

    OUTPUT_KEYS = (
        "depth",
        "mask",
        "normal",
        "points",
        "point_map",
        "confidence",
        "intrinsics",
        "fov_x",
        "fov_y",
    )

    def __init__(self, model: str | Path, device: str = "cuda:0"):
        import torch
        from moge.model.v2 import MoGeModel

        checkpoint = Path(model).expanduser().resolve(strict=False)
        if checkpoint.is_dir():
            checkpoint = checkpoint / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MoGe checkpoint file is missing: {checkpoint}")

        self.torch = torch
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(checkpoint).to(self.device).eval()

    def infer(self, bgr: np.ndarray) -> dict[str, np.ndarray]:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Expected a BGR image with shape [H,W,3], got {bgr.shape}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = (
            self.torch.from_numpy(rgb)
            .to(self.device, dtype=self.torch.float32)
            .permute(2, 0, 1)
            / 255.0
        )
        autocast_enabled = self.device.type == "cuda"
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.float16 if autocast_enabled else self.torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output: dict[str, Any] = self.model.infer(image)

        result: dict[str, np.ndarray] = {}
        for name in self.OUTPUT_KEYS:
            value = output.get(name)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            result[name] = np.asarray(value)
        return result
