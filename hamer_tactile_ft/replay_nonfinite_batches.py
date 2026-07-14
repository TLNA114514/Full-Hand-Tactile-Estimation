#!/usr/bin/env python3
"""Replay a captured tactile batch under one numerical precision."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def early_gpu_selection():
    for index, value in enumerate(sys.argv):
        if value in ("--gpu", "--gpus") and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            break


early_gpu_selection()
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_PLATFORM", "egl")

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


FT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = FT_DIR.parent
sys.path.insert(0, str(WORKSPACE_DIR / "hamer"))
sys.path.insert(0, str(FT_DIR))

import audit_gradient_conflicts as base
from dataset import OpenTouchTactileDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="Capture directory or one nonfinite_grad_rank*.jsonl file.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), required=True)
    parser.add_argument("--gpu", "--gpus", dest="gpu", default="0")
    parser.add_argument("--event_index", type=int, default=-1, help="Replay all captured rank events by default; select one with a nonnegative index.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def capture_files(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(path.glob("nonfinite_grad_rank*.jsonl"))


def load_events(path):
    events = []
    for file_path in capture_files(path):
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    events.sort(key=lambda item: (item.get("global_step", -1), item.get("rank", -1)))
    return events


def finite_tensor_stats(value):
    tensor = value.detach().float()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    return {
        "shape": list(tensor.shape),
        "nonfinite_count": int((~finite).sum().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "abs_max": float(finite_values.abs().max().item()) if finite_values.numel() else None,
    }


def gradient_stats(model):
    first_nonfinite = None
    nonfinite_count = 0
    branch_squared = {}
    total_squared = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.tactile_head.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared = gradient.pow(2).sum()
        total_squared += squared
        branch = name.split(".", 1)[0]
        branch_squared[branch] = branch_squared.get(
            branch,
            torch.zeros((), device=gradient.device),
        ) + squared
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite_count += 1
            if first_nonfinite is None:
                first_nonfinite = name

    total_norm = float(torch.sqrt(total_squared).item())
    branch_norms = {}
    for branch, squared in branch_squared.items():
        value = float(torch.sqrt(squared).item())
        branch_norms[branch] = value if math.isfinite(value) else None
    return {
        "first_nonfinite_parameter": first_nonfinite,
        "nonfinite_parameter_count": nonfinite_count,
        "grad_norm": total_norm if math.isfinite(total_norm) else None,
        "branch_grad_norms": branch_norms,
    }


def prepare_batch(event, device):
    samples = event.get("batch", {}).get("samples", [])
    payload_path = Path(event.get("replay_batch_path", "")).expanduser()
    replay_payload = torch.load(payload_path, map_location="cpu") if payload_path.is_file() else None
    if replay_payload is not None:
        batch = {
            key: value
            for key, value in replay_payload.items()
            if key not in {"cpu_rng_state", "cuda_rng_state"}
        }
        sample_count = int(batch["img"].shape[0])
    else:
        if not samples:
            raise ValueError("Captured event contains neither an exact tensor payload nor sample provenance")
        records = [{
            "sample_dir": sample["sample_dir"],
            "dataset": sample["dataset"],
            "hand": sample["hand"],
            "is_right": int(sample["is_right"]),
        } for sample in samples]
        cfg = base.load_model_cfg()
        dataset = OpenTouchTactileDataset(
            cfg=cfg,
            split="train",
            data_dir=[],
            train=False,
            sample_records=records,
        )
        batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)))
        sample_count = len(records)
    return base.move_to_device(batch, device), replay_payload, payload_path, sample_count


def replay_event(model, event, event_index, args, device):
    batch, replay_payload, payload_path, sample_count = prepare_batch(event, device)
    optimizer = torch.optim.AdamW(model.tactile_head.parameters(), lr=1e-4)
    scaler = None
    if args.precision == "fp16":
        initial_scale = event.get("scaler_scale_before") or 65536.0
        scaler = torch.cuda.amp.GradScaler(init_scale=float(initial_scale))
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    elif args.precision == "bf16":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = torch.autocast(device_type="cuda", enabled=False)

    optimizer.zero_grad(set_to_none=True)
    if replay_payload is not None:
        cpu_rng_state = replay_payload.get("cpu_rng_state")
        cuda_rng_state = replay_payload.get("cuda_rng_state")
        if isinstance(cpu_rng_state, torch.Tensor):
            torch.set_rng_state(cpu_rng_state.cpu())
        if isinstance(cuda_rng_state, torch.Tensor):
            torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=device)
    with autocast:
        output = model.forward_step(batch, train=True)
        loss = model.compute_loss(batch, output, train=True)
    scale_before = float(scaler.get_scale()) if scaler is not None else None
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    grad = gradient_stats(model)
    # A non-finite fp16 step is safe to pass to GradScaler: it skips the update.
    # Finite replay events deliberately do not update model weights, so all rank
    # batches are compared against exactly the same checkpoint state.
    if scaler is not None and grad["nonfinite_parameter_count"] > 0:
        scaler.step(optimizer)
        scaler.update()
    scale_after = float(scaler.get_scale()) if scaler is not None else None
    return {
        "event_index": event_index,
        "captured_rank": event.get("rank"),
        "captured_global_step": event.get("global_step"),
        "sample_count": sample_count,
        "used_exact_tensor_payload": replay_payload is not None,
        "replay_batch_path": str(payload_path) if replay_payload is not None else None,
        "loss": float(loss.detach().float().cpu()) if torch.isfinite(loss.detach()).item() else None,
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "scaler_scale_before": scale_before,
        "scaler_scale_after": scale_after,
        "optimizer_step_skipped": bool(
            scaler is not None and grad["nonfinite_parameter_count"] > 0
        ),
        "replay_update_suppressed": True,
        "gradient": grad,
        "tensor_stats": {
            name: finite_tensor_stats(output[name])
            for name in (
                "pred_tactile",
                "support_logits",
                "positive_logits",
                "pressure_logits",
                "coarse_pressure",
                "residual",
            )
            if output.get(name) is not None
        },
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.capture)
    if not events:
        result = {
            "status": "no_captured_nonfinite_event",
            "capture": str(Path(args.capture).expanduser().resolve()),
            "precision": args.precision,
        }
        with (output_dir / "replay_result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print("No captured non-finite event; replay was not run.")
        return
    if args.event_index >= len(events):
        raise IndexError(f"event_index={args.event_index} outside [0, {len(events) - 1}]")
    selected = list(enumerate(events)) if args.event_index < 0 else [(args.event_index, events[args.event_index])]

    first_event = selected[0][1]
    checkpoint = Path(args.checkpoint or first_event.get("checkpoint", "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Replay checkpoint not found: {checkpoint}")
    seed = int(args.seed if args.seed is not None else first_event.get("seed", 2029))
    pl.seed_everything(seed, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Replay requires a CUDA GPU")
    device = torch.device("cuda:0")
    model_args = SimpleNamespace(tactile_head_type="zero_ordinal_residual")
    model, _loss_config, model_config = base.load_model(model_args, checkpoint, checkpoint.parent, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.tactile_head.parameters():
        parameter.requires_grad_(True)
    model.train()

    replay_results = []
    for event_index, event in selected:
        print(
            f"Replaying event {event_index}/{len(events) - 1}: "
            f"rank={event.get('rank')} step={event.get('global_step')} precision={args.precision}",
            flush=True,
        )
        replay_results.append(replay_event(model, event, event_index, args, device))
    reproduced = sum(item["gradient"]["nonfinite_parameter_count"] > 0 for item in replay_results)
    result = {
        "status": "replayed",
        "precision": args.precision,
        "checkpoint": str(checkpoint),
        "captured_event_count": len(events),
        "replayed_event_count": len(replay_results),
        "nonfinite_reproduced_count": reproduced,
        "seed": seed,
        "events": replay_results,
        "model_config": model_config,
    }
    with (output_dir / "replay_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "precision": args.precision,
        "replayed_event_count": len(replay_results),
        "nonfinite_reproduced_count": reproduced,
    }, indent=2))
    print(f"Wrote replay result to: {output_dir / 'replay_result.json'}")


if __name__ == "__main__":
    main()
