#!/usr/bin/env python
"""Upload durable epoch snapshots to one resumable WandB run."""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path


class UploadTimeout(TimeoutError):
    pass


def _timeout_handler(signum, frame):
    del signum, frame
    raise UploadTimeout("WandB upload attempt timed out")


def _pending_payloads(queue_dir):
    return sorted(Path(queue_dir).glob("epoch_*.json"))


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _acquire_lock(lock_path, wait_seconds):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + max(float(wait_seconds), 0.0)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} acquired_unix={time.time():.6f}\n")
            handle.flush()
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                return None
            time.sleep(1.0)


def _upload_pending(args):
    pending = _pending_payloads(args.queue_dir)
    if not pending:
        return 0

    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_SILENT"] = "true"
    os.environ.setdefault("WANDB_INIT_TIMEOUT", str(args.attempt_timeout))
    os.environ.setdefault("WANDB_HTTP_TIMEOUT", "20")
    runtime_dir = Path(args.queue_dir) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(runtime_dir)

    import wandb

    config = _load_json(args.config) if args.config else {}
    run = None
    uploaded = []
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(int(args.attempt_timeout), 1))
    try:
        run = wandb.init(
            project=args.project,
            name=args.name,
            id=args.run_id,
            resume="allow",
            config=config,
            dir=str(runtime_dir),
        )
        for payload_path in pending:
            payload = _load_json(payload_path)
            payload_run_id = str(payload.get("run_id", "") or "")
            if payload_run_id != args.run_id:
                raise RuntimeError(
                    f"Queued payload run ID mismatch at {payload_path}: "
                    f"{payload_run_id!r} != {args.run_id!r}"
                )
            metrics = dict(payload.get("metrics", {}))
            metrics.setdefault("trainer/epoch", float(payload["epoch"]))
            metrics.setdefault(
                "trainer/global_step",
                float(payload["global_step"]),
            )
            run.log(
                metrics,
                step=int(payload["global_step"]),
                commit=True,
            )
            uploaded.append(payload_path)
        run.finish(exit_code=0)
        run = None
    finally:
        signal.alarm(0)
        if run is not None:
            try:
                signal.alarm(10)
                run.finish(exit_code=1)
            except Exception:
                pass
            finally:
                signal.alarm(0)

    sent_dir = Path(args.queue_dir) / "sent"
    sent_dir.mkdir(parents=True, exist_ok=True)
    for payload_path in uploaded:
        os.replace(payload_path, sent_dir / payload_path.name)
    return len(uploaded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--retries", type=int, default=24)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--attempt-timeout", type=int, default=120)
    parser.add_argument("--lock-wait-seconds", type=int, default=30)
    args = parser.parse_args()

    args.queue_dir = str(Path(args.queue_dir).expanduser().resolve())
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        args.config = str(Path(args.config).expanduser().resolve())

    lock_handle = _acquire_lock(
        queue_dir / ".upload.lock",
        args.lock_wait_seconds,
    )
    if lock_handle is None:
        print("Another WandB epoch uploader owns the queue lock.", flush=True)
        return

    try:
        retries = max(int(args.retries), 1)
        for attempt in range(retries):
            try:
                while _pending_payloads(queue_dir):
                    count = _upload_pending(args)
                    print(
                        f"Uploaded {count} epoch payload(s) to WandB "
                        f"run {args.run_id}.",
                        flush=True,
                    )
                return
            except Exception as exc:
                print(
                    f"WandB epoch upload attempt {attempt + 1}/{retries} "
                    f"failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt + 1 < retries:
                    time.sleep(max(int(args.interval), 1))
        raise SystemExit(
            "WandB upload retries exhausted; pending epoch payloads remain local."
        )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    main()
