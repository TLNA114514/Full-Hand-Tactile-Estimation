#!/usr/bin/env python3
"""Low-overhead Linux I/O monitor for tactile training process sessions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import time
from collections import Counter
from pathlib import Path


_STOP = False


def _request_stop(_signum, _frame):
    global _STOP
    _STOP = True


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return ""


def _process_stat(pid: int):
    raw = _read_text(Path(f"/proc/{pid}/stat")).strip()
    close = raw.rfind(")")
    if close < 0:
        return None
    try:
        process_id = int(raw[: raw.find(" ")])
        command = raw[raw.find("(") + 1 : close]
        fields = raw[close + 2 :].split()
        return {
            "pid": process_id,
            "command": command,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
        }
    except (IndexError, TypeError, ValueError):
        return None


def _key_values(path: Path, *, suffix: str = "") -> dict[str, int]:
    result = {}
    for line in _read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        token = value.strip().split()[0] if value.strip() else "0"
        try:
            result[f"{key}{suffix}"] = int(token)
        except ValueError:
            continue
    return result


def _pressure(path: Path) -> dict[str, float]:
    result = {}
    for line in _read_text(path).splitlines():
        parts = line.split()
        if not parts:
            continue
        prefix = parts[0]
        for token in parts[1:]:
            if "=" not in token:
                continue
            name, value = token.split("=", 1)
            try:
                result[f"{prefix}_{name}"] = float(value)
            except ValueError:
                continue
    return result


def _cmdline(pid: int) -> str:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return ""
    return " ".join(part.decode("utf-8", "replace") for part in payload.split(b"\0") if part)


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _latest_registry_session(registry_dir: str) -> int:
    candidates = []
    hostname = socket.gethostname()
    for path in Path(registry_dir).expanduser().glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("hostname") != hostname:
                continue
            child_pid = int(record["child_pid"])
            session_id = int(record["process_group_id"])
            stat = _process_stat(child_pid)
            if stat is None or stat["session"] != session_id:
                continue
            candidates.append((float(record.get("started_unix", 0.0)), session_id))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise RuntimeError(f"No active supervised training session under {registry_dir}")
    return max(candidates)[1]


def monitor(args) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    system_path = output_dir / "system_io.csv"
    waits_path = output_dir / "process_d_waits.csv"
    summary_path = output_dir / "system_io_summary.json"
    session_id = (
        _latest_registry_session(args.registry_dir)
        if args.registry_dir
        else int(args.session_id if args.session_id is not None else os.getsid(0))
    )
    interval = max(float(args.interval), 0.25)
    flush_interval = max(float(args.flush_interval), interval)

    system_fields = (
        "time_unix", "elapsed_s", "session_id", "process_count", "state_R",
        "state_S", "state_D", "state_Z", "load1", "load5", "load15",
        "mem_available_kb", "dirty_kb", "writeback_kb", "io_some_avg10",
        "io_some_avg60", "io_some_avg300", "io_some_total", "io_full_avg10",
        "io_full_avg60", "io_full_avg300", "io_full_total",
    )
    wait_fields = (
        "time_unix", "elapsed_s", "pid", "ppid", "pgrp", "session_id",
        "state", "wchan", "read_bytes", "write_bytes", "rchar", "wchar",
        "syscr", "syscw", "rss_kb", "command",
    )
    system_exists = system_path.exists() and system_path.stat().st_size > 0
    waits_exists = waits_path.exists() and waits_path.stat().st_size > 0
    started = time.monotonic()
    last_flush = started
    samples = 0
    d_observations = 0
    wchan_counts: Counter[str] = Counter()
    pid_d_counts: Counter[str] = Counter()

    with system_path.open("a", encoding="utf-8", newline="") as system_file, \
         waits_path.open("a", encoding="utf-8", newline="") as waits_file:
        system_writer = csv.DictWriter(system_file, fieldnames=system_fields)
        waits_writer = csv.DictWriter(waits_file, fieldnames=wait_fields)
        if not system_exists:
            system_writer.writeheader()
        if not waits_exists:
            waits_writer.writeheader()

        while not _STOP:
            now = time.time()
            elapsed = time.monotonic() - started
            states: Counter[str] = Counter()
            processes = []
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                stat = _process_stat(int(entry.name))
                if stat is None or stat["session"] != session_id:
                    continue
                states[stat["state"]] += 1
                processes.append(stat)

            memory = _key_values(Path("/proc/meminfo"))
            pressure = _pressure(Path("/proc/pressure/io"))
            try:
                load1, load5, load15 = os.getloadavg()
            except OSError:
                load1 = load5 = load15 = float("nan")
            system_writer.writerow(
                {
                    "time_unix": f"{now:.6f}",
                    "elapsed_s": f"{elapsed:.3f}",
                    "session_id": session_id,
                    "process_count": len(processes),
                    "state_R": states.get("R", 0),
                    "state_S": states.get("S", 0),
                    "state_D": states.get("D", 0),
                    "state_Z": states.get("Z", 0),
                    "load1": load1,
                    "load5": load5,
                    "load15": load15,
                    "mem_available_kb": memory.get("MemAvailable", 0),
                    "dirty_kb": memory.get("Dirty", 0),
                    "writeback_kb": memory.get("Writeback", 0),
                    **{f"io_{key}": pressure.get(key, 0.0) for key in (
                        "some_avg10", "some_avg60", "some_avg300", "some_total",
                        "full_avg10", "full_avg60", "full_avg300", "full_total",
                    )},
                }
            )
            for stat in processes:
                if stat["state"] != "D":
                    continue
                pid = stat["pid"]
                wchan = _read_text(Path(f"/proc/{pid}/wchan")).strip() or "unavailable"
                process_io = _key_values(Path(f"/proc/{pid}/io"))
                status = _key_values(Path(f"/proc/{pid}/status"))
                command = _cmdline(pid) or stat["command"]
                waits_writer.writerow(
                    {
                        "time_unix": f"{now:.6f}",
                        "elapsed_s": f"{elapsed:.3f}",
                        "pid": pid,
                        "ppid": stat["ppid"],
                        "pgrp": stat["pgrp"],
                        "session_id": session_id,
                        "state": "D",
                        "wchan": wchan,
                        "read_bytes": process_io.get("read_bytes", 0),
                        "write_bytes": process_io.get("write_bytes", 0),
                        "rchar": process_io.get("rchar", 0),
                        "wchar": process_io.get("wchar", 0),
                        "syscr": process_io.get("syscr", 0),
                        "syscw": process_io.get("syscw", 0),
                        "rss_kb": status.get("VmRSS", 0),
                        "command": command[:1000],
                    }
                )
                d_observations += 1
                wchan_counts[wchan] += 1
                pid_d_counts[f"{pid}:{stat['command']}"] += 1
            samples += 1
            if time.monotonic() - last_flush >= flush_interval:
                system_file.flush()
                waits_file.flush()
                last_flush = time.monotonic()
            time.sleep(interval)

        system_file.flush()
        waits_file.flush()

    _atomic_json(
        summary_path,
        {
            "format": "tactile_prior_runtime_io_v1",
            "session_id": session_id,
            "interval_seconds": interval,
            "duration_seconds": time.monotonic() - started,
            "sample_count": samples,
            "d_state_observation_count": d_observations,
            "top_d_state_wchan": wchan_counts.most_common(30),
            "top_d_state_processes": pid_d_counts.most_common(30),
            "system_csv": str(system_path),
            "d_wait_csv": str(waits_path),
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--session-id", type=int)
    source.add_argument(
        "--registry-dir",
        help="Auto-attach to the newest active process-supervisor record on this host.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--flush-interval", type=float, default=30.0)
    return parser


def main() -> None:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _request_stop)
    raise SystemExit(monitor(build_parser().parse_args()))


if __name__ == "__main__":
    main()
