#!/usr/bin/env python
"""Run tactile jobs in an isolated process group and clean their descendants."""

import argparse
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from process_lifecycle import SUPERVISOR_PID_ENV, set_parent_death_signal


DEFAULT_REGISTRY = Path(__file__).resolve().parent / "run_processes"


def _process_start_ticks(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()
    except (FileNotFoundError, ProcessLookupError, PermissionError, TypeError, ValueError):
        return None
    return int(fields[21]) if len(fields) > 21 else None


def _pid_matches(pid, start_ticks):
    current = _process_start_ticks(pid)
    try:
        return current is not None and current == int(start_ticks)
    except (TypeError, ValueError):
        return False


def _process_group_exists(pgid):
    try:
        os.killpg(int(pgid), 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _signal_group(pgid, sig):
    try:
        os.killpg(int(pgid), int(sig))
        return True
    except ProcessLookupError:
        return False


def _wait_for_group_exit(pgid, timeout):
    deadline = time.monotonic() + max(float(timeout), 0.0)
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.2)
    return not _process_group_exists(pgid)


def _atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _local_records(registry_dir):
    hostname = socket.gethostname()
    records = []
    for path in sorted(Path(registry_dir).glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue
        if record.get("hostname") == hostname:
            records.append((path, record))
    return records


def _record_status(record):
    supervisor_alive = _pid_matches(
        record.get("supervisor_pid", -1),
        record.get("supervisor_start_ticks", -1),
    )
    child_alive = _pid_matches(
        record.get("child_pid", -1),
        record.get("child_start_ticks", -1),
    )
    return supervisor_alive, child_alive


def list_runs(registry_dir):
    records = _local_records(registry_dir)
    if not records:
        print("No registered tactile processes on this host.")
        return 0
    for path, record in records:
        supervisor_alive, child_alive = _record_status(record)
        print(
            f"{record.get('run_id')} "
            f"supervisor={record.get('supervisor_pid')}:{'alive' if supervisor_alive else 'dead'} "
            f"child={record.get('child_pid')}:{'alive' if child_alive else 'dead'} "
            f"pgid={record.get('process_group_id')} "
            f"started={record.get('started_utc')} "
            f"command={record.get('command_text')} "
            f"state={path}"
        )
    return 0


def cleanup_stale(registry_dir):
    removed = 0
    for path, record in _local_records(registry_dir):
        supervisor_alive, child_alive = _record_status(record)
        if not supervisor_alive and not child_alive:
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"Removed {removed} stale tactile process record(s).")
    return 0


def terminate_all(registry_dir, grace_seconds, kill_wait_seconds):
    failures = 0
    for path, record in _local_records(registry_dir):
        supervisor_alive, child_alive = _record_status(record)
        if supervisor_alive:
            print(f"Requesting supervisor shutdown: {record.get('run_id')}")
            try:
                os.kill(int(record["supervisor_pid"]), signal.SIGTERM)
            except ProcessLookupError:
                supervisor_alive = False
        elif child_alive:
            pgid = int(record["process_group_id"])
            try:
                current_pgid = os.getpgid(int(record["child_pid"]))
            except ProcessLookupError:
                continue
            if current_pgid != pgid:
                print(f"Refusing mismatched process group for {record.get('run_id')}")
                failures += 1
                continue
            print(f"Stopping orphaned process group {pgid}: {record.get('run_id')}")
            _signal_group(pgid, signal.SIGTERM)

    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        active = [
            record
            for _, record in _local_records(registry_dir)
            if any(_record_status(record))
        ]
        if not active:
            break
        time.sleep(0.25)

    for path, record in _local_records(registry_dir):
        supervisor_alive, child_alive = _record_status(record)
        if child_alive:
            pgid = int(record["process_group_id"])
            try:
                current_pgid = os.getpgid(int(record["child_pid"]))
            except ProcessLookupError:
                current_pgid = None
            if current_pgid == pgid:
                print(f"Force-killing process group {pgid}: {record.get('run_id')}")
                _signal_group(pgid, signal.SIGKILL)
                _wait_for_group_exit(pgid, kill_wait_seconds)
        if supervisor_alive:
            try:
                os.kill(int(record["supervisor_pid"]), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not any(_record_status(record)):
            path.unlink(missing_ok=True)
    return failures


class Supervisor:
    def __init__(self, command, registry_dir, grace_seconds, kill_wait_seconds):
        self.command = list(command)
        self.registry_dir = Path(registry_dir)
        self.grace_seconds = float(grace_seconds)
        self.kill_wait_seconds = float(kill_wait_seconds)
        self.child = None
        self.process_group_id = None
        self.state_path = None
        self.forwarded_signal = None
        self.shutdown_deadline = None

    def _handle_signal(self, signum, _frame):
        if self.child is None or self.child.poll() is not None:
            return
        if self.forwarded_signal is not None:
            print("Second shutdown signal received; force-killing tactile process group.", flush=True)
            _signal_group(self.process_group_id, signal.SIGKILL)
            return
        forwarded = signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM
        self.forwarded_signal = forwarded
        self.shutdown_deadline = time.monotonic() + self.grace_seconds
        print(
            f"Forwarding {signal.Signals(forwarded).name} to tactile process group "
            f"{self.process_group_id}; force kill in {self.grace_seconds:.0f}s if needed.",
            flush=True,
        )
        _signal_group(self.process_group_id, forwarded)

    def run(self):
        cleanup_stale(self.registry_dir)
        environment = dict(os.environ)
        environment[SUPERVISOR_PID_ENV] = str(os.getpid())
        environment["TACTILE_RUN_ID"] = uuid.uuid4().hex
        self.child = subprocess.Popen(
            self.command,
            env=environment,
            start_new_session=True,
            preexec_fn=lambda: set_parent_death_signal(
                signal.SIGKILL,
                expected_parent_pid=os.getppid(),
            ),
        )
        self.process_group_id = self.child.pid
        run_id = environment["TACTILE_RUN_ID"]
        self.state_path = self.registry_dir / (
            f"{socket.gethostname()}-{os.getpid()}-{run_id}.json"
        )
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "hostname": socket.gethostname(),
            "supervisor_pid": os.getpid(),
            "supervisor_start_ticks": _process_start_ticks(os.getpid()),
            "child_pid": self.child.pid,
            "child_start_ticks": _process_start_ticks(self.child.pid),
            "process_group_id": self.process_group_id,
            "started_unix": time.time(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cwd": os.getcwd(),
            "command": self.command,
            "command_text": " ".join(self.command),
        }
        _atomic_write_json(self.state_path, record)
        print(
            f"Tactile supervisor: pid={os.getpid()}, child={self.child.pid}, "
            f"process_group={self.process_group_id}, run_id={run_id}",
            flush=True,
        )

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, self._handle_signal)

        try:
            while self.child.poll() is None:
                if (
                    self.shutdown_deadline is not None
                    and time.monotonic() >= self.shutdown_deadline
                ):
                    print(
                        f"Grace period expired; force-killing process group "
                        f"{self.process_group_id}.",
                        flush=True,
                    )
                    _signal_group(self.process_group_id, signal.SIGKILL)
                    self.shutdown_deadline = None
                time.sleep(0.2)
            return_code = int(self.child.returncode)
        finally:
            if self.process_group_id is not None and _process_group_exists(
                self.process_group_id
            ):
                _signal_group(self.process_group_id, signal.SIGTERM)
                if not _wait_for_group_exit(self.process_group_id, self.kill_wait_seconds):
                    _signal_group(self.process_group_id, signal.SIGKILL)
                    _wait_for_group_exit(self.process_group_id, self.kill_wait_seconds)
            if self.state_path is not None:
                self.state_path.unlink(missing_ok=True)
        return return_code


def parse_args():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list-runs", action="store_true")
    actions.add_argument("--cleanup-stale", action="store_true")
    actions.add_argument("--terminate-all", action="store_true")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--grace-seconds", type=float, default=60.0)
    parser.add_argument("--kill-wait-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main():
    args = parse_args()
    if args.list_runs:
        return list_runs(args.registry_dir)
    if args.cleanup_stale:
        return cleanup_stale(args.registry_dir)
    if args.terminate_all:
        return terminate_all(
            args.registry_dir,
            args.grace_seconds,
            args.kill_wait_seconds,
        )
    if not args.command:
        raise SystemExit("A command is required after --")
    return Supervisor(
        args.command,
        args.registry_dir,
        args.grace_seconds,
        args.kill_wait_seconds,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
