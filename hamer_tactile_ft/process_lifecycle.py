"""Linux process-lifecycle helpers for supervised training and worker pools."""

import ctypes
import errno
import os
import signal
import sys


PR_SET_PDEATHSIG = 1
SUPERVISOR_PID_ENV = "TACTILE_SUPERVISOR_PID"


def _pid_exists(pid):
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def set_parent_death_signal(death_signal=signal.SIGKILL, expected_parent_pid=None):
    """Ask Linux to kill this process when its current parent exits."""
    if sys.platform != "linux":
        return False

    parent_pid = os.getppid()
    expected_parent_pid = (
        parent_pid if expected_parent_pid is None else int(expected_parent_pid)
    )
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(
        PR_SET_PDEATHSIG,
        int(death_signal),
        0,
        0,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))

    # The parent may have exited between getppid() and prctl(). Do not allow
    # that race to leave an orphan attached to init.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), int(death_signal))
    return True


def configure_supervised_process():
    """Enable parent-death handling for a supervised root or spawned rank."""
    supervisor_pid_text = os.environ.get(SUPERVISOR_PID_ENV, "").strip()
    if not supervisor_pid_text:
        if "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ:
            return set_parent_death_signal(
                signal.SIGKILL,
                expected_parent_pid=os.getppid(),
            )
        return False
    supervisor_pid = int(supervisor_pid_text)
    if not _pid_exists(supervisor_pid):
        os.kill(os.getpid(), signal.SIGKILL)
    return set_parent_death_signal(signal.SIGKILL, expected_parent_pid=os.getppid())


def initialize_worker_parent_death_signal(_worker_id=None):
    """DataLoader/ProcessPool initializer; must remain top-level and picklable."""
    return set_parent_death_signal(signal.SIGKILL, expected_parent_pid=os.getppid())
