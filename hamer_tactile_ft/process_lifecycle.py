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


def _distributed_global_rank_from_environment():
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        value = os.environ.get(name, "").strip()
        if value:
            return int(value)
    local_rank = os.environ.get("LOCAL_RANK", "").strip()
    if local_rank:
        node_rank = int(
            os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "0"))
        )
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE", "").strip()
        if node_rank and not local_world_size:
            raise RuntimeError(
                "Cannot reconstruct global rank from LOCAL_RANK on a multi-node "
                "worker without LOCAL_WORLD_SIZE"
            )
        return node_rank * int(local_world_size or "1") + int(local_rank)
    return 0


def initialize_worker_historical_lightning_seed(worker_id):
    """Restore the Lightning 2.1 worker RNG path and retain parent-death safety."""

    set_parent_death_signal(signal.SIGKILL, expected_parent_pid=os.getppid())
    try:
        from lightning_fabric.utilities.seed import pl_worker_init_function
    except ImportError:
        from pytorch_lightning.utilities.seed import pl_worker_init_function
    # This must run after prctl: the historical Lightning callback owns the
    # final Python, NumPy, and Torch worker RNG state.
    return pl_worker_init_function(
        int(worker_id),
        rank=_distributed_global_rank_from_environment(),
    )
