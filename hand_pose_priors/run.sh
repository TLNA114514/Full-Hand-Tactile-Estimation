#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HAMER_ROOT="${HAMER_ROOT:-$REPO_ROOT/hamer}"
HAMER_PYTHON="${HAMER_PYTHON:-/home/ma-user/work/cfzhao/tactile/bin/python}"
HAMER_ARCHIVE="${HAMER_ARCHIVE:-$HAMER_ROOT/hamer_demo_data.tar.gz}"
HAMER_DATA_URL="${HAMER_DATA_URL:-https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz}"
HAMER_CHECKPOINT="${HAMER_CHECKPOINT:-$HAMER_ROOT/_DATA/hamer_ckpts/checkpoints/hamer.ckpt}"
HAMER_MANO="${HAMER_MANO:-$HAMER_ROOT/_DATA/data/mano/MANO_RIGHT.pkl}"
HAMER_SIDECAR_ROOT="${HAMER_SIDECAR_ROOT:-/home/ma-user/work/cfzhao/hand_pose_sidecars/touchanything_hamer_v1}"
DYNHAMR_WORK_ROOT="${DYNHAMR_WORK_ROOT:-/home/ma-user/work/cfzhao/hand_pose_dynhamr}"
DYNHAMR_CHECKOUT="${DYNHAMR_CHECKOUT:-$DYNHAMR_WORK_ROOT/code/Dyn-HaMR}"
DYNHAMR_TRIAL_ROOT="${DYNHAMR_TRIAL_ROOT:-$DYNHAMR_WORK_ROOT/trials/touchanything_arrange_pillow_v3}"
DYNHAMR_RUN_NAME="${DYNHAMR_RUN_NAME:-static_focal_standard_v1}"
DYNHAMR_COMMIT="${DYNHAMR_COMMIT:-fa9cd7412c205fd15ee4139c8caacf79bf6167e6}"
DYNHAMR_REPOSITORY="${DYNHAMR_REPOSITORY:-https://github.com/ZhengdiYu/Dyn-HaMR.git}"
DYNHAMR_GPU="${DYNHAMR_GPU:-0}"

mode="${1:-}"
if [[ -z "$mode" ]]; then
    printf 'Usage: %s setup-hamer|check-hamer|smoke-hamer|build-hamer|verify-hamer|status-hamer|visualize-hamer|setup-dynhamr|prepare-dynhamr|check-dynhamr|run-dynhamr|audit-dynhamr [arguments...]\n' "$0" >&2
    exit 2
fi
shift

if [[ ! -x "$HAMER_PYTHON" ]]; then
    printf 'HaMeR Python is not executable: %s\n' "$HAMER_PYTHON" >&2
    exit 1
fi

setup_hamer() {
    if [[ ! -s "$HAMER_CHECKPOINT" ]]; then
        mkdir -p "$HAMER_ROOT"
        if [[ ! -s "$HAMER_ARCHIVE" ]]; then
            printf '[hand-pose] Downloading official HaMeR demo assets to %s\n' "$HAMER_ARCHIVE"
            curl --fail --location --retry 5 --retry-all-errors \
                --continue-at - --output "$HAMER_ARCHIVE" "$HAMER_DATA_URL"
        fi
        printf '[hand-pose] Extracting HaMeR assets under %s/_DATA\n' "$HAMER_ROOT"
        tar --warning=no-unknown-keyword -xzf "$HAMER_ARCHIVE" -C "$HAMER_ROOT"
    fi
    exec "$HAMER_PYTHON" "$SCRIPT_DIR/hamer_sam3_smoke.py" \
        --checkpoint "$HAMER_CHECKPOINT" --mano "$HAMER_MANO" \
        --check-only "$@"
}

require_hamer_assets() {
    if [[ ! -s "$HAMER_CHECKPOINT" || ! -s "$HAMER_MANO" ]]; then
        printf 'HaMeR assets are missing. Run %s setup-hamer first.\n' "$0" >&2
        exit 1
    fi
}

gpu_count() {
    if [[ -n "${HAMER_GPUS:-}" ]]; then
        printf '%s\n' "$HAMER_GPUS"
        return
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local visible="${CUDA_VISIBLE_DEVICES// /}"
        if [[ -n "$visible" && "$visible" != "-1" ]]; then
            local commas="${visible//[^,]/}"
            printf '%s\n' "$(( ${#commas} + 1 ))"
            return
        fi
    fi
    "$HAMER_PYTHON" -c 'import torch; print(torch.cuda.device_count())'
}

setup_dynhamr() {
    mkdir -p "$(dirname "$DYNHAMR_CHECKOUT")"
    marker="$DYNHAMR_CHECKOUT/.dynhamr_commit"
    entrypoint="$DYNHAMR_CHECKOUT/dyn-hamr/run_opt.py"
    if [[ ! -s "$entrypoint" && ! -d "$DYNHAMR_CHECKOUT/.git" ]]; then
        printf '[dynhamr-setup] Cloning pinned source without large tracked media: %s\n' \
            "$DYNHAMR_CHECKOUT"
        git clone --filter=blob:none --no-checkout "$DYNHAMR_REPOSITORY" "$DYNHAMR_CHECKOUT"
        if git -C "$DYNHAMR_CHECKOUT" sparse-checkout --help >/dev/null 2>&1; then
            git -C "$DYNHAMR_CHECKOUT" sparse-checkout init --cone
            git -C "$DYNHAMR_CHECKOUT" sparse-checkout set \
                dyn-hamr README.md LICENSE requirements.txt setup.py
            git -C "$DYNHAMR_CHECKOUT" checkout --detach "$DYNHAMR_COMMIT"
        else
            printf 'Git sparse-checkout is unavailable. Sync a pinned source snapshot and write %s.\n' \
                "$marker" >&2
            exit 1
        fi
    fi
    if [[ -s "$marker" ]]; then
        actual_commit="$(tr -d '[:space:]' < "$marker")"
    else
        actual_commit="$(git -C "$DYNHAMR_CHECKOUT" rev-parse HEAD)"
    fi
    if [[ "$actual_commit" != "$DYNHAMR_COMMIT" ]]; then
        printf 'Dyn-HaMR checkout is at %s, expected %s: %s\n' \
            "$actual_commit" "$DYNHAMR_COMMIT" "$DYNHAMR_CHECKOUT" >&2
        exit 1
    fi
    if [[ ! -s "$entrypoint" ]]; then
        printf 'Dyn-HaMR entrypoint is missing from the pinned checkout: %s\n' "$entrypoint" >&2
        exit 1
    fi
    "$HAMER_PYTHON" "$SCRIPT_DIR/patch_dynhamr_runtime.py" \
        --root "$DYNHAMR_CHECKOUT"
    for data_parent in "$DYNHAMR_CHECKOUT/_DATA" "$DYNHAMR_CHECKOUT/dyn-hamr/_DATA"; do
        mkdir -p "$data_parent"
        data_link="$data_parent/data"
        if [[ -e "$data_link" && ! -L "$data_link" ]]; then
            printf 'Refusing to replace non-symlink Dyn-HaMR data directory: %s\n' "$data_link" >&2
            exit 1
        fi
        ln -sfn "$HAMER_ROOT/_DATA/data" "$data_link"
    done
    DYNHAMR_SOURCE="$DYNHAMR_CHECKOUT/dyn-hamr" "$HAMER_PYTHON" - <<'PY'
import importlib.util
import os
import sys

required = ("torch", "numpy", "scipy", "cv2", "smplx", "trimesh", "pyrender", "hydra", "omegaconf", "tensorboard")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(f"Dyn-HaMR runtime is missing Python packages: {missing}")
sys.path.insert(0, os.environ["DYNHAMR_SOURCE"])
from body_model import MANO  # noqa: F401
print("[dynhamr-setup] Core imports passed")
PY
    printf '[dynhamr-setup] Ready: %s (%s)\n' "$DYNHAMR_CHECKOUT" "$actual_commit"
}

case "$mode" in
    -h|--help|help)
        printf 'Usage: %s setup-hamer|check-hamer|smoke-hamer|build-hamer|verify-hamer|status-hamer|visualize-hamer|setup-dynhamr|prepare-dynhamr|check-dynhamr|run-dynhamr|audit-dynhamr [arguments...]\n' "$0"
        ;;
    setup-hamer)
        setup_hamer "$@"
        ;;
    check-hamer)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/hamer_sam3_smoke.py" \
            --checkpoint "$HAMER_CHECKPOINT" --mano "$HAMER_MANO" \
            --check-only "$@"
        ;;
    smoke-hamer)
        require_hamer_assets
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/hamer_sam3_smoke.py" \
            --checkpoint "$HAMER_CHECKPOINT" --mano "$HAMER_MANO" "$@"
        ;;
    build-hamer)
        require_hamer_assets
        workers="$(gpu_count)"
        if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
            printf 'No CUDA workers were selected; set CUDA_VISIBLE_DEVICES or HAMER_GPUS.\n' >&2
            exit 1
        fi
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
        printf '[hand-pose] Building HaMeR sidecar with %s GPU process(es): %s\n' \
            "$workers" "$HAMER_SIDECAR_ROOT"
        exec "$HAMER_PYTHON" -m torch.distributed.run \
            --standalone --nproc_per_node="$workers" \
            "$SCRIPT_DIR/export_hamer_sidecars.py" build \
            --checkpoint "$HAMER_CHECKPOINT" \
            --mano "$HAMER_MANO" \
            --output-dir "$HAMER_SIDECAR_ROOT" \
            "$@"
        ;;
    verify-hamer)
        require_hamer_assets
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/export_hamer_sidecars.py" verify \
            --output-dir "$HAMER_SIDECAR_ROOT" "$@"
        ;;
    status-hamer)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/export_hamer_sidecars.py" inspect \
            --output-dir "$HAMER_SIDECAR_ROOT" "$@"
        ;;
    visualize-hamer)
        require_hamer_assets
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/visualize_hamer_sidecars.py" \
            --sidecar-root "$HAMER_SIDECAR_ROOT" \
            --mano "$HAMER_MANO" "$@"
        ;;
    self-test)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/export_hamer_sidecars.py" self-test "$@"
        ;;
    setup-dynhamr)
        setup_dynhamr
        ;;
    prepare-dynhamr)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/prepare_dynhamr_touchanything.py" \
            --sidecar-root "$HAMER_SIDECAR_ROOT" \
            --mano "$HAMER_MANO" \
            --output-root "$DYNHAMR_TRIAL_ROOT" \
            "$@"
        ;;
    check-dynhamr)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/launch_dynhamr.py" \
            --checkout "$DYNHAMR_CHECKOUT" \
            --trial-root "$DYNHAMR_TRIAL_ROOT" \
            --run-name "$DYNHAMR_RUN_NAME" \
            --gpu "$DYNHAMR_GPU" \
            --validate-only \
            "$@"
        ;;
    run-dynhamr)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/launch_dynhamr.py" \
            --checkout "$DYNHAMR_CHECKOUT" \
            --trial-root "$DYNHAMR_TRIAL_ROOT" \
            --run-name "$DYNHAMR_RUN_NAME" \
            --gpu "$DYNHAMR_GPU" \
            "$@"
        ;;
    audit-dynhamr)
        exec "$HAMER_PYTHON" "$SCRIPT_DIR/audit_dynhamr_trial.py" \
            --checkout "$DYNHAMR_CHECKOUT" \
            --trial-root "$DYNHAMR_TRIAL_ROOT" \
            --run-dir "$DYNHAMR_TRIAL_ROOT/outputs/$DYNHAMR_RUN_NAME" \
            --mano "$HAMER_MANO" \
            "$@"
        ;;
    *)
        printf 'Unknown mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac
