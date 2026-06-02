#!/bin/bash

# ==============================================================================
# Hamer & HaWoR OpenTouch 数据集一键式指标评估启动脚本
# Conda 环境要求: opentouch (Python 3.10)
# ==============================================================================

# 获取脚本所在文件夹的绝对路径
EVAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$EVAL_DIR"

# 默认参数 (单场景单片段评估时使用)
HDF5_PATH="../opentouch/data/fablab_ml_p1.hdf5"
CLIPS="demo_05 demo_10 demo_15"

print_usage() {
    echo "用法: ./run_eval.sh [hamer|hawor|both] [GPU_ID] [split]"
    echo "例如: "
    echo "  ./run_eval.sh hamer 4         # 对默认片段进行 Hamer 评估"
    echo "  ./run_eval.sh hamer 4 test    # 对 test 划分进行 Hamer 评估"
    echo "  ./run_eval.sh hawor 5 val     # 对 val 划分进行 HaWoR 评估"
}

if [ -z "$1" ]; then
    print_usage
    exit 1
fi

MODEL_TYPE=$1
GPU_ID=${2:-4} # 默认 GPU 4
SPLIT=$3       # 评估的数据集划分 (可选: train, val, test, all)

echo "======================================================================"
echo "🎯 Hamer & HaWoR 自动评估工具已就绪"
if [ -n "$SPLIT" ]; then
    echo "📂 评测模式: 数据集划分划分评估 (Split: $SPLIT)"
else
    echo "📂 评测模式: 指定文件及片段评估"
    echo "📂 数据集路径: $HDF5_PATH"
    echo "🎥 评估片段: $CLIPS"
fi
echo "⚙️ GPU 设备: $GPU_ID"
echo "======================================================================"

# 激活 opentouch conda 环境
echo "💡 请确保您已激活 opentouch 环境: conda activate opentouch"

if [ "$MODEL_TYPE" == "hamer" ]; then
    if [ -n "$SPLIT" ]; then
        echo "🚀 启动 Hamer 评估 (GPU: $GPU_ID, Split: $SPLIT)..."
        python eval_hamer.py \
            --checkpoint "../hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
            --split "$SPLIT" \
            --gpu "$GPU_ID"
    else
        echo "🚀 启动 Hamer 评估 (GPU: $GPU_ID, Clips: $CLIPS)..."
        python eval_hamer.py \
            --checkpoint "../hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
            --hdf5_path "$HDF5_PATH" \
            --clips $CLIPS \
            --gpu "$GPU_ID"
    fi

elif [ "$MODEL_TYPE" == "hawor" ]; then
    if [ -n "$SPLIT" ]; then
        echo "🚀 启动 HaWoR 评估 (GPU: $GPU_ID, Split: $SPLIT)..."
        python eval_hawor.py \
            --checkpoint "../HaWoR/weights/hawor/checkpoints/hawor.ckpt" \
            --split "$SPLIT" \
            --gpu "$GPU_ID" \
            --img_focal 493
    else
        echo "🚀 启动 HaWoR 评估 (GPU: $GPU_ID, Clips: $CLIPS)..."
        python eval_hawor.py \
            --checkpoint "../HaWoR/weights/hawor/checkpoints/hawor.ckpt" \
            --hdf5_path "$HDF5_PATH" \
            --clips $CLIPS \
            --gpu "$GPU_ID" \
            --img_focal 493
    fi

elif [ "$MODEL_TYPE" == "both" ]; then
    echo "🚀 同时并行启动两款模型评估..."
    
    if [ -n "$SPLIT" ]; then
        echo "👉 Hamer 将在 GPU 4 上运行 (Split: $SPLIT)..."
        python eval_hamer.py \
            --checkpoint "../hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
            --split "$SPLIT" \
            --gpu "4" &

        echo "👉 HaWoR 将在 GPU 5 上运行 (Split: $SPLIT)..."
        python eval_hawor.py \
            --checkpoint "../HaWoR/weights/hawor/checkpoints/hawor.ckpt" \
            --split "$SPLIT" \
            --gpu "5" \
            --img_focal 493 &
    else
        echo "👉 Hamer 将在 GPU 4 上运行 (Clips: $CLIPS)..."
        python eval_hamer.py \
            --checkpoint "../hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
            --hdf5_path "$HDF5_PATH" \
            --clips $CLIPS \
            --gpu "4" &

        echo "👉 HaWoR 将在 GPU 5 上运行 (Clips: $CLIPS)..."
        python eval_hawor.py \
            --checkpoint "../HaWoR/weights/hawor/checkpoints/hawor.ckpt" \
            --hdf5_path "$HDF5_PATH" \
            --clips $CLIPS \
            --gpu "5" \
            --img_focal 493 &
    fi
    
    wait
    echo "✅ 两款模型评估全部结束！"
else
    print_usage
    exit 1
fi
