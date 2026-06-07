#!/bin/bash

# 1. 定义源（大仓）与目标（GitHub仓）的根目录
SRC_BASE="/code/users/jiangrui/Full-Hand-Tactile-Estimation"
DEST_BASE="/code/users/jiangrui/full_hand_tactile_estimation"

# 2. 【核心配置项】在这里指定需要同步的文件夹（支持相对路径）
# 以后如果想同步新的文件夹，直接在这里换行添加名字即可！
SYNC_DIRS=(
    "evaluation"
    "opentouch_hamer_ft"
    " hamer_tactile_ft "
    # "未来你想同步的新文件夹A"
    # "未来你想同步的新文件夹B"
)

echo "=========================================="
echo "🚀 开始同步文件夹到 GitHub 仓库..."
echo "📅 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 3. 循环同步
for DIR in "${SYNC_DIRS[@]}"; do
    if [ -d "$SRC_BASE/$DIR" ]; then
        echo "🔄 正在同步: $DIR"
        echo "   源路径: $SRC_BASE/$DIR"
        echo "   目标路径: $DEST_BASE/$DIR"
        
        # 确保目标父目录存在
        mkdir -p "$(dirname "$DEST_BASE/$DIR")"
        
        # 使用 rsync 增量同步，排除 .git 文件、缓存文件、模型权重/Checkpoints 以及大文件
        rsync -av --delete \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.DS_Store' \
            --exclude='*checkpoint*' \
            --exclude='*ckpt*' \
            --exclude='*weights*' \
            --exclude='*.ckpt' \
            --exclude='*.pt' \
            --exclude='*.pth' \
            --exclude='*.pkl' \
            --exclude='wandb/' \
            --exclude='logs/' \
            --exclude='lightning_logs/' \
            --exclude='*.h5' \
            --exclude='*.hdf5' \
            --exclude='*.npy' \
            --exclude='*.npz' \
            --exclude='*.mp4' \
            --exclude='*.png' \
            --exclude='*.jpg' \
            --exclude='*.jpeg' \
            --exclude='*.gif' \
            --max-size='10m' \
            "$SRC_BASE/$DIR/" "$DEST_BASE/$DIR/"
            
        echo "   ✅ $DIR 同步完成"
    else
        echo "⚠️ 警告: 源目录 $SRC_BASE/$DIR 不存在，已跳过。"
    fi
done

echo "=========================================="
echo "🎉 所有指定目录同步完毕！"
echo "👉 你现在可以去目标仓库查看更新并执行 git 提交："
echo "   cd $DEST_BASE"
echo "=========================================="
