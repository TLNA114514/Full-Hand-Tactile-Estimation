#!/bin/bash

# ==============================================================================
# 🚀 sync_to_server.sh
# 仿照 sync.sh，用于将本地代码同步到 ModelArts 远程训练服务器。
# 支持手动执行，也可通过 Git post-commit hook 自动触发。
# ==============================================================================

# ------------------ 1. 配置项 ------------------
# 同步方式: 
# - "rsync" (推荐！增量同步，速度极快，且过滤大文件、数据集与临时缓存)
# - "scp"   (使用您指定的 scp 指令，直接全量覆盖复制)
SYNC_METHOD="rsync"
MODE="${1:-sync}"

# SSH 连接参数
PORT="32504"
KEY_PATH="/code/users/jiangrui/Full-Hand-Tactile-Estimation/cfzhao.pem"
REMOTE_USER="ma-user"
REMOTE_HOST="dev-modelarts.cn-north-11.huaweicloud.com"
REMOTE_DIR="/home/ma-user/work/cfzhao/"

# 本地项目目录
SRC_DIR="/code/users/jiangrui/Full-Hand-Tactile-Estimation"

RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='.nfs*'
    --exclude='.DS_Store'
    --exclude='*checkpoint*'
    --exclude='*ckpt*'
    --exclude='*weights*'
    --exclude='*.ckpt'
    --exclude='*.pt'
    --exclude='*.pth'
    --exclude='*.pkl'
    --exclude='wandb/'
    --exclude='logs/'
    --exclude='lightning_logs/'
    --exclude='hamer_tactile_ft/touchanything_bboxes_cache/'
    --exclude='hamer_tactile_ft/egotactile_bboxes_cache/'
    --exclude='hamer_tactile_ft/full_bboxes_cache/'
    --exclude='hamer_tactile_ft/test_bboxes_cache/'
    --exclude='hamer_tactile_ft/touchanything_all_bboxes.json'
    --exclude='hamer_tactile_ft/egotactile_all_bboxes.json'
    --exclude='hamer_tactile_ft/opentouch_all_bboxes.json'
    --exclude='hamer_tactile_ft/opentouch_test_bboxes.json'
    --exclude='hamer_tactile_ft/dataset_frames_registry.json'
    --exclude='hamer_tactile_infiller/manifests'
    --exclude='preprocess/artifacts/'
    --exclude='outputs'
)

# ------------------ 2. 状态检查与输出 ------------------
echo "=========================================="
if [ "$MODE" = "--check" ] || [ "$MODE" = "check" ]; then
    echo "🌐 开始检查远程服务器连接..."
else
    echo "🌐 开始同步代码到远程服务器..."
fi
echo "📅 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "⚙️  模式: $SYNC_METHOD"
echo "=========================================="

# 检查私钥文件是否存在
if [ ! -f "$KEY_PATH" ]; then
    echo "❌ 错误: 未找到私钥文件 $KEY_PATH"
    echo "⚠️ 请确保该文件存在并具有正确的权限 (chmod 600 $KEY_PATH)"
    exit 1
fi

if [ "$MODE" = "--check" ] || [ "$MODE" = "check" ]; then
    echo "🔑 私钥存在: $KEY_PATH"
    echo "🔐 私钥权限: $(stat -c '%a %U %G' "$KEY_PATH" 2>/dev/null || stat -f '%Lp %Su %Sg' "$KEY_PATH")"

    echo "🧭 DNS 解析测试: $REMOTE_HOST"
    getent hosts "$REMOTE_HOST" || {
        echo "❌ DNS 解析失败。请检查本机网络/DNS，或稍后重试。"
        exit 2
    }

    echo "🔌 SSH 连接测试..."
    ssh -p "$PORT" \
        -i "$KEY_PATH" \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=no \
        "$REMOTE_USER@$REMOTE_HOST" \
        "echo SSH_OK && hostname && test -d '$REMOTE_DIR' && echo REMOTE_DIR_OK"
    SSH_STATUS=$?
    if [ $SSH_STATUS -ne 0 ]; then
        echo "❌ SSH 连接失败。请检查端口、密钥、用户名或 ModelArts 实例状态。"
        exit $SSH_STATUS
    fi

    echo "🧪 rsync dry-run 测试..."
    rsync -avn --delete \
        -e "ssh -p $PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
        "${RSYNC_EXCLUDES[@]}" \
        "$SRC_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/Full-Hand-Tactile-Estimation/"
    RSYNC_STATUS=$?
    if [ $RSYNC_STATUS -ne 0 ]; then
        echo "❌ rsync dry-run 失败。请查看上面的具体 rsync 错误。"
        exit $RSYNC_STATUS
    fi

    echo "✅ 连接、目标目录和 rsync dry-run 都通过。"
    exit 0
fi

# ------------------ 3. 执行同步 ------------------
if [ "$SYNC_METHOD" = "rsync" ]; then
    echo "🔄 正在使用 rsync 进行增量同步 (保持两端文件完全一致)..."
    
    # 使用 rsync 增量同步，不限制文件大小，但排除了 checkpoint 和训练日志相关目录
    # 这样可以豁免对远端服务器训练生成的权重、日志的 --delete 操作，防止远端训练结果被清空
    rsync -avz --delete \
        -e "ssh -p $PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
        "${RSYNC_EXCLUDES[@]}" \
        "$SRC_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/Full-Hand-Tactile-Estimation/"

    SYNC_STATUS=$?
else
    echo "📦 正在使用 scp 进行全量复制..."
    
    # 使用用户指定的原始 scp 命令
    scp -P "$PORT" -i "$KEY_PATH" -o StrictHostKeyChecking=no -r "$SRC_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
    
    SYNC_STATUS=$?
fi

# ------------------ 4. 结果处理 ------------------
echo "=========================================="
if [ $SYNC_STATUS -eq 0 ]; then
    echo "🎉 同步成功！正在远端服务器上触发自动路径适配..."
    ssh -p "$PORT" -i "$KEY_PATH" -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_DIR/Full-Hand-Tactile-Estimation && python3 auto_adapt_paths.py"
    
    echo "👉 路径已全部自动修改并部署至: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/Full-Hand-Tactile-Estimation/"
else
    echo "❌ 同步失败，请检查网络连接、端口或 SSH 密钥配置。"
fi
echo "=========================================="

exit $SYNC_STATUS
