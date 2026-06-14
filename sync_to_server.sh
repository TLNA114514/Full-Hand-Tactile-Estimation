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

# SSH 连接参数
PORT="30719"
KEY_PATH="/code/users/jiangrui/cfzhao.pem"
REMOTE_USER="ma-user"
REMOTE_HOST="dev-modelarts.cn-north-11.huaweicloud.com"
REMOTE_DIR="/home/ma-user/work/cfzhao/"

# 本地项目目录
SRC_DIR="/code/users/jiangrui/Full-Hand-Tactile-Estimation"

# ------------------ 2. 状态检查与输出 ------------------
echo "=========================================="
echo "🌐 开始同步代码到远程服务器..."
echo "📅 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "⚙️  模式: $SYNC_METHOD"
echo "=========================================="

# 检查私钥文件是否存在
if [ ! -f "$KEY_PATH" ]; then
    echo "❌ 错误: 未找到私钥文件 $KEY_PATH"
    echo "⚠️ 请确保该文件存在并具有正确的权限 (chmod 600 $KEY_PATH)"
    exit 1
fi

# ------------------ 3. 执行同步 ------------------
if [ "$SYNC_METHOD" = "rsync" ]; then
    echo "🔄 正在使用 rsync 进行增量同步 (保持两端文件完全一致)..."
    
    # 使用 rsync 增量同步，不限制大小和类型，确保两端文件完全一致
    # （默认排除了本地 Git 数据库目录 .git/，如也需要同步，可将该行删除或注释）
    rsync -avz --delete \
        -e "ssh -p $PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
        --exclude='.git/' \
        --exclude='.DS_Store' \
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
    echo "🎉 同步成功！代码已部署至远程服务器："
    echo "👉 路径: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/Full-Hand-Tactile-Estimation/"
else
    echo "❌ 同步失败，请检查网络连接、端口或 SSH 密钥配置。"
fi
echo "=========================================="

exit $SYNC_STATUS
