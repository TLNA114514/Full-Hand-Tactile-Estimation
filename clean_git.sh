#!/bin/bash
# clean_git.sh
# 这是一个一键清理子项目 Git 仓库并在 US.T 根目录下重新初始化大仓库的脚本。

set -e

# 获取脚本所在的根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=== 开始清理子目录的 Git 仓库 ==="

# 查找所有子目录中的 .git 目录（排除根目录本身）
find . -mindepth 2 -name ".git" -type d | while read -r git_dir; do
    echo "正在清除子项目 Git 目录: $git_dir"
    rm -rf "$git_dir"
done

# 清理子目录中的 .gitignore 和 .gitmodules 以免干扰根目录的 Git 规则
find . -mindepth 2 -name ".gitignore" -type f | while read -r gitignore_file; do
    echo "正在清除冗余的子项目 .gitignore 文件: $gitignore_file"
    rm -f "$gitignore_file"
done

find . -mindepth 2 -name ".gitmodules" -type f | while read -r gitmodules_file; do
    echo "正在清除冗余的子项目 .gitmodules 文件: $gitmodules_file"
    rm -f "$gitmodules_file"
done

echo "=== 子目录 Git 清理完成 ==="
echo ""
echo "=== 初始化根 Git 仓库 ==="

if [ -d ".git" ]; then
    echo "检测到根目录已存在 .git，跳过初始化。"
else
    git init
    echo "成功在 $ROOT_DIR 初始化根 Git 仓库。"
fi

echo ""
echo "=== 正在检查当前 Git 状态 ==="
git status

echo ""
echo "=== 完成！ ==="
echo "提示：请接下来检查您的 .gitignore 文件，以确保没有大文件或 checkpoint 被错误地加入暂存区。"
