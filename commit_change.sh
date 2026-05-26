#!/bin/bash

# 检查是否传入了提交信息
if [ -z "$1" ]; then
    echo "错误：请输入提交信息作为参数。"
    echo "用法：./commit.sh \"提交信息\""
    exit 1
fi

# 定义提交信息
commit_message="$1"

# 检查是否是 Git 仓库
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "错误：当前目录不是一个 Git 仓库！"
    exit 1
fi

# 获取当前分支名
current_branch=$(git rev-parse --abbrev-ref HEAD)

# 显示当前修改状态
echo "当前修改文件："
git status --short

# 检查是否有待提交的更改
if [ -z "$(git status --porcelain)" ]; then
    echo -e "\n没有需要提交的更改。"
    exit 0
fi

# 提示当前所在分支并等待用户确认
echo -e "\n当前所在分支为：$current_branch"
echo "按 Enter 键确认上传本次更改，或 Ctrl+C 取消操作..."
read -r 

# 添加所有更改
git add .

# 提交更改
if git commit -m "$commit_message"; then
    echo "提交成功！"
else
    echo "错误：提交失败！"
    exit 1
fi

# 推送更改到远程仓库的当前分支
if git push origin "$current_branch"; then
    echo -e "\n✓ 提交并推送到分支 $current_branch 完成！"
else
    echo -e "\n错误：推送失败！"
    exit 1
fi