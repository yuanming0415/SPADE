#!/bin/dash

# 检查是否传入参数
if [ $# -eq 0 ]; then
    echo "错误：请提供打包名称作为参数"
    echo "用法：$0 <名称>"
    exit 1
fi

# 定义路径和名称
SOURCE_DIR="/root/autodl-tmp/T2I-CompBench-main/UniDet_eval/biaozhupius"
TARGET_DIR="/root/autodl-tmp/RPG-DiffusionMaster-main/0501ex"
FILENAME="${1}_biaozhupius.tar.gz" 

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误：源目录 $SOURCE_DIR 不存在"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 打包并压缩（自动添加 .tar.gz 扩展名）
tar -czvf "$FILENAME" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# 移动打包文件
mv -v "$FILENAME" "$TARGET_DIR/"

echo "打包完成！文件保存在：$TARGET_DIR/$FILENAME"
