#!/bin/dash

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误：请提供打包名称作为参数"
    echo "用法：$0 <名称>"
    exit 1
fi

# 定义公共路径
TARGET_DIR="/root/autodl-tmp/RPG-DiffusionMaster-main/0501ex"
PARAM="$1"  # 带括号的参数

# 定义三个数据集配置
DATASETS="\
biaozhupius:/root/autodl-tmp/T2I-CompBench-main/UniDet_eval/biaozhupius:\${PARAM}_biaozhupius.tar.gz
samples:/root/autodl-tmp/T2I-CompBench-main/examples/samples:\${PARAM}_samples.tar.gz
bakimg:/root/autodl-tmp/RPG-DiffusionMaster-main/bakimg:\${PARAM}_bakimg.tar.gz"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 遍历所有数据集
echo "$DATASETS" | while IFS=: read -r name source_path filename; do
    # 替换变量得到实际文件名
    eval "filename=$filename"
    
    # 检查源目录
    if [ ! -d "$source_path" ]; then
        echo "错误：$name 的源目录不存在 $source_path"
        exit 2
    fi

    # 打包操作
    echo "正在打包 $name..."
    tar -czf "$filename" -C "$(dirname "$source_path")" "$(basename "$source_path")"
    
    # 移动文件
    if mv -f "$filename" "$TARGET_DIR/"; then
        echo "已生成：$TARGET_DIR/$filename"
    else
        echo "移动 $name 打包文件失败"
        exit 3
    fi
done

echo "全部打包完成！三套数据已保存到：$TARGET_DIR"
