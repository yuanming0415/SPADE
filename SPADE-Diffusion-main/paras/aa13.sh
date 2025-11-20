#!/bin/bash

# 定义输入文件
input_file="spatial_val.txt"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "文件不存在: $input_file"
    exit 1
fi

# 定义关键词和对应的输出
declare -A keywords
keywords=(
    ["next to"]="位置没问题"
    ["on the top of"]="位置没问题"
    ["on side of"]="位置没问题"
    ["on the left of"]="位置没问题"
    ["near"]="位置没问题"
    ["on the right of"]="位置需要更换"
    ["on the bottom of"]="位置需要更换"
)

# 逐行处理文件
while IFS= read -r line; do
    # 遍历关键词
    for keyword in "${!keywords[@]}"; do
        # 检查当前行是否包含关键词
        if [[ "$line" == *"$keyword"* ]]; then
            echo "${keywords[$keyword]}"
            break
        fi
    done
done < "$input_file"
