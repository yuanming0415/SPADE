#!/bin/bash

# 遍历文件 spatial001.txt 到 spatial300.txt
for i in $(seq 1 300); do
    # 格式化为3位数字（如 1 → 001）
    num=$(printf "%03d" "$i")
    filename="spatial${num}.txt"

    # 检查文件是否存在
    if [ ! -f "$filename" ]; then
        echo "警告：文件 ${filename} 不存在，跳过"
        continue
    fi

    # 使用临时文件处理内容
    sed '2d; 3{h;d}; 2G' "$filename" > "${filename}.tmp" && mv "${filename}.tmp" "$filename"
done
