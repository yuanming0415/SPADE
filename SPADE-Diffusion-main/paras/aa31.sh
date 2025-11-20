#!/bin/bash

# 遍历所有文件（spatial001.txt 到 spatial300.txt）
for i in {1..300}; do
    # 格式化数字为3位（如 1 → 001）
    num=$(printf "%03d" "$i")
    filename="spatial${num}.txt"

    # 使用sed删除第二行，并将第三行移动到第二行位置
    sed -i '2d; 3s/.*//; 3g; 3N; D' "$filename"

    # 以下备用方案（兼容性更好）：
    # sed -i '2d; 3{h;d}; 2G' "$filename"
done
