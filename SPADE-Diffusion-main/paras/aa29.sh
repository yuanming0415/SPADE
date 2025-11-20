#!/bin/bash

for i in {1..300}
do
    # 生成三位数序号
    num=$(printf "%03d" "$i")
    filename="spatial${num}.txt"
    
    # 检查文件存在性
    if [ ! -f "$filename" ]; then
        echo "警告：文件 $filename 不存在，跳过处理"
        continue
    fi

    # 创建临时文件
    tmpfile=$(mktemp)

    # 按行处理：
    # 1. 第一行替换两个数字为 0.998 和 0.002
    # 2. 第二行替换两个数字为 0.002 和 0.998
    sed '
    1s/\([[:digit:].]\+\)[,]\([[:digit:].]\+\)/0.998,0.002/
    2s/\([[:digit:].]\+\)[,]\([[:digit:].]\+\)/0.002,0.998/
    ' "$filename" > "$tmpfile"

    # 保留原文件权限覆盖
    mv "$tmpfile" "$filename"
    
    echo "已处理文件: $filename"
done
