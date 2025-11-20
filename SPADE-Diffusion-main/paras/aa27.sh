#!/bin/bash

for i in {1..300}
do
    # 生成三位数序号
    num=$(printf "%03d" "$i")
    filename="spatial${num}.txt"
    
    # 检查文件是否存在
    if [ ! -f "$filename" ]; then
        echo "警告：文件 $filename 不存在，跳过处理"
        continue
    fi

    # 使用临时文件交换行内容
    tmpfile=$(mktemp)
    
    # 先写第二行（行尾自动换行）
    sed -n '2p' "$filename" > "$tmpfile"
    # 追加第一行（行尾自动换行）
    sed -n '1p' "$filename" >> "$tmpfile"
    
    # 覆盖原文件（保留原文件权限）
    mv "$tmpfile" "$filename"
    
    echo "已处理文件: $filename"
done
