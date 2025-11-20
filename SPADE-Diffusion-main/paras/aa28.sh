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

    # 按顺序处理内容：
    # 1. 保留原第一行
    # 2. 创建修改后的第二行
    # 3. 追加原第二行作为第三行
    {
        sed -n '1p' "$filename"  # 原第一行
        sed -n '1p' "$filename" | sed 's/Final split ratio/& r/'  # 修改后的第二行
        sed -n '2p' "$filename"  # 原第二行（新第三行）
    } > "$tmpfile"

    # 保留原文件权限覆盖
    mv "$tmpfile" "$filename"
    
    echo "已重构文件: $filename"
done
