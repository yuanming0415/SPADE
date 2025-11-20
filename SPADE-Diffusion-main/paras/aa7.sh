#!/bin/bash

# 读取 spatial_val.txt 的每一行
line_number=1
while IFS= read -r line; do
    # 提取当前行的前两个单词
    first_two_words=$(echo "$line" | awk '{print $1, $2}')
    
    # 构造目标文件名
    filename=$(printf "spatial%03d.txt" "$line_number")
    
    # 检查目标文件是否存在
    if [ -f "$filename" ]; then
        # 将前两个单词追加到目标文件的末尾（不换行，不加额外空格）
        echo -n "$first_two_words" >> "$filename"
        echo "已处理文件: $filename"
    else
        echo "文件不存在: $filename"
    fi
    
    # 行号加 1
    line_number=$((line_number + 1))
done < spatial_val.txt
