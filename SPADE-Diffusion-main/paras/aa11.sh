#!/bin/bash

# 定义输入文件
input_file="spatial_val.txt"

# 检查输入文件是否存在
if [ ! -f "$input_file" ]; then
    echo "文件不存在: $input_file"
    exit 1
fi

# 逐行处理 spatial_val.txt
line_number=1
while IFS= read -r line; do
    # 提取当前行的最后两个单词
    last_two_words=$(echo "$line" | awk '{print $(NF-1), $NF}')
    
    # 构造目标文件名
    filename=$(printf "spatial%03d.txt" "$line_number")
    
    # 检查目标文件是否存在
    if [ -f "$filename" ]; then
        # 将最后两个单词追加到目标文件的末尾（不换行，不加额外空格）
        echo -n "$last_two_words" >> "$filename"
        echo "已处理文件: $filename"
    else
        echo "文件不存在: $filename"
    fi
    
    # 行号加 1
    line_number=$((line_number + 1))
done < "$input_file"
