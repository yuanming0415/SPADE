#!/bin/bash

# 定义输入文件
input_file="spatial_val.txt"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "文件不存在: $input_file"
    exit 1
fi

# 创建一个临时文件
temp_file=$(mktemp)

# 逐行处理文件
while IFS= read -r line; do
    # 去掉每一行的第一个和第二个单词
    new_line=$(echo "$line" | awk '{for(i=3; i<=NF; i++) printf $i " "; print ""}')
    
    # 将处理后的行写入临时文件
    echo "$new_line" >> "$temp_file"
done < "$input_file"

# 用临时文件替换原文件
mv "$temp_file" "$input_file"

echo "处理完成: $input_file"
