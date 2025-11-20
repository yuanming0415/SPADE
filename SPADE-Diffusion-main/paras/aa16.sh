#!/bin/sh

# 定义输入文件
input_file="spatial_val.txt"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "文件不存在: $input_file"
    exit 1
fi

# 定义关键词
keywords="on the right of
on the bottom of"

# 逐行处理文件
line_number=1
while IFS= read -r line; do
    # 检查当前行是否包含关键词
    for keyword in $keywords; do
        if echo "$line" | grep -q "$keyword"; then
            # 构造目标文件名
            filename=$(printf "spatial%03d.txt" "$line_number")
            
            # 检查目标文件是否存在
            if [ -f "$filename" ]; then
                # 在目标文件末尾添加 ***
                echo -n "***" >> "$filename"
                echo "已处理文件: $filename"
            else
                echo "文件不存在: $filename"
            fi
            break
        fi
    done
    # 行号加 1
    line_number=$((line_number + 1))
done < "$input_file"
