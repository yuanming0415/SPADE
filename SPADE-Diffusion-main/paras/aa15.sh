#!/bin/sh

# 定义输入文件
input_file="spatial_val.txt"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "文件不存在: $input_file"
    exit 1
fi

# 定义关键词和对应的输出
keywords="next to:左右结构
on the top of:上下结构
on side of:左右结构
on the left of:左右结构
near:左右结构
on the right of:左右结构
on the bottom of:上下结构"

# 逐行处理文件
while IFS= read -r line; do
    # 遍历关键词
    echo "$keywords" | while IFS= read -r keyword_pair; do
        keyword=$(echo "$keyword_pair" | cut -d ':' -f 1)
        output=$(echo "$keyword_pair" | cut -d ':' -f 2)
        
        # 检查当前行是否包含关键词
        if echo "$line" | grep -q "$keyword"; then
            echo "$output"
            break
        fi
    done
done < "$input_file"
