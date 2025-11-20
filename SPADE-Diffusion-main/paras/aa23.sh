#!/bin/sh

# 定义输入文件
input_file="spatial_val.txt"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "文件不存在: $input_file"
    exit 1
fi

# 定义关键词和对应的输出
keywords="next to:位置没问题
on the top of:位置没问题
on side of:位置没问题
on the left of:位置没问题
near:位置没问题
on the right of:位置需要更换
on the bottom of:位置需要更换"

# 初始化行号计数器
line_number=1

# 逐行处理文件
while IFS= read -r line; do
    # 遍历关键词
    echo "$keywords" | while IFS= read -r keyword_pair; do
        keyword=$(echo "$keyword_pair" | cut -d ':' -f 1)
        output=$(echo "$keyword_pair" | cut -d ':' -f 2)

        # 检查当前行是否包含关键词
        if echo "$line" | grep -q "$keyword"; then
            # 如果输出是“位置需要更换”，则获取当前行号并处理对应的文件
            if [ "$output" = "位置需要更换" ]; then
                # 构造目标文件名（格式为 spatial001.txt, spatial002.txt, ...）
                filename=$(printf "spatial%03d.txt" "$line_number")
                
                # 检查目标文件是否存在
                if [ -f "$filename" ]; then
                    # 获取最后一行
                    last_line=$(tail -n 1 "$filename")
                    
                    # 检查最后一行是否以 ****** 开头
                    if echo "$last_line" | grep -q '^\*\*\*\*\*\*'; then
                        # 删除最后一行起始的 ******，保留空格
                        new_last_line=$(echo "$last_line" | sed 's/^\*\*\*\*\*\*//')
                        
                        # 用新的最后一行替换文件内容
                        sed -i '$d' "$filename"  # 删除最后一行
                        echo "$new_last_line" >> "$filename"  # 写入新的最后一行
                        
                        echo "已处理文件: $filename"
                    else
                        echo "最后一行不以 '******' 开头，跳过: $filename"
                    fi
                else
                    echo "文件不存在: $filename"
                fi
            else
                echo "$output"
            fi
            break
        fi
    done
    # 行号加 1
    line_number=$((line_number + 1))
done < "$input_file"
