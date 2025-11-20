#!/bin/bash

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    file_name="spatial${i}.txt"
    
    # 检查文件是否存在
    if [[ -f "$file_name" ]]; then
        # 读取第一行
        first_line=$(head -n 1 "$file_name")
        
        # 判断第一行是否包含逗号或分号
        if [[ "$first_line" == *","* ]]; then
            # 第一类，添加包含逗号的内容
            sed -i '1i\Final split ratio: 0.998,0.002' "$file_name"
        elif [[ "$first_line" == *";"* ]]; then
            # 第二类，添加包含分号的内容
            sed -i '1i\Final split ratio: 0.998;0.002' "$file_name"
        else
            echo "Warning: $file_name does not contain a comma or semicolon in the first line."
        fi
    else
        echo "Warning: $file_name does not exist."
    fi
done
