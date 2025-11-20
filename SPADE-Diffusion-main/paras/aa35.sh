#!/bin/bash

# 确保在 Bash 中运行，但如果需要兼容 /bin/sh，可以修改为以下方式（不过这里还是推荐用 Bash）
# 如果坚持用 /bin/sh，可以将 [[ ]] 替换为 [ ]，但这里直接用 Bash 语法

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    file_name="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$file_name" ]; then
        # 读取第一行
        first_line=$(head -n 1 "$file_name")
        
        # 判断第一行是否包含逗号或分号
        if echo "$first_line" | grep -q ","; then
            # 第一类，添加包含逗号的内容
            sed -i '1i\Final split ratio: 0.998,0.002' "$file_name"
        elif echo "$first_line" | grep -q ";"; then
            # 第二类，添加包含分号的内容
            sed -i '1i\Final split ratio: 0.998;0.002' "$file_name"
        else
            echo "Warning: $file_name does not contain a comma or semicolon in the first line."
        fi
    else
        echo "Warning: $file_name does not exist."
    fi
done
