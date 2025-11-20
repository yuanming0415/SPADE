#!/bin/sh

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    file_name="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$file_name" ]; then
        # 使用 sed 删除第一行
        sed '1d' "$file_name" > "${file_name}.tmp" && mv "${file_name}.tmp" "$file_name"
    else
        echo "Warning: $file_name does not exist."
    fi
done
