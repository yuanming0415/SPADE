#!/bin/sh

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    file_name="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$file_name" ]; then
        # 使用 sed 修改第三行
        sed -i '3s/Final split ratio/Final split ratio all/' "$file_name"
    else
        echo "Warning: $file_name does not exist."
    fi
done
