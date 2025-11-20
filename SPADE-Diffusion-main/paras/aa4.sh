#!/bin/bash

# 遍历 spatial001.txt 到 spatial300.txt
for i in {1..300}
do
    # 格式化文件名，确保是三位数
    filename=$(printf "spatial%03d.txt" $i)
    
    # 检查文件是否存在
    if [ -f "$filename" ]; then
        # 使用 sed 删除 ^M 字符
        sed -i 's/\r//g' "$filename"
        echo "已处理文件: $filename"
    else
        echo "文件不存在: $filename"
    fi
done
