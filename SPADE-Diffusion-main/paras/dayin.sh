#!/bin/sh

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    # 构造文件名
    filename="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$filename" ]; then
        cat "$filename"
        echo ""  # 打印空行分隔文件内容
    else
        echo "文件不存在: $filename"
    fi
done
