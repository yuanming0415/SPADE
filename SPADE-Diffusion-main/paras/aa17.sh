#!/bin/sh

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    # 构造文件名
    filename="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$filename" ]; then
        # 获取文件的最后一行
        last_line=$(tail -n 1 "$filename")
        
        # 检查最后一行是否是 ***
        if [ "$last_line" = "***" ]; then
            # 删除最后一行
            sed -i '$d' "$filename"
            echo "已处理文件: $filename"
        fi
    else
        echo "文件不存在: $filename"
    fi
done
