#!/bin/bash

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300)
do
    # 构造文件名
    filename="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$filename" ]; then
        # 使用 sed 去掉每行末尾的 1 个空格
        sed -i 's/ $//' "$filename"
        echo "已处理文件: $filename"
    else
        echo "文件不存在: $filename"
    fi
done
