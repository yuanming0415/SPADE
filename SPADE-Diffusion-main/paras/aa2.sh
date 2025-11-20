#!/bin/bash

# 遍历从1到300的数字
for i in $(seq 1 300); do
    # 格式化数字为三位数，例如1变成001
    num=$(printf "%03d" $i)
    # 构造文件名
    filename="spatial${num}.txt"
    
    # 检查文件是否存在
    if [ -f "$filename" ]; then
        # 使用sed命令在第一行和第二行的末尾添加空格
        # -i 表示直接修改文件，''在macOS中需要，Linux下可省略
        sed -i '1,2s/$/ /' "$filename"
        echo "已处理文件: $filename"
    else
        echo "警告: 文件 $filename 不存在，跳过。"
    fi
done
