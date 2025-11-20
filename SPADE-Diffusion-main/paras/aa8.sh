#!/bin/bash

# 使用 seq 生成 1 到 300 的数字序列
for i in $(seq 1 300); do
    # 构造文件名
    filename=$(printf "spatial%03d.txt" "$i")
    
    # 检查文件是否存在
    if [ -f "$filename" ]; then
        # 获取文件的总行数
        total_lines=$(wc -l < "$filename")
        
        # 如果文件行数大于1，才进行处理
        if [ "$total_lines" -gt 1 ]; then
            # 获取倒数第二行
            penultimate_line=$(tail -n 2 "$filename" | head -n 1)
            
            # 获取最后一行
            last_line=$(tail -n 1 "$filename")
            
            # 删除最后一行
            sed -i '$d' "$filename"
            
            # 将最后一行追加到倒数第二行的末尾（不换行）
            sed -i "\$s/$/${last_line}/" "$filename"
            
            echo "已处理文件: $filename"
        else
            echo "文件行数不足，跳过: $filename"
        fi
    else
        echo "文件不存在: $filename"
    fi
done
