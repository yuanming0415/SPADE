#!/bin/bash

for i in $(seq 1 300); do
    num=$(printf "%03d" $i)
    filename="spatial${num}.txt"
    
    if [ -f "$filename" ]; then
        # 1. 转换换行符为 Unix 格式（清理 \r）
        dos2unix "$filename" 2>/dev/null  # 静默执行
        
        # 2. 添加空格
        sed -i '1,2s/$/ /' "$filename"
        
        # 3. 如果需要保留 Windows 换行符，可以转换回来
        # unix2dos "$filename" 2>/dev/null
        
        echo "已处理文件: $filename"
    else
        echo "警告: 文件 $filename 不存在，跳过。"
    fi
done
