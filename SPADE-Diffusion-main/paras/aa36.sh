#!/bin/sh

# 遍历 spatial001.txt 到 spatial300.txt
for i in $(seq -w 1 300); do
    file_name="spatial${i}.txt"
    
    # 检查文件是否存在
    if [ -f "$file_name" ]; then
        # 使用 awk 处理文件
        awk '
        NR==1 {
            line1 = $0;  # 保存第一行
            if (index($0, ",") > 0) {
                add_line = "Final split ratio: 0.002,0.998";
            } else if (index($0, ";") > 0) {
                add_line = "Final split ratio: 0.002;0.998";
            } else {
                add_line = "";  # 如果没有逗号或分号，则不添加
            }
        }
        NR==2 {
            print line1;  # 打印第一行
            print add_line;  # 打印添加的行
            print $0;  # 打印原来的第二行
            next;  # 跳过默认的打印动作
        }
        { print }  # 对于其他行（如果有的话），直接打印（不过这里每个文件只有两行）
        ' "$file_name" > "${file_name}.tmp" && mv "${file_name}.tmp" "$file_name"
    else
        echo "Warning: $file_name does not exist."
    fi
done
