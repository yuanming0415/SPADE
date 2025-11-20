#!/bin/dash

# 获取排序后的文件列表（修复八进制问题）
set -- $(
    for file in spatial*.txt; do
        [ -f "$file" ] || continue
        num=$(echo "$file" | sed -nE 's/^spatial([0-9]{3})\.txt$/\1/p')
        [ -n "$num" ] || continue
        
        # 移除前导零并转换为十进制
        dec_num=$(echo "$num" | sed 's/^0*//')
        [ -z "$dec_num" ] && dec_num=0  # 处理全零情况
        
        printf "%d %s\n" "$dec_num" "$file"
    done | sort -n | cut -d' ' -f2
)

total_files=$#
expected=261

# 验证文件总数
if [ "$total_files" -ne "$expected" ]; then
    echo "错误：找到 $total_files 个文件，需要 $expected 个文件"
    exit 1
fi

# 执行重命名（添加临时后缀避免冲突）
counter=1
for old in "$@"; do
    new=$(printf "spatial%03d.txt" "$counter")
    
    if [ "$old" = "$new" ]; then
        counter=$((counter + 1))
        continue
    fi

    # 添加临时后缀避免冲突
    tmp_name=".tmp_$new"
    mv -- "$old" "$tmp_name"
    mv -- "$tmp_name" "$new"
    
    echo "已重命名: $old -> $new"
    counter=$((counter + 1))
done

echo "操作完成！共处理 $total_files 个文件"
