#!/bin/dash

# 获取排序后的文件列表
set -- $(
    for file in spatial*.txt; do
        [ -f "$file" ] || continue
        num=$(echo "$file" | sed -nE 's/^spatial([0-9]{3})\.txt$/\1/p')
        [ -n "$num" ] || continue
        printf "%d %s\n" "$num" "$file"
    done | sort -n | cut -d' ' -f2
)

total_files=$#
expected=296

# 验证文件总数
if [ "$total_files" -ne "$expected" ]; then
    echo "错误：找到 $total_files 个文件，需要 $expected 个文件"
    exit 1
fi

# 执行重命名
counter=1
for old in "$@"; do
    new=$(printf "spatial%03d.txt" "$counter")
    
    if [ "$old" = "$new" ]; then
        counter=$((counter + 1))
        continue
    fi

    if [ -e "$new" ]; then
        echo "错误：目标文件 $new 已存在"
        exit 1
    fi

    mv -- "$old" "$new"
    echo "已重命名: $old -> $new"
    counter=$((counter + 1))
done

echo "操作完成！共处理 $total_files 个文件"
