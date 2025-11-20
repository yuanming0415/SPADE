import os
import re

# 匹配文件名模式
pattern = re.compile(r'spatial(\d{3})\.txt')

# 获取所有符合条件的文件并提取数字
files = []
for filename in os.listdir('.'):
    match = pattern.fullmatch(filename)
    if match:
        num = int(match.group(1))
        files.append((num, filename))

# 按原始数字排序
files.sort()

# 检查文件数量
if len(files) != 296:
    print(f"错误：找到 {len(files)} 个文件，但需要296个文件。")
    exit()

# 重命名文件
for new_idx, (old_num, old_name) in enumerate(files, start=1):
    new_name = f"spatial{new_idx:03d}.txt"
    
    if old_name != new_name:
        # 检查是否有重名冲突
        if os.path.exists(new_name):
            print(f"错误：目标文件 {new_name} 已存在，请检查原始文件。")
            exit()
        
        os.rename(old_name, new_name)
        print(f"重命名：{old_name} -> {new_name}")

print("所有文件重命名完成！")
