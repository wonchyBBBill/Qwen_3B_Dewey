import os
import re

input_file = "Art_as_Experience_Cleaned.txt"
output_folder = "Art_as_Experience_Chapters"

os.makedirs(output_folder, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 按章节分割（匹配 "CHAPTER" 后跟罗马数字或阿拉伯数字）
# 匹配模式：CHAPTER + 空格 + (罗马数字或数字) + 换行
chapter_pattern = r'(CHAPTER\s+[IVXLCDM]+|[Cc]hapter\s+\d+)\s*\n'

# 找到所有章节标题的位置
matches = list(re.finditer(chapter_pattern, content))

if not matches:
    print("未找到章节标题，尝试其他模式...")
    # 备用模式
    chapter_pattern = r'CHAPTER\s+[IVXLCDM]+'
    matches = list(re.finditer(chapter_pattern, content))

print(f"找到 {len(matches)} 个章节")

# 分割并保存每个章节
for i, match in enumerate(matches):
    # 确定章节标题
    chapter_title = match.group(0).strip()
    # 清理标题作为文件名
    clean_title = re.sub(r'[\\/*?:"<>|]', "", chapter_title)
    clean_title = clean_title.replace(" ", "_")
    
    # 确定章节内容的起始位置
    start_pos = match.start()
    
    # 确定结束位置（下一个章节的开始，或文件末尾）
    if i + 1 < len(matches):
        end_pos = matches[i + 1].start()
    else:
        end_pos = len(content)
    
    # 提取章节内容
    chapter_content = content[start_pos:end_pos].strip()
    
    # 文件名
    filename = f"{i+1:02d}_{clean_title}.txt"
    filepath = os.path.join(output_folder, filename)
    
    # 保存文件
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(chapter_content)
    
    print(f"已保存: {filename}")

print(f"\n完成！所有章节已保存到文件夹: {output_folder}")