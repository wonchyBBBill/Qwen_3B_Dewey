import os
import re

# 3 input files
input_files = [
    "Art_as_Experience_Cleaned.txt",
    "Democracy_and_Education.txt",
    "Reconstruction_in_Philosophy.txt"
]
output_folder = "Dewey_Chapters"

os.makedirs(output_folder, exist_ok=True)

chapter_pattern = r'Chapter[^.?!\n]*$|CHAPTER[^.?!\n]*$'

for input_file in input_files:
    
    print(f"\n处理文件: {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    matches = list(re.finditer(chapter_pattern, content, re.MULTILINE))
    
    if not matches:
        print(f"  未找到章节标题，跳过: {input_file}")
        continue
    
    print(f"  找到 {len(matches)} 个章节")
    
    # 使用输入文件名（去掉扩展名）作为前缀，避免不同文件章节重名
    base_name = os.path.splitext(os.path.basename(input_file))[0][:5]
    
    # 分割并保存每个章节
    for i, match in enumerate(matches):
        # 确定章节内容的起始位置
        start_pos = match.start()
        
        # 确定结束位置（下一个 Chapter 的开始，或文件末尾）
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        # 提取章节内容
        chapter_content = content[start_pos:end_pos].strip()
        
        # 文件名（加上输入文件名前缀）
        filename = f"{base_name}_{i+1:02d}.txt"
        filepath = os.path.join(output_folder, filename)
        
        # 保存文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chapter_content)
        
        print(f"  已保存: {filename}")

print(f"\n完成！所有章节已保存到文件夹: {output_folder}")