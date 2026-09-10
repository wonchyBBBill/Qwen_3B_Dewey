import re

def fix_newlines(text):
    # 第一步：删除断词连字符（行尾的 "-" 加换行符）
    text = re.sub(r'-\n', '', text)
    
    # 第二步：将单个换行符替换为空格（保留双换行符）
    # 先保护双换行符，再替换单换行符，最后恢复
    text = re.sub(r'\n\n', '\x00', text)  # 用特殊字符占位
    text = re.sub(r'\n', ' ', text)       # 单换行变空格
    text = re.sub(r'\x00', '\n\n', text)  # 恢复双换行
    
    return text

with open("Art_as_Experience_Annotation_Header_Cleaned.txt", 'r', encoding='utf-8') as f:
    text = f.read()

cleaned_text = fix_newlines(text)
    
with open("Art_as_Experience_Cleaned.txt", 'w', encoding='utf-8') as f:
    f.write(cleaned_text)