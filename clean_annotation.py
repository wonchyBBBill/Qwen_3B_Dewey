import re

def clean_file(input_path, output_path):

    if input_path == output_path:
        raise ValueError("输入文件路径和输出文件路径不能相同，以避免覆盖原文件。")
    
    # 读取文件
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 删除以 * 开头、\n\n 结尾的注释
    cleaned_text = re.sub(r'\*.*?\n\n', '', text, flags=re.DOTALL)
    
    # 写入处理后的文本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"处理完成：{input_path} -> {output_path}")

# 使用方法
clean_file('Art_as_Experience_OCR.txt', 'Art_as_Experience_Annotation_Cleaned.txt')  # 保存为新文件