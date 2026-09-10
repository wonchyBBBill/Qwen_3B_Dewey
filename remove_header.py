import re

def remove_headers_by_title(text, title):
    """
    删除页眉行：包含标题且不等于标题本身的行
    同时删除紧随其后的空行
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # 如果这一行包含标题，但不是纯粹的标题本身 → 页眉，删除
        if title in line_stripped and line_stripped != title:
            print(f"🗑️ 删除页眉: {line_stripped}")
            i += 1
            # 跳过后面紧跟的空行
            if i < len(lines) and lines[i].strip() == '':
                print(f"   同时删除后面的空行")
                i += 1
            continue
        
        # 保留
        cleaned_lines.append(line)
        i += 1
    
    return '\n'.join(cleaned_lines)


# ========== 使用 ==========

chapter_titles = [
    "THE LIVE CREATURE",
    "ETHERIAL THINGS",
    "HAVING AN EXPERIENCE",
    "THE ACT OF EXPRESSION",
    "THE EXPRESSIVE OBJECT",
    "SUBSTANCE AND FORM",
    "THE NATURAL HISTORY OF FORM",
    "THE ORGANIZATION OF ENERGIES",
    "THE COMMON SUBSTANCE OF THE ARTS",
    "THE VARIED SUBSTANCE OF THE ARTS",
    "THE HUMAN CONTRIBUTION",
    "THE CHALLENGE TO PHILOSOPHY",
    "CRITICISM AND PERCEPTION",
    "ART AND CIVILIZATION",
    "ART AS EXPERIENCE"
]

with open("Art_as_Experience_Annotation_Cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

for title in chapter_titles:
    text = remove_headers_by_title(text, title)

text = re.sub(r'\n{3,}', '\n\n', text)

with open("Art_as_Experience_Annotation_Header_Cleaned.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("✅ 所有页眉清理完成！")