import pytesseract
from pdf2image import convert_from_path
import os
import time

# ====== 请修改这里 ======
pdf_path = "Art as Experience.pdf"
output_txt = "Art_as_Experience_OCR.txt"

# 从第42页开始到最后一页（设为 None 表示到末尾）
start_page = 8
page_count = None  # None 表示到文件末尾
# ========================

# 如果 Tesseract 不在系统 PATH 中，请取消下面一行的注释并填入正确路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 先获取总页数，以便计算需要处理多少页
from pdf2image import pdfinfo_from_path
info = pdfinfo_from_path(pdf_path)
total_pages = info.get("Pages", 0)

# 确定实际要处理的页码范围
if page_count is None:
    last_page = total_pages
else:
    last_page = min(start_page + page_count - 1, total_pages)

print(f"📖 共 {total_pages} 页，从第 {start_page} 页开始转换到第 {last_page} 页...")
start_time = time.time()

# 批量转换
images = convert_from_path(
    pdf_path, 
    dpi=300, 
    first_page=start_page, 
    last_page=last_page
)

print(f"✅ 共 {len(images)} 页，开始 OCR 识别...")
print("⏳ 这个过程可能需要较长时间，请耐心等待...\n")

text_content = ""
for i, img in enumerate(images):
    actual_page_num = start_page + i
    print(f"  正在识别第 {actual_page_num}/{last_page} 页...")
    text = pytesseract.image_to_string(img, lang="eng")
    #text_content += f"\n\n========== 第 {actual_page_num} 页 ==========\n\n"
    text_content += text

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text_content)

elapsed = time.time() - start_time
print(f"\n✅ 完成！共 {len(images)} 页，耗时 {elapsed:.1f} 秒")
print(f"📁 保存为：{output_txt}")