from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\PMYLS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

pdf_path = r"C:\Users\PMYLS\Downloads\archive\Resumes PDF\Accountant\0.pdf"

pages = convert_from_path(
    pdf_path,
    poppler_path=r"PASTE_POPPLER_BIN_PATH_HERE"
)

text = ""

for page in pages:
    text += pytesseract.image_to_string(page)

print(text[:3000])