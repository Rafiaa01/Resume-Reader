from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = \
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_path = r"C:\Users\PMYLS\Downloads\archive\Resumes PDF\data science resumes/Image_8.pdf"

pages = convert_from_path(
    pdf_path,
    poppler_path=r"C:\poppler\Library\bin"
)

text = ""

for page in pages:
    text += pytesseract.image_to_string(page)

print(text[:3000])