import pdfplumber

pdf_path = r"C:\Users\PMYLS\Downloads\archive\Resumes PDF\Accountant\0.pdf"

with pdfplumber.open(pdf_path) as pdf:
    text = pdf.pages[0].extract_text()

print("RESULT:")
print(text)
