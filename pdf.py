import os
import pdfplumber
import pandas as pd

main_folder = r"C:\Users\PMYLS\Downloads\archive\Resumes PDF"

data = []

for root, folders, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)

            text = ""
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                category = os.path.basename(root)

                data.append({
                    "category": category,
                    "file_name": file,
                    "file_path": pdf_path,
                    "resume_text": text
                })

                print("Done:", file)

            except Exception as e:
                print("Error:", file, e)

df = pd.DataFrame(data)
df.to_csv("all_resumes_text.csv", index=False)

print("All resume data saved in all_resumes_text.csv")