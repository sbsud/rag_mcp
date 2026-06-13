# Check what PyPDF2 actually extracted
# from PyPDF2 import PdfReader

# reader = PdfReader("data/Sudarshan_Shubakar_Resume_MongoDB_SeniorStaffEngineer.docx.pdf")
# for i, page in enumerate(reader.pages):
#     print(f"\n=== PAGE {i} ===")
#     print(page.extract_text())


    # Option 1: pdfminer — handles layout much better than PyPDF2
    # pip install pdfminer.six
path = "data/my_resume.pdf"
from pdfminer.high_level import extract_text
print(extract_text(str(path)))