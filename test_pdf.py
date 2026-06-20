from pypdf import PdfReader

reader = PdfReader("documents/Final_Report.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

print(text[:1000])