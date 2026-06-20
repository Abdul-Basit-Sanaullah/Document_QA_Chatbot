from pypdf import PdfReader
from transformers import pipeline

# Read PDF
reader = PdfReader("documents/Final_Report.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()

    if extracted:
        text += extracted

print("PDF Loaded Successfully!")
print("Characters:", len(text))

# Load QA Model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# Ask Questions
while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    result = qa_pipeline(
        {
            "question": question,
            "context": text
        }
    )

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:")
    print(round(result["score"] * 100, 2), "%")