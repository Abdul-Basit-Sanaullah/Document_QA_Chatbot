import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

st.title("📚 Document Question Answering Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    st.success("PDF Loaded Successfully!")

    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Get Answer"):

        result = qa_pipeline(
            {
                "question": question,
                "context": text
            }
        )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Confidence")
        st.write(f"{result['score']*100:.2f}%")