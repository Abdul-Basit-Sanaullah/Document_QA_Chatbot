import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="📚"
)

st.title("📚 Document Question Answering Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    try:
        # Read PDF
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

        if len(text.strip()) == 0:
            st.error("No text could be extracted from this PDF.")
            st.stop()

        st.success("PDF Loaded Successfully!")

        # Load QA Model
        qa_pipeline = pipeline(
            task="question-answering",
            model="deepset/roberta-base-squad2"
        )

        question = st.text_input(
            "Ask a Question"
        )

        if st.button("Get Answer"):

            if question.strip() == "":
                st.warning("Please enter a question.")
            else:

                # Limit context size for better performance
                context = text[:3000]

                result = qa_pipeline(
                    question=question,
                    context=context
                )

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Confidence")
                st.write(f"{result['score'] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error: {str(e)}")
