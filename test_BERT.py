from transformers import pipeline

context = """
Natural Language Processing (NLP) is a field of Artificial Intelligence.
It allows computers to understand human language.
"""

qa_pipeline = pipeline(
    task="question-answering",
    model="deepset/roberta-base-squad2"
)

result = qa_pipeline(
    question="What is NLP?",
    context=context
)

print(result)