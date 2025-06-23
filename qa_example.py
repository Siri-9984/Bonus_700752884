from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define your context
context = """
Maria had always dreamed of visiting Japan, so she finally booked a trip during the cherry blossom season.
She spent her days exploring Kyoto’s temples and trying local street food.
"""

# Define questions
questions = [
    "What motivated Maria to visit Japan?",
    "What activities did Maria do in Kyoto?"
]

# Ask questions and print answers with scores
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']:.2f}")
    print("-" * 50)
