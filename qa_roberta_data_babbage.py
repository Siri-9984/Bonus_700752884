# Step 1: Install required libraries (uncomment if not already installed)
# !pip install transformers

from transformers import pipeline

# Step 2: Load the custom pretrained QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Step 3: Define context and question
context = """
Charles Babbage was an English mathematician, philosopher, inventor and mechanical engineer 
who originated the concept of a digital programmable computer. 
He is considered the father of the computer.
"""

question = "Who is considered the father of the computer?"

# Step 4: Run the model
result = qa_pipeline(question=question, context=context)

# Step 5: Print and validate result
print("Answer:", result['answer'])
print("Score:", result['score'])
print("Start index:", result['start'])
print("End index:", result['end'])

# Optional: Check correctness
assert result['answer'] == 'Charles Babbage', "Answer mismatch!"
assert result['score'] > 0.70, "Score is too low!"
print("\n✅ Output meets expected criteria.")
