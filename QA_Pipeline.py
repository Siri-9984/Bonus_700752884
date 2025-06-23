from transformers import pipeline

# Step 1: Load the QA pipeline
qa_pipeline = pipeline("question-answering")

# Step 2: Provide context and question
context = """
Charles Babbage was an English mathematician, philosopher, inventor and mechanical engineer 
who originated the concept of a programmable computer. Considered by some to be the "father of the computer", 
Babbage is credited with inventing the first mechanical computer.
"""

question = "Who is considered the father of the computer?"

# Step 3: Run the pipeline
result = qa_pipeline(question=question, context=context)

# Step 4: Print the result
print(result)
