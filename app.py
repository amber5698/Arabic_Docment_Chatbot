import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb

# Load the Jais model and tokenizer (ensure this is the correct model ID)
model_name = "InceptionAI/Jais"  # Replace with an actual model ID from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Chroma DB for document retrieval
client = chromadb.Client()
collection = client.create_collection("arabic_books")

# Function to add documents to the collection (example documents)
def add_documents(documents):
    for doc in documents:
        collection.add(doc)

# Example: Add documents (list of dictionaries with 'text' and 'metadata')
documents = [
    {"text": "كتاب عن الفلسفة", "metadata": {"title": "فلسفة", "author": "أحمد"}},
    {"text": "رواية تاريخية", "metadata": {"title": "تاريخ", "author": "علي"}},
]
add_documents(documents)

# Function to generate response using RAG
def generate_response(user_input):
    # Retrieve relevant documents based on user input
    results = collection.query(user_input, n_results=3)  # Retrieve top 3 relevant documents

    # Prepare context for generation
    context = "\n".join([result['text'] for result in results['documents']])

    # Tokenize input with context
    inputs = tokenizer.encode(f"Use the given context to answer the question. Context: {context}, Question: {user_input}", return_tensors='pt')

    # Generate response
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI setup
st.title("Arabic Book Chatbot")
st.write("Welcome to the Arabic Book Chatbot! Ask me anything about our collection.")

user_input = st.text_input("Your Question:")

if st.button("Ask"):
    if user_input:
        response = generate_response(user_input)
        st.write("**Response:**", response)
    else:
        st.write("Please enter a question.")
