import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer from Hugging Face
model_name = "FreedomIntelligence/AceGPT-7B-chat"  # Change this to the desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    """Generate a response from the model based on user input."""
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI setup
st.title("Arabic Book Collection Chatbot")
st.subheader("Ask me anything about our Arabic books!")

# User input
user_input = st.text_input("Your Question:")

if st.button("Ask"):
    if user_input:
        response = generate_response(user_input)
        st.write("**Chatbot:**", response)
    else:
        st.write("Please enter a question.")

# Footer
st.markdown("---")
st.write("This chatbot uses AceGPT to provide answers based on our collection of Arabic books.")
