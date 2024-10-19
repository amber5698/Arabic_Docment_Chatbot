import logging
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set up logging
logger = logging.getLogger("streamlit_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Log the start of the application
logger.info("Streamlit app started")

# Load the model and tokenizer from Hugging Face
model_name = "FreedomIntelligence/AceGPT-7B-chat"  # Change this to the desired model
logger.info(f"Loading model: {model_name}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    st.error("Failed to load the model. Please check the logs for details.")

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
        logger.info(f"User asked: {user_input}")
        logger.info(f"Chatbot responded: {response}")
    else:
        st.write("Please enter a question.")

# Footer
st.markdown("---")
st.write("This chatbot uses AceGPT to provide answers based on our collection of Arabic books.")
