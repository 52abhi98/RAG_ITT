import streamlit as st
import numpy as np
import torch
import json
import os
import fitz  # PyMuPDF
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import InferenceClient
import torch.nn.functional as F

# Load GTE model and tokenizer
model_path = 'Alibaba-NLP/gte-large-en-v1.5'
# model_path = 'lw2134/policy_gte_large_2plus'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Function to get GTE embeddings
def get_gte_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]  # Using the CLS token
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    return embeddings.squeeze().numpy()

# Function to split text (from PDF)
def split_text(text, max_length=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i + max_length]))
    return chunks

# Querying function using GTE
def query_embedding(query, pdf_embeddings, pdf_chunks):
    query_embedding = get_gte_embedding(query)
    similarities = np.dot(pdf_embeddings, query_embedding) / (np.linalg.norm(pdf_embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_n_indices = similarities.argsort()[-4:][::-1]  # Top 4 matches
    return [(pdf_chunks[i], similarities[i]) for i in top_n_indices]

# Set up the Mistral model repo id
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Create the inference client
llm_client = InferenceClient(
    model=repo_id,
    token="", 
    timeout=120,
)

def call_llm(inference_client: InferenceClient, context: str, query: str):
    QA_generation_prompt = f"""
You are a knowledgeable assistant. Based on the context provided, answer the following query. 
If the Context does not contain the answer to the Query, respond with "Information is not provided in the context"

Instructions:
-Analyze the Query 
-Find the Answer for the given Query in given Context
-If the Context does not contain the Answer to the Query, respond with "Information is not provided in the Context" and do not respond anything else then that

Context: {context}
Query: {query}

Answer:
"""

    response = inference_client.post(
        json={
            "inputs": QA_generation_prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Streamlit interface
st.title("PDF Query System")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a local directory
    save_path = os.path.join("uploaded_files", uploaded_file.name)  # Specify the directory to save
    os.makedirs("uploaded_files", exist_ok=True)  # Create directory if it doesn't exist
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the saved PDF
    pdf_text = extract_text_from_pdf(save_path)
    pdf_chunks = split_text(pdf_text)
    pdf_embeddings = np.array([get_gte_embedding(chunk) for chunk in pdf_chunks])

    # User query input
    query_text = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        if query_text:
            relevant_context = query_embedding(query_text, pdf_embeddings, pdf_chunks)
            # Format the context for the Mistral prompt
            context_text = "\n".join(f"{chunk}" for chunk, score in relevant_context)
            # Generate the final response using Mistral
            response = call_llm(llm_client, context_text, query_text)

            # Display the generated response
            st.text_area("Response:", response, height=300)
        else:
            st.warning("Please enter a query.")
