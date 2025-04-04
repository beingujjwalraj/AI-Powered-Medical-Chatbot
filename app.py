from flask import Flask, render_template, request
from langchain_community.llms import HuggingFaceHub
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re

app = Flask(__name__)

load_dotenv()

# Configuration
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

# Initialize embeddings
from src.helper import download_hugging_face_embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone setup
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Reduced context chunks
)

# Simplified prompt
system_prompt = """You are a medical expert. Provide clear, concise answers using this context:
{context}

Answer in complete sentences using everyday language. Structure your response:
1. Brief definition
2. Main causes
3. Affected groups
4. Common treatments
Do not use markdown or special formatting."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Model configuration
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={
        "temperature": 0.4,  # Slightly higher for flexibility
        "max_new_tokens": 300,
        "top_p": 0.85,
        "repetition_penalty": 1.1
    },
)

# Conservative response cleaner
def clean_response(text):
    # Basic cleaning only
    clean_patterns = [
        r'\(Source:.*?\)',
        r'\bPage\s\d+',
        r'\bGALE ENCYCLOPEDIA.*',
        r'\s+'
    ]
    
    for pattern in clean_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Fix common punctuation issues
    text = text.replace('..', '.').replace('  ', ' ')
    return text.strip()

# Chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        query = request.form["msg"].strip()
        if not query:
            return "Please enter a valid medical question"
            
        result = rag_chain.invoke({"input": query})
        return clean_response(result["answer"])
    
    except Exception as e:
        return "I'm having trouble answering that right now. Please try again later."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)