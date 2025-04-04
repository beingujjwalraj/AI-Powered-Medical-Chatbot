

# Medical Chatbot 🏥

An AI-powered medical chatbot using RAG (Retrieval-Augmented Generation) architecture to provide accurate health information retrieval from medical documents.

https://github.com/user-attachments/assets/cfe7d5b2-beff-41f2-a7b8-682768b26f5a
## Features
- 🩺 Symptom analysis and condition information  
- 💊 Medication and treatment options  
- 📄 PDF document processing (medical journals/encyclopedias)  
- 🔍 Context-aware responses using Pinecone vector search  
- 🧠 Mistral-7B LLM for natural language understanding  
- 🔒 Local deployment with Flask backend  

## Technologies Used
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.11-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-green)
![Mistral-7B](https://img.shields.io/badge/LLM-Mistral_7B-informational)

## Configuration
| Parameter               | Description                                |
|-------------------------|--------------------------------------------|
| `PINECONE_API_KEY`      | Pinecone vector database API key           |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face Hub access token           |
| `PINECONE_INDEX_NAME`   | Pinecone index name (default: medicalbot)  |
| `DATA_DIR`              | Path to medical PDFs (default: data/)      |

## Tech Stack

### Backend
- **Flask** - Web framework  
- **LangChain** - LLM orchestration  
- **Pinecone** - Vector database  
- **Sentence Transformers** - Text embeddings  

### AI/ML
- **Mistral-7B-Instruct** - Large Language Model  
- **RAG Architecture** - Retrieval-Augmented Generation  
- **Hugging Face Hub** - Model hosting  

### Data Processing
- **PyPDF** - PDF text extraction  
- **Recursive Text Splitting** - Document chunking  

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the project  
2. Create your feature branch  
   ```bash
   git checkout -b feature/AmazingFeature



