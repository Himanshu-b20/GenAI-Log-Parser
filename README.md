# 🔍 Log Parser with RAG Architecture using Groq & ChromaDB
This project implements an AI-powered log parsing and question-answering system using a Retrieval-Augmented Generation (RAG) architecture. It allows users to upload raw log files, parse and vectorize their contents, store them in ChromaDB, and ask natural language questions about the logs. The system retrieves relevant log chunks and forwards them to a Groq-hosted LLM for contextual and accurate responses.

## 🚀 Features
- 📂 Upload & Process Logs: Supports uploading multiple .txt log files via a user interface.
- 🧠 RAG-based Architecture: Combines retrieval and generation for better log understanding.
- 🧾 Text Parsing & Chunking: Log files are parsed and split into meaningful text chunks.
- 🔍 Embedding & Vector Storage: Chunks are embedded and stored in ChromaDB for fast similarity-based retrieval.
- 💬 Natural Language Querying: Users can ask questions about the uploaded logs.
- ⚡ LLM Integration via Groq: Retrieves relevant context from ChromaDB and sends it to a fast, Groq-hosted LLM for final response generation.

## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **ChromaDB**
- **Groq + LLM (model=llama-3.3-70b-versatile)**
- **Sentence Embeddings (using sentence-transformers/all-MiniLM-L6-v2)**

## 📦 Architecture Overview
┌────────────────────────┐
│   1. Upload Log Files  │◄─────────────┐
└─────────┬──────────────┘              │
          │                             │
          ▼                             │
┌────────────────────────┐              │
│ 2. Parse & Chunk Logs  │              │
└─────────┬──────────────┘              │
          │                             │
          ▼                             │
┌────────────────────────────────────┐  │
│ 3. Generate Embeddings            │  │
│    (using embedding model)        │  │
└─────────┬──────────────────────────┘  │
          │                             │
          ▼                             │
┌──────────────────────────────┐       │
│ 4. Store Vectors in ChromaDB │       │
└─────────┬────────────────────┘       │
          │                             │
          ▼                             │
┌─────────────────────────────┐        │
│ 5. User Inputs Query        │        │
└─────────┬───────────────────┘        │
          │                             │
          ▼                             │
┌────────────────────────────────────────────┐
│ 6. Retrieve Relevant Chunks from ChromaDB  │
└─────────┬───────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│ 7. Send Query + Context to Groq-hosted LLM │
└─────────┬───────────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ 8. Return Answer to User    │
└─────────────────────────────┘


## 🧪 Setup & Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Himanshu-b20/GenAI-Log-Parser.git
   ```
2. **Install dependencies:**:   
   ```commandline
    pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**:   
   ```commandline
    streamlit run /main.py
   ```
