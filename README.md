# 🤖 Basic RAG System with ChromaDB and Groq
A lightweight Retrieval-Augmented Generation (RAG) application using ChromaDB for vector storage and Groq's Mixtral model for intelligent question answering.
Basic RAG System with ChromaDB and Groq
🎯 Project Overview
This project implements a fundamental RAG (Retrieval-Augmented Generation) system that combines document retrieval with language generation to provide accurate, context-aware responses. The system uses ChromaDB for efficient vector storage and Groq's fast inference capabilities for response generation.
Key Features

Vector Database Integration: ChromaDB for persistent document storage and similarity search
Semantic Embeddings: all-MiniLM-L6-v2 model for consistent document embeddings
Fast Inference: Groq's Mixtral-8x7b model for rapid response generation
Flexible Document Processing: Automatic text chunking with configurable overlap
Interactive Interface: Command-line interface with multiple operation modes
Extensible Architecture: Easy to modify and extend for different use cases

# 🏗️  System Architecture
┌─────────────────┐     ┌──────────────────┐    ┌─────────────────┐
│   Input Query   │───▶│  Vector Search   │───▶│   Response      │
│                 │     │   (ChromaDB)     │    │  Generation     │
└─────────────────┘     └──────────────────┘    │   (Groq)        │
                                                └─────────────────┘
                                 ▲                      
                                 │                      
                       ┌──────────────────┐             
┌─────────────────┐     │   Document       │             
│   Documents     │───▶│   Processing     │             
│   (.txt files)  │     │  & Embedding     │             
└─────────────────┘     └──────────────────┘
# 🚀 Quick Start
Prerequisites

Python 3.8+
Groq API key (free at console.groq.com)
2GB+ RAM for embedding model

Installation

Clone the repository
git clone <your-repo-url>
cd rag-groq-app

Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Configure API key
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here

Basic Usage

Load documents into the system
bashpython rag_system.py --load sample_docs/sample_doc1.txt sample_docs/sample_doc2.txt

Ask a single question
bashpython rag_system.py --query "What is machine learning?"

Interactive mode
bashpython rag_system.py --interactive

# 📁 Project Structure
rag-groq-app/
├── rag_system.py           # Main RAG application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
├── README.md             # This documentation
├── LICENSE               # Open source license
├── sample_docs/          # Sample documents for testing
│   ├── sample_doc1.txt   # AI and ML concepts
│   ├── sample_doc2.txt   # Python programming
│   └── sample_doc3.txt   # Vector databases
├── chroma_db/            # ChromaDB storage (auto-created)
└── tests/                # Unit tests
    └── test.py
# 🧪 Testing
Run the test suite to ensure everything works correctly:
bashpython -m pytest tests/ -v
Sample Queries to Test

"What is machine learning?"
"How does Python differ from other programming languages?"
"Explain vector databases and their use cases"
"What are the benefits of RAG systems?"

🔄 API Reference
BasicRAGSystem Class
__init__(collection_name, model_name)
Initialize the RAG system with specified parameters.
load_documents(file_paths)
Load and process documents into the vector database.
query(question, n_results=3)
Complete RAG pipeline: retrieve relevant documents and generate response.
retrieve_documents(query, n_results=3)
Retrieve relevant documents for a given query.
generate_response(query, retrieved_docs)
Generate response using Groq with retrieved context.
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
Development Setup

Fork the repository
Create a feature branch: git checkout -b feature-name
Make your changes and add tests
Run tests: python -m pytest
Submit a pull request

📈 Roadmap

 Support for PDF and DOCX documents
 Web interface using Streamlit
 Advanced chunking strategies
 Multi-language support
 Evaluation metrics and benchmarking
 Docker containerization

🐛 Troubleshooting
Common Issues
Import Error: setuptools.build_meta
bashpip install --upgrade pip setuptools wheel
Invalid API Key Error

Verify your Groq API key in the .env file
Ensure no extra spaces or quotes around the key

Memory Issues

Reduce batch size in document processing
Use smaller embedding models for limited memory systems

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
# 🙏 Acknowledgments

ChromaDB for the vector database
Groq for fast LLM inference
Sentence Transformers for embedding models
LangChain for document processing utilities
