import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from groq import Groq
from typing import List, Dict
import argparse

class BasicRAGSystem:
    def __init__(self, collection_name: str = "documents", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with ChromaDB and sentence transformers
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Set Groq API key
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def load_documents(self, file_paths: List[str]) -> None:
        """
        Load and process documents into the vector database
        """
        all_chunks = []
        
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            
            # Read document
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Create document object
            doc = Document(page_content=content, metadata={"source": file_path})
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Process chunks in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            self._add_chunks_to_db(batch)
            print(f"Processed batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
    
    def _add_chunks_to_db(self, chunks: List[Document]) -> None:
        """
        Add chunks to ChromaDB with embeddings
        """
        # Extract text content
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Create unique IDs
        ids = [f"doc_{i}_{hash(text)}" for i, text in enumerate(texts)]
        
        # Prepare metadata
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return retrieved_docs
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate response using Groq with retrieved context
        """
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Create prompt
        prompt = f"""
You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, n_results: int = 3) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, n_results)
        
        # Generate response
        response = self.generate_response(question, retrieved_docs)
        
        return {
            'question': question,
            'answer': response,
            'retrieved_documents': retrieved_docs
        }
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection
        """
        count = self.collection.count()
        return {
            'collection_name': self.collection.name,
            'document_count': count
        }

def main():
    parser = argparse.ArgumentParser(description='Basic RAG System')
    parser.add_argument('--load', nargs='+', help='File paths to load into the system')
    parser.add_argument('--query', type=str, help='Query to ask the system')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = BasicRAGSystem()
    
    # Load documents
    if args.load:
        print("Loading documents...")
        rag_system.load_documents(args.load)
        print("Documents loaded successfully!")
    
    # Single query mode
    if args.query:
        result = rag_system.query(args.query)
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"\nRetrieved {len(result['retrieved_documents'])} documents")
    
    # Interactive mode
    if args.interactive or (not args.load and not args.query):
        print(f"\n=== RAG System Interactive Mode ===")
        info = rag_system.get_collection_info()
        print(f"Collection: {info['collection_name']}")
        print(f"Documents: {info['document_count']}")
        print("Type 'quit' to exit\n")
        
        while True:
            question = input("Ask a question: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
            
            if question:
                result = rag_system.query(question)
                print(f"\nAnswer: {result['answer']}")
                print("-" * 50)

if __name__ == "__main__":
    main()