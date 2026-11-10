import os
import chromadb
from chromadb.utils import embedding_functions

def run_rag_simulation():
    print("Initializing Vector Database (ChromaDB) with SentenceTransformers...")
    # Connect to local persistent storage
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Explicitly use sentence-transformers to avoid default ONNX download timeouts
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.get_or_create_collection(
        name="enterprise_kb",
        embedding_function=ef
    )

    # Sample Knowledge Base
    documents = [
        "Enterprise employees are permitted up to 5 days of remote work per month with manager approval.",
        "GitHub Actions provides free cloud CI/CD runners for automated pipeline execution on GitHub repositories.",
        "RAG (Retrieval-Augmented Generation) combines semantic vector search with LLM generation to eliminate hallucinations."
    ]
    ids = ["doc1", "doc2", "doc3"]

    print("Adding documents to Vector Store...")
    collection.upsert(
        documents=documents,
        ids=ids
    )

    # Simulate a user query
    query = "How many remote work days are allowed?"
    print(f"Query: '{query}'")

    # Perform Vector Similarity Search
    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    retrieved_doc = results['documents'][0][0]
    print(f"Top Vector Match Retrieved: {retrieved_doc}")
    print(">>> RAG Pipeline Test Passed Successfully!")

if __name__ == "__main__":
    run_rag_simulation()