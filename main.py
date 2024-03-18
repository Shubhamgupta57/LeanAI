print("Initializing...")
import logging
import sys
import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
# Set up the local model and embedding model settings
Settings.llm = Ollama(model="zephyr", request_timeout=300)
embed_model = 'local'
Settings.embed_model = embed_model
try:
    # Initialize the Chroma database client with a specified path
    db_path = "./chroma_db"
    db_client = chromadb.PersistentClient(path=db_path)
    # Create or retrieve a collection from the database
    data_collection = db_client.get_collection("lean_data")
    # Assign the Chroma vector store to the storage context
    vector_store = ChromaVectorStore(chroma_collection=data_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Initialize the index and query engine for processing queries
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    query_engine = index.as_query_engine(streaming=True)
    # Continuously process user queries
    while True:
        try:
            user_query = input("What's on your mind?\n").strip()
            if not user_query:
                continue  # Skip empty queries
            # Obtain the response from the query engine
            response = query_engine.query(user_query)
            top_response = response.get_response()  # Get the most relevant response
            confidence = top_response.confidence  # Extract the confidence level
            # Check the confidence level of the top response
            if confidence < 0.8:  # Adjusted to 80% confidence threshold
                print("I'm not confident enough to provide an accurate response. Please contact support for further assistance.")
            else:
                response.print_response_stream()  # Display the response if confidence is high
            print("\n")  # New line for clear separation between interactions
        except Exception as e:
            print(f"An error occurred while processing your query: {e}")
            logging.error("Error processing user query", exc_info=True)
except Exception as e:
    print(f"Failed to initialize the chatbot system: {e}")
    logging.error("Initialization failure", exc_info=True)