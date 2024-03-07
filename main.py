print("initializing...")
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from llama_index.llms.ollama import Ollama

# Local model
Settings.llm = Ollama(model="zephyr", request_timeout=300)
embed_model = 'local'
Settings.embed_model = embed_model


# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_collection("lean_data")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load from disk
index = VectorStoreIndex.from_vector_store(
    vector_store, embed_model=embed_model
)


# Initiate a query engine
query_engine = index.as_query_engine(streaming=True)

while True:
    query = input("Whats on your mind?\n")
    if query == "":
        continue

    response = query_engine.query(query)
    response.print_response_stream()

    print("\n")