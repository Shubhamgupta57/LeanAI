print("training...")
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Delete existing chromadb storage
import shutil
import sys
import os

# Set the path for the directory you want to delete
directory_path = "./chroma_db"

if os.path.exists(directory_path):
    try:
        shutil.rmtree(directory_path)
        print("Chromadb has been deleted successfully")
    except OSError as e:
        print(e)
        print("Chromadb cannot be removed")
        sys.exit()

# Local model
embed_model = 'local'
Settings.embed_model = embed_model

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("lean_data")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build an index

# Save to disk
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

print("Training done!!")