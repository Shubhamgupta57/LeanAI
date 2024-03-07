print("initializing...")
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import os

from llama_index.llms.replicate import Replicate

# import pyttsx3
# engine = pyttsx3.init()

os.environ["REPLICATE_API_TOKEN"] = "r8_9diCKWU4vXAaXEfLn9NN8Xk9MLzvOKk3EqgSz"

# # Local model
# Settings.llm = Ollama(model="zephyr", request_timeout=300)
# embed_model = 'local'
# Settings.embed_model = embed_model

# Replicate
llm = Replicate(
    model="mistralai/mistral-7b-instruct-v0.2",
    max_tokens=100000,
    temperature=0.6
)
Settings.llm = llm
embed_model = 'local'
Settings.embed_model = embed_model

# Logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # Load documents
# documents = SimpleDirectoryReader("data").load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_collection("lean_data")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build an index

# # Save to disk
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )

# Load from disk
index = VectorStoreIndex.from_vector_store(
    vector_store, embed_model=embed_model
)

# print("done")




# Initiate a query engine
query_engine = index.as_query_engine(streaming=True)

# # Make a query to the index
# response = query_engine.query('How did the author start "Y Combinator"?')
# response.print_response_stream()

while True:
    # engine.say("Whats on your mind?")
    # engine.runAndWait()
    query = input("Whats on your mind?\n")
    if query == "":
        continue

    response = query_engine.query(query)
    # engine.say(response.response_txt)
    # engine.runAndWait()
    response.print_response_stream()

    print("\n")


# Ollama example
# from llama_index.llms.ollama import Ollama
# llm = Ollama(model="llama2", request_timeout=30.0)
# response = llm.stream_complete("How to check if number is even in python")
# for r in response:
#     print(r.delta, end="")