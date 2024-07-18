#!/usr/bin/env python3

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama 
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
import numpy as np
import time
#import logging
#import sys
import subprocess
import re

#enable debug logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def qdrant_test_search(client):
    return(client.search(
        collection_name="llamaindex_db",
        query_vector = np.random.rand(768).tolist(),
        limit=1
    ))

def query_from_index(index, query):
    print("Querying model")
    query_engine = index.as_query_engine()
    return query_engine.query(query)

def get_qdrant_container_ports():
    try:
        #filter docker ps to only show port info of qdrant-based containers
        result = subprocess.run(
            ["sudo", "docker", "ps", "--format", "{{.Ports}}", "--filter", "ancestor=qdrant/qdrant"],
            capture_output=True,
            text=True,
            check=True
        )
        
        #parse docker ps output to get container host ports and return a list of those unique ports
        return list(set(re.findall(r'(?<=:)\d+(?=->)', result.stdout)))
    except subprocess.CalledProcessError: 
        return "Error: docker ps failed" 

#-----------------------------
#ingest data into qdrant containers
#-----------------------------

qdrant_container_ports = get_qdrant_container_ports()
#check if there is at least one nonempty entry in list of ports
if not qdrant_container_ports[0]: 
    raise RuntimeError("No qdrant containers are currently running")

#load data
documents = SimpleDirectoryReader("data").load_data()

#set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

#set up qdrant client
client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="llamaindex_db")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

start_time = time.perf_counter()
print(f"Ingesting index into qdrant database using host port . Starting time: {start_time}")

#create index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

end_time = time.perf_counter()
print(f"Ingestion complete. Ending time: {end_time}")
print(f"Total ingestion time: {(end_time - start_time):.6f} seconds")

client = QdrantClient(host="localhost", port=6334)
vector_store = QdrantVectorStore(client=client, collection_name="llamaindex_db")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        show_progress=True
)

qdrant_test_search(client)

#set up ollama container connection
Settings.llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=600.0)

#-----------------------------
#query model
#-----------------------------
print(query_from_index(index, "What is the article about?")) #note that this currently creates a query engine based off the most recently created index