#!/usr/bin/env python3
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama 
from multiprocessing import Pool
import subprocess
import os

#parallel loading of documents
#parallel creation of indices for each document

#WIKI_DATA_DIR = "/data/jasonc/extracted_text"
WIKI_DATA_DIR = "/data/jasonc/6g_ingest_data"
DB_PORT = 6500

def create_index(file_path):
    print(f"Creating index for file {file_path}")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", embed_batch_size=256)
    print(f"Loading {file_path}")
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()
    print(f"Ingesting {file_path}")
    VectorStoreIndex.from_documents(document, storage_context=storage_context, insert_batch_size=8192, embed_model=embed_model, show_progress=True)
    #index = VectorStoreIndex.from_documents(document, storage_context=storage_context, insert_batch_size=8192, show_progress=True)
    print(f"Index for {file_path} completed")

if __name__ == "__main__":
    print("Make sure an Ollama container is running")
    Settings.llm = Ollama(model="llama2", request_timeout=360.0)
    #subprocess.run(["sudo", "docker", "run", "-d", "--name", "wiki_db", "-p", f"{FILLER_PORT}:6333", "-v", "wiki_db:/qdrant", "qdrant/qdrant"])

    #print("Loading documents")
    import time
    #documents = SimpleDirectoryReader(WIKI_DATA_DIR).load_data()

    print("Starting ingestion")
    wiki_db_client = QdrantClient(host="localhost", port=DB_PORT) 
    vector_store = QdrantVectorStore(client=wiki_db_client, collection_name="llamaindex_wiki_store") 
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    file_paths = [os.path.join(WIKI_DATA_DIR, f) for f in os.listdir(WIKI_DATA_DIR)]

#    start = time.perf_counter()
    with Pool() as pool:
        indices = pool.map(create_index, file_paths)
#    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", device="cuda", embed_batch_size=128)
#    documents = SimpleDirectoryReader(WIKI_DATA_DIR).load_data(0)
#    index = VectorStoreIndex.from_documents(documents, show_progress=True)
        
#    print(len(indices))

    #vector_store now updated to have all the indexes since all writing to same db?
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    doc_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
#    end = time.perf_counter()
#    print(end - start)

    #doc_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    print("Sending test queries to database")
    query_engine = doc_index.as_query_engine()
    print(query_engine.query("What is the etymology of anarchism?"))
    print(query_engine.query("What do the article's authors say about if scientists predicted COVID?"))
    print(query_engine.query("What is the difference between a sweater and cardigan?"))

#609.6568180549657 with parallel on 50m
#719.42 without parallel on 50m
#88.78 with parallel on parallel_test
#1337.882174 without parallel on 100m