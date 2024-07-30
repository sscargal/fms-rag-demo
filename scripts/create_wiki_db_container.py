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
import argparse
import os

#parallel loading of documents
#parallel creation of indices for each document

#WIKI_DATA_DIR = "/data/jasonc/extracted_text"

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest="data_dir")
parser.add_argument("-p", dest="db_port")
args = parser.parse_args()
print("Make sure an Ollama container is running")
Settings.llm = Ollama(model="llama2", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", embed_batch_size=256)
print("Starting ingestion")
wiki_db_client = QdrantClient(host="localhost", port=args.db_port) 
vector_store = QdrantVectorStore(client=wiki_db_client, collection_name="llamaindex_wiki_store") 
storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader(args.data_dir).load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, insert_batch_size=8192, show_progress=True)

print("Sending test queries to database")
query_engine = index.as_query_engine()
print(query_engine.query("What is the etymology of anarchism?"))
print(query_engine.query("What do the article's authors say about if scientists predicted COVID?"))
print(query_engine.query("What is the difference between a sweater and cardigan?"))

#609.6568180549657 with parallel on 50m
#719.42 without parallel on 50m
#88.78 with parallel on parallel_test
#1337.882174 without parallel on 100m