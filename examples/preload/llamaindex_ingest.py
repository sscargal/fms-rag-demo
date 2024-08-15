#!/usr/bin/env python3
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DB_PORT = 7000
DATA_DIR = "wiki/"
COLLECTION_NAME = "llamaindex_wiki_store"

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", embed_batch_size=256)
wiki_db_client = QdrantClient(host="localhost", port=DB_PORT) 
vector_store = QdrantVectorStore(client=wiki_db_client, collection_name=COLLECTION_NAME) 
storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print("Starting ingestion")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
print("Ingestion complete")