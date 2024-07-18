#!/usr/bin/env python3

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.memory import SimpleComposableMemory
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.memory import VectorMemory
from llama_index.core.llms import ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama 
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.vector_stores.qdrant import QdrantVectorStore
from fmsdemo import DOC_INDEX_PERSIST_DIR
from fmsdemo import DOC_DATA_DIR
from fmsdemo import DOC_DB_COLLECTION_NAME
from fmsdemo import CHAT_INDEX_PERSIST_DIR
from fmsdemo import CHAT_DB_COLLECTION_NAME
#import numpy as np
import time
#import logging
#import sys
import subprocess
import re
import os


#enable debug logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#def qdrant_test_search(client):
#    return(client.search(
#        collection_name="llamaindex_db",
#        query_vector = np.random.rand(768).tolist(),
#        limit=1
#    ))


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
    
def load_or_create_doc_index(doc_qdrant_client):
    doc_vector_store = QdrantVectorStore(client=doc_qdrant_client, collection_name=DOC_DB_COLLECTION_NAME)
    if os.path.exists(DOC_INDEX_PERSIST_DIR):
        print("Document index found on disk. Loading existing index.")
        storage_context = StorageContext.from_defaults(persist_dir=DOC_INDEX_PERSIST_DIR, vector_store=doc_vector_store)
        doc_index = load_index_from_storage(storage_context)
    else:
        print("No document index found on disk. Creating and ingesting a new index into qdrant database.")
        print("Loading data")
        documents = SimpleDirectoryReader(DOC_DATA_DIR).load_data()
        storage_context = StorageContext.from_defaults(vector_store=doc_vector_store)
        start_time = time.perf_counter()
        print(f"Starting ingestion. Starting time: {start_time}")

        doc_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        end_time = time.perf_counter()
        print(f"Ingestion complete. Ending time: {end_time}")
        print(f"Total ingestion time: {(end_time - start_time):.6f} seconds")
        print("Persisting created index to disk.")
        doc_index.storage_context.persist(persist_dir=DOC_INDEX_PERSIST_DIR)
    return doc_index
     
#set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

qdrant_container_ports = get_qdrant_container_ports()
if len(qdrant_container_ports) < 2: 
    raise RuntimeError("At least two qdrant containers must be running")

#load data
doc_qdrant_client = QdrantClient(host="localhost", port=qdrant_container_ports[0])
doc_index = load_or_create_doc_index(doc_qdrant_client)

#create secondary memory sources by ingesting data into qdrant containers




print("Ingesting documents into qdrant database")
secondary_memory_sources = []
client = QdrantClient(host="localhost", port=qdrant_container_ports[0])
if client.get_collections().collections: #skip reindexing if collection already exists
    print(f"Database already exists in container. Skipping indexing")
else:
    vector_store = QdrantVectorStore(client=client, collection_name=DOC_DB_COLLECTION_NAME)
#trying to make them all use the same collection name causes conflicts. Collection is written to 
#to docker volume and updates in the other mounted clients storage/collections directory, so will say  
#collection already exists, but then will say it doesn't exist later. Metadata or something important
#maybe stored in storage/collection? This may also only be because using the same local host volume mounted
#to each of their storage dirs
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

if not os.path.exists(DOC_INDEX_PERSIST_DIR):
    print(f"Ingesting index into the qdrant database using host port {port}. Starting time: {start_time}")
    start_time = time.perf_counter()

    #create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    end_time = time.perf_counter()
    print(f"Ingestion complete. Ending time: {end_time}")
    print(f"Total ingestion time: {(end_time - start_time):.6f} seconds")
    print("Persisting created index to disk.")
    index.storage_context.persist(persist_dir=DOC_INDEX_PERSIST_DIR)


#set up qdrant database that will be used as secondary memory
chat_history_qdrant_client = QdrantClient(host="localhost", port=qdrant_container_ports[1])

chat_history_vector_store = QdrantVectorStore(client=chat_history_qdrant_client, collection_name=CHAT_DB_COLLECTION_NAME)
from qdrant_client import models 
if chat_history_qdrant_client.get_collections().collections: #skip reindexing if collection already exists:
    print(f"{CHAT_DB_COLLECTION_NAME} already exists in container. Skipping indexing")
else:
    chat_history_qdrant_client.create_collection(
        collection_name=CHAT_DB_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
vector_memory = VectorMemory.from_defaults(vector_store=chat_history_vector_store, retriever_kwargs={"similarity_top_k": 6})
message = ChatMessage(role="user", content="Test")
msgs = [
    ChatMessage.from_str("Bob likes burgers.", "user"),
    ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
    ChatMessage.from_str("Alice likes apples.", "user"),
    ChatMessage.from_str("I am talking about the COVID 19 pandemic", "assistant"),
    ChatMessage.from_str("When was the COVID 19 pandemic?", "user"),
]
for msg in msgs:
    vector_memory.put(msg)

#if not os.path.exists(CHAT_DB_COLLECTION_NAME):
#    print("Creating mock chat history database")
#    vector_memory = VectorMemory.from_defaults(
#        retriever_kwargs={"similarity_top_k": 6}
#    )
#    msgs = [
#        ChatMessage.from_str("Bob likes burgers.", "user"),
#        ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
#        ChatMessage.from_str("Alice likes apples.", "user"),
#        ChatMessage.from_str("I am talking about the COVID 19 pandemic", "assistant"),
#        ChatMessage.from_str("When was the COVID 19 pandemic?", "user"),
#    ]
#    vector_memory.set(msgs)
#
#    print("Ingesting mock database")
#    chat_history_qdrant_client = QdrantClient(host="localhost", port=qdrant_container_ports[1])
#    vector_store = QdrantVectorStore(client=chat_history_qdrant_client, collection_name=CHAT_DB_COLLECTION_NAME)
#    storage_context = StorageContext.from_defaults(vector_store=vector_store)
#    chat_index = VectorStoreIndex.from_vector_store(
#        vector_store=vector_store,
#        storage_context=storage_context
#    )
#
#    og_vector_store = vector_memory._vector_store
#    all_nodes = og_vector_store.get_all_nodes()
#    print(f"DEBUG: {all_nodes}")
#    chat_index.insert_nodes(all_nodes)
#    print("Persisting mock database to disk")
#    chat_index.storage_context.persist(persist_dir=CHAT_INDEX_PERSIST_DIR)
#
#    secondary_memory_sources.append(vector_memory)
#    
#    import numpy as np
#    print(chat_history_qdrant_client.search(
#        collection_name=CHAT_DB_COLLECTION_NAME,
#        query_vector = np.random.rand(768).tolist(),
#        limit=1
#    ))
        
#set up composable memory
composable_memory = SimpleComposableMemory(
    primary_memory=ChatMemoryBuffer.from_defaults(),
    secondary_memory_sources=secondary_memory_sources
)

##set up ollama container connection
#Settings.llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=600.0)
#
##create query engine
#print("Creating query engine with composable memory.")
#retriever = ComposableMemoryRetriever(memory=composable_memory)
#query_engine = RetrieverQueryEngine.from_args(retriever)
#
#query_engine.query("What is the article about?")

#-----------------------------
#query model
#-----------------------------
#print(query_from_index(index, "Do you know any articles about Japan's nuclear power?")) #note that this currently creates a query engine based off the most recently created index