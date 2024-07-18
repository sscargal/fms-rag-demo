#!/usr/bin/env python3

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import ArgPackComponent
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import NodeWithScore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.response_synthesizers import TreeSummarize
from qdrant_client import QdrantClient
from typing import Any
from typing import Dict
from typing import List 
from typing import Optional
import os
DOC_INDEX_PERSIST_DIR = "doc_index"
DOC_DATA_DIR = "data"
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

def load_or_create_index(doc_qdrant_client):
    doc_vector_store = QdrantVectorStore(client=doc_qdrant_client, collection_name="llamaindex_db")
    if os.path.exists(DOC_INDEX_PERSIST_DIR):
        print("Document index found on disk. Loading existing index.")
        storage_context = StorageContext.from_defaults(persist_dir=DOC_INDEX_PERSIST_DIR, vector_store=doc_vector_store)
        doc_index = load_index_from_storage(storage_context)
    else:
        print("No document index found on disk. Creating and ingesting a new index.")
        documents = SimpleDirectoryReader("data").load_data()
        storage_context = StorageContext.from_defaults(vector_store=doc_vector_store)

        doc_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        print("Persisting created index to disk.")
        doc_index.storage_context.persist(persist_dir="doc_index")
        
    return doc_index

llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=600.0)
doc_qdrant_client = QdrantClient(host="localhost", port=6333)
doc_index = load_or_create_index(doc_qdrant_client)
retriever = doc_index.as_retriever(similarity_top_k=2)

from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage

class CustomChatEngine(ContextChatEngine):
    def __init__(self, retriever, memory, **kwargs):
        super().__init__(**kwargs)
        self._retriever = retriever
        self._memory = memory

    def chat(self, message: str, **kwargs):
        # Get chat history
        chat_history = self._memory.get()

        # Retrieve relevant context
        retrieved_nodes = self._retriever.retrieve(message)
        retrieved_text = "\n".join([node.get_content() for node in retrieved_nodes])

        # Prepare messages for the LLM
        messages = [
            ChatMessage(role="system", content=f"You are a helpful assistant. Use the following context to answer questions: {retrieved_text}"),
            *chat_history,
            ChatMessage(role="user", content=message)
        ]

        # Generate response
        response = self._llm.chat(messages)

        # Update memory
        self._memory.put(ChatMessage(role="user", content=message))
        self._memory.put(ChatMessage(role="assistant", content=str(response)))

        return response
    
from llama_index.core.memory import ChatMemoryBuffer, SimpleComposableMemory

# Create a primary memory buffer
chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Create the SimpleComposableMemory
composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer
)

chat_engine = CustomChatEngine(
    retriever=retriever,
    memory=composable_memory,
    llm=llm
)

response = chat_engine.chat("What is the article about?")
print(response)