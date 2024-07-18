#!/usr/bin/env python3

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import ChatPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import ArgPackComponent
from llama_index.core.query_pipeline import FnComponent 
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.memory import SimpleComposableMemory
from qdrant_client import QdrantClient
import os
DOC_INDEX_PERSIST_DIR = "doc_index"
DOC_DATA_DIR = "data"

REWRITE_PROMPT = ChatPromptTemplate([
    ChatMessage(
        role=MessageRole.SYSTEM,
        content = (
        "Please write a query to a semantic search engine using context from the chat history. "
        "Your query should capture the full intent of the user's message."
        "\n"
        "Chat history: {chat_history_str}"
        "\n"
        "Latest user message: {query_str}\n"
        'Query:"""\n'
        )
    )
])
    
SYSTEM_PROMPT = ChatPromptTemplate([
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are a very helpful Q&A system. You will be provided with " #the previous chat history, which includes "
            #"history from previous sessions, as well as possibly relevant context from documents " 
            "possibly relevant context from documents "
            "to assist in answering a user message." 
            #"If you are not provided with sufficient context to "
            #"respond to a question by the user, simply say it and do not guess the answer."
        )
    )
])
    
CONTEXT_PROMPT = ChatPromptTemplate([
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "Here is some context that may be relevant:\n"
            "-----\n"
            "{node_context}\n"
            "-----\n"
            "Please write a response to the following query, using the above context if it is relevant:\n"
            "{query_str}\n"
        )
    )
])



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

def rewrite_query(query_str, chat_history_str):
    rewrite_input = REWRITE_PROMPT.format(chat_history_str=chat_history_str, query_str=query_str)
    rewritten_query = llm.complete(rewrite_input)
    print(f"REWRITTEN QUERY: {rewritten_query}")
    return str(rewritten_query)

#function for a module that stores the rewritten query so rewriting only happens once
def get_rewritten_query(rewritten_query):
    return rewritten_query

def synthesize_response(query_str, retrieved_nodes):
    print(f"RETRIEVED NODES: {retrieved_nodes}")
    node_context = "" 
    
    for idx, node in enumerate(retrieved_nodes):
        node_text = node.get_content(metadata_mode="llm")
        node_context += f"Context Chunk {idx}:\n{node_text}\n\n"
    
    # Combine system prompt and QA prompt
    combined_prompt = SYSTEM_PROMPT.format() + CONTEXT_PROMPT.format(
        node_context=node_context,
        query_str=query_str
    )
    print(f"COMBINED PROMPT: {combined_prompt}")
    
    response = llm.complete(combined_prompt)
    return str(response)

#for whatever reason, this wrapper is needed for the pipeline to only return one output
def get_model_output(response):
    return response

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=600.0)
doc_qdrant_client = QdrantClient(host="localhost", port=6333)
doc_index = load_or_create_index(doc_qdrant_client)
retriever = doc_index.as_retriever(similarity_top_k=6)
reranker = ColbertRerank(top_n=3)
pipeline_memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Create a primary memory buffer
chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Create the SimpleComposableMemory
composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer
)

pipeline = QueryPipeline(
    modules={
        "input": InputComponent(),
        "rewriter": FnComponent(rewrite_query),
        "rewritten_query": FnComponent(get_rewritten_query),
        "query_retriever": retriever, 
        "rewrite_retriever": retriever,
        "join": ArgPackComponent(),
        "reranker": reranker,
        "synthesizer": FnComponent(synthesize_response),
        "output": FnComponent(get_model_output),
    },
    verbose=True
)

#use two retriever modules using same retrieval engine: one for rewritten query, one for the original
pipeline.add_link("input", "rewriter", src_key="query_str", dest_key="query_str")
pipeline.add_link("input", "rewriter", src_key="chat_history_str", dest_key="chat_history_str")

pipeline.add_link("rewriter", "rewritten_query") #rewrite user query using chat history
pipeline.add_link("rewritten_query", "rewrite_retriever") #retrieve context with rewritten query

pipeline.add_link("input", "query_retriever", src_key="query_str") #retrieve context with original query
#join the retrieved data into one argpack component (destination keys are needed but can be named anything)
pipeline.add_link("rewrite_retriever", "join", src_key="nodes", dest_key="rewrite_nodes")
pipeline.add_link("query_retriever", "join", src_key="nodes", dest_key="query_nodes")

#rank the relevance of both query versions' retrieved nodes
pipeline.add_link("join", "reranker", dest_key="nodes")
pipeline.add_link("input", "reranker", src_key="query_str", dest_key="query_str")

#synthesize most relevant context into query
#pipeline.add_link("rewriter", "synthesizer", src_key="query_str", dest_key="query_str")
pipeline.add_link("rewritten_query", "synthesizer", dest_key="query_str") #pass rewritten query to synthesizer
pipeline.add_link("reranker", "synthesizer", src_key="nodes", dest_key="retrieved_nodes")

pipeline.add_link("synthesizer", "output", dest_key="response")

#create a chat session with the messages specified in user_inputs 
user_inputs = [ 
    "What do the authors of the article say about if scientists predicted the COVID pandemic?", 
    "Can you tell me a little more about the emergence mechanisms behind it?",
    "What was the first question I asked you?",
    "Thanks, that's what I needed to know!",
]

for msg in user_inputs:
    #get memory
    chat_history = pipeline_memory.get()

    #prepare inputs
    chat_history_str = "\n".join([str(x) for x in chat_history])

    #enable debug logging
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # run pipeline
    response, intermediates = pipeline.run_with_intermediates(
        query_str=msg,
        chat_history_str=chat_history_str
    )
    print("Query Retriever Output:", intermediates["query_retriever"])
    print("Rewrite Retriever Output:", intermediates["rewrite_retriever"])
    print("Join Output:", intermediates["join"])

    print("pipeline run successful!")
    print(response)

    #update memory
    user_msg = ChatMessage(role="user", content=msg)
    pipeline_memory.put(user_msg)
    print(str(user_msg))

    response_msg = ChatMessage(role="assistant", content=response)
    pipeline_memory.put(response_msg)
    print()