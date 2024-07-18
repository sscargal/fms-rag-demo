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
from llama_index.core.query_pipeline import FnComponent 
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.memory import SimpleComposableMemory
from llama_index.core.memory import VectorMemory
from qdrant_client import QdrantClient
from fmsdemo import DOC_INDEX_PERSIST_DIR
from fmsdemo import DOC_DATA_DIR
from fmsdemo import DOC_DB_COLLECTION_NAME
from fmsdemo import CHAT_DB_COLLECTION_NAME
from fmsdemo import CHAT_INDEX_PERSIST_DIR
import os

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
            "You are a very helpful Q&A system. To assist in answering a user message, you will be provided with "
            "the following: chat history from this current user session, chat history from past sessions other than "
            "the current one, and possibly relevant context from documents. Unless the user specifies otherwise, note that "
            "the user's references to past chat history likely refer to previous messages in the current session. "
            "If you are not provided with sufficient context to respond to the user, simply say it "
            "and do not guess the answer."
        )
    )
])
    
CONTEXT_PROMPT = ChatPromptTemplate([
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "Here is some context that may be relevant:\n"
            "-----\n"
            "Chat History:\n\n"
            "{chat_history_str}\n"
            "-----\n"
            "Document Data:\n\n"
            "{node_context}\n"
            "-----\n"
            "Please write a response to the following query, using the above context if it is relevant:\n"
            "{query_str}"
        )
    )
])


def load_or_create_doc_index(doc_qdrant_client):
    doc_vector_store = QdrantVectorStore(client=doc_qdrant_client, collection_name=DOC_DB_COLLECTION_NAME)
    if os.path.exists(DOC_INDEX_PERSIST_DIR):
        print("Document index found on disk. Loading existing index.")
        storage_context = StorageContext.from_defaults(persist_dir=DOC_INDEX_PERSIST_DIR, vector_store=doc_vector_store)
        doc_index = load_index_from_storage(storage_context)
    else:
        print("No document index found on disk. Creating and ingesting a new index.")
        documents = SimpleDirectoryReader(DOC_DATA_DIR).load_data()
        storage_context = StorageContext.from_defaults(vector_store=doc_vector_store)

        doc_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        print("Persisting created index to disk.")
        doc_index.storage_context.persist(persist_dir=DOC_INDEX_PERSIST_DIR)
        
    return doc_index

#manually .get() from primary and secondary and manually format to distinguish between current and past sessoin 
#sources
def get_chat_history_str(query_str, pipeline_memory):
    retrieved_messages = [str(msg) for msg in pipeline_memory.get(query_str)]
    lines = "\n".join(retrieved_messages).split("\n")


    #get start and end indices of secondary context messages 
    start_index = next((i for i, line in enumerate(lines) if line.startswith("=====Relevant messages from memory source ")), -1)
    end_index = next((i for i, line in enumerate(lines) if line.startswith("This is the end of the retrieved message dialogues.")), -1)
    
    if start_index == -1 or end_index == -1:
        return "\n".join(retrieved_messages) #return original output if markers not found
    
    chat_history_str = ["Current session chat history:\n"]
    chat_history_str += lines[end_index+1:]
    chat_history_str.append("Past session chat history:\n")
    chat_history_str += lines[start_index:end_index+1]

    #join the reorderd lines back into a single string
    return '\n'.join(chat_history_str)

    #looked through SimpleComposableMemory and VectorMemory source code and tested this function.
    #Passing in a query to .get() will activate primary and secondary memory retrievers with the settings
    #they were configured with, eg top_k vectors to return
    #return "\n".join([str(msg) for msg in retrieved_messages])

def rewrite_query(query_str, chat_history_str):
    rewrite_input = REWRITE_PROMPT.format(chat_history_str=chat_history_str, query_str=query_str)
    rewritten_query = llm.complete(rewrite_input)
    return str(rewritten_query)

def synthesize_response(query_str, retrieved_nodes, chat_history_str):
    node_context = "" 
    
    for idx, node in enumerate(retrieved_nodes):
        node_text = node.get_content(metadata_mode="llm")
        node_context += f"Context Chunk {idx}:\n{node_text}\n\n"
    
    # Combine system prompt and QA prompt
    combined_prompt = SYSTEM_PROMPT.format() + CONTEXT_PROMPT.format(
        node_context=node_context,
        query_str=query_str,
        chat_history_str=chat_history_str
    )
    print(f"COMBINED PROMPT: {combined_prompt}")
    
    response = "This is a debug string"
    # response = llm.complete(combined_prompt)
    return str(response)

#for whatever reason, this wrapper is needed for the pipeline to only return one output
def get_model_output(response):
    return response

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=600.0)
doc_qdrant_client = QdrantClient(host="localhost", port=6333)
doc_index = load_or_create_doc_index(doc_qdrant_client)#, DOC_DB_COLLECTION_NAME)
retriever = doc_index.as_retriever(similarity_top_k=6)
reranker = ColbertRerank(top_n=2)

# Create a primary memory buffer
chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1500)

#set up qdrant database as secondary memory
chat_history_qdrant_client = QdrantClient(host="localhost", port=6334)

chat_history_vector_store = QdrantVectorStore(client=chat_history_qdrant_client, collection_name=CHAT_DB_COLLECTION_NAME)
vector_memory = VectorMemory.from_defaults(vector_store=chat_history_vector_store, retriever_kwargs={"similarity_top_k": 6})
#TEST SECONDARY MEMORY
#vector_memory = VectorMemory.from_defaults(
#    vector_store=None,  # leave as None to use default in-memory vector store
#    retriever_kwargs={"similarity_top_k": 2},
#)
#msgs = [
#    ChatMessage.from_str("Bob likes burgers.", "user"),
#    ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
#    ChatMessage.from_str("Alice likes apples.", "user"),
#    ChatMessage.from_str("I am talking about the COVID 19 pandemic", "assistant"),
#    ChatMessage.from_str("When was the COVID 19 pandemic?", "user"),
#]
#vector_memory.set(msgs)

# Create the SimpleComposableMemory
composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory]
)

#change this when testing composable memory integration into this
pipeline_memory = composable_memory 

pipeline = QueryPipeline(
    modules={
        "input": InputComponent(),
        "memory_retriever": FnComponent(get_chat_history_str),
        "rewriter": FnComponent(rewrite_query),
        "query_retriever": retriever, 
        "reranker": reranker,
        "synthesizer": FnComponent(synthesize_response),
        "output": FnComponent(get_model_output),
    },
    verbose=True
)

pipeline.add_link("input", "rewriter", src_key="query_str", dest_key="query_str")
pipeline.add_link("input", "memory_retriever", src_key="query_str", dest_key="query_str")
pipeline.add_link("input", "memory_retriever", src_key="pipeline_memory", dest_key="pipeline_memory")
pipeline.add_link("memory_retriever", "rewriter", dest_key="chat_history_str")

pipeline.add_link("rewriter", "query_retriever") #rewrite user query using chat history

#rank the retrieved nodes
pipeline.add_link("query_retriever", "reranker", dest_key="nodes")
pipeline.add_link("input", "reranker", src_key="query_str", dest_key="query_str")

#synthesize most relevant context and chat history into query
pipeline.add_link("input", "synthesizer", src_key="query_str", dest_key="query_str")
#pipeline.add_link("rewriter", "synthesizer", dest_key="query_str") #pass rewritten query to synthesizer
pipeline.add_link("reranker", "synthesizer", src_key="nodes", dest_key="retrieved_nodes")
pipeline.add_link("memory_retriever", "synthesizer", dest_key="chat_history_str")

pipeline.add_link("synthesizer", "output", dest_key="response")

#create a chat session with the messages specified in user_inputs 
user_inputs = [ 
    "What do the authors of the article say about if scientists predicted the COVID pandemic?", 
    "What was the first question I asked you?",
    "Goodbye"
]

for msg in user_inputs:
    #enable debug logging
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # run pipeline
    response = pipeline.run(
        query_str=msg,
        pipeline_memory=pipeline_memory
    )

    print(response)

    #update memory
    user_msg = ChatMessage(role="user", content=msg)
    pipeline_memory.put(user_msg)
    print(str(user_msg))

    response_msg = ChatMessage(role="assistant", content=response)
    pipeline_memory.put(response_msg)
    print()