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
import pandas as pd

REWRITE_PROMPT = ChatPromptTemplate([
    ChatMessage(
        role=MessageRole.SYSTEM,
        content = (
        "Write a query to a semantic search engine using context from the given chat history. "
        "Your query should capture the full intent of the user's message. Do not directly answer the user's message yourself."
        "Merely rewrite it in such a way that another engine could successfully answer what the user intends to ask."
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

#by default, SimpleComposableMemory's .get() method returns a strangely formatted composition of the primary and secondary sources (with primary
#being listed later and after a statement that the history has ended). This function reparses .get()'s output
def get_chat_history_str(query_str, pipeline_memory):
    retrieved_messages = [str(msg) for msg in pipeline_memory.get(query_str)]
    lines = "\n".join(retrieved_messages).split("\n")

    #get start and end indices of secondary context messages 
    start_index = next((i for i, line in enumerate(lines) if line.startswith("=====Relevant messages from memory source ")), -1)
    end_index = next((i for i, line in enumerate(lines) if line.startswith("This is the end of the retrieved message dialogues.")), -1)
    
    if start_index == -1 or end_index == -1:
        return "\n".join(retrieved_messages) #return original output if markers not found
    
    #label the current and past histories and put the current session (primary memory) first
    chat_history_str = ["Current session chat history:\n"]
    chat_history_str += lines[end_index+1:]
    chat_history_str.append("Past session chat history:\n")
    chat_history_str += lines[start_index:end_index+1]

    return '\n'.join(chat_history_str)

    #passing in a query to .get() will activate primary and secondary memory retrievers with the settings
    #they were configured with, eg top_k vectors to return. This is important for not exceeding context size

def rewrite_query(query_str, chat_history_str):
    rewrite_input = REWRITE_PROMPT.format(chat_history_str=chat_history_str, query_str=query_str)
    #rewritten_query = llm.complete(rewrite_input)
    return "DEBUG COVID COVID COVID"
    #return str(rewritten_query)

#this function gives timing control over retrieval rather than just having the retriever itself as a module in the pipeline
def doc_index_retrieve(query_str, retriever, benchmark_tracker):
    benchmark_tracker.memory_retrieval_tps = 100
    return retriever.retrieve(query_str)

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
    #response = llm.complete(combined_prompt)
    return str(response)

def create_benchmark_df(query_str):
    print("create benchmark df DEBUG")
    return pd.DataFrame(columns=["Query","Composable memory retrieval tokens/sec", "Document index retrieval tokens/sec",
                                 "Time to final result", "Overall pipeline tokens/sec"])

def get_pipeline_output(response, benchmark_df):
    print("DEBUG")
    return {"response": response, "benchmark_df": benchmark_df}

def build_and_run_pipeline(doc_index, pipeline_memory, queries):
    print("Building RAG pipeline")

    #set up pipeline components
    retriever = doc_index.as_retriever(similarity_top_k=6)
    reranker = ColbertRerank(top_n=2)

    #build pipeline
    pipeline = QueryPipeline(
        modules={
            "input": InputComponent(),
            "memory_retriever": FnComponent(get_chat_history_str),
            "rewriter": FnComponent(rewrite_query),
            "query_retriever": FnComponent(doc_index_retrieve), 
            "reranker": reranker,
            "synthesizer": FnComponent(synthesize_response),
            "df": FnComponent(create_benchmark_df),
            "output": FnComponent(get_pipeline_output),
        },
        verbose=True
    )
    #set up benchmark tracking
    pipeline.add_link("input", "query_retriever", src_key="benchmark_tracker", dest_key="benchmark_tracker")

    pipeline.add_link("input", "rewriter", src_key="query_str", dest_key="query_str")
    pipeline.add_link("input", "memory_retriever", src_key="query_str", dest_key="query_str")
    pipeline.add_link("input", "memory_retriever", src_key="pipeline_memory", dest_key="pipeline_memory")
    pipeline.add_link("memory_retriever", "rewriter", dest_key="chat_history_str") #send context from memory to use in query rewrite 

    pipeline.add_link("input", "query_retriever", src_key="retriever", dest_key="retriever")
    pipeline.add_link("rewriter", "query_retriever", dest_key="query_str") #rewrite user query and retrieve context using that

    #rank the retrieved nodes
    pipeline.add_link("query_retriever", "reranker", dest_key="nodes")
    pipeline.add_link("input", "reranker", src_key="query_str", dest_key="query_str")

    #synthesize most relevant context and chat history into query
    pipeline.add_link("input", "synthesizer", src_key="query_str", dest_key="query_str")
    pipeline.add_link("reranker", "synthesizer", src_key="nodes", dest_key="retrieved_nodes")
    pipeline.add_link("memory_retriever", "synthesizer", dest_key="chat_history_str")
    pipeline.add_link("synthesizer", "output", dest_key="response")
    
    #create dataframe with benchmark data
    pipeline.add_link("input", "df", src_key="query_str")
    pipeline.add_link("df", "output", dest_key="benchmark_df")

#    pipeline.add_link("input", "benchmark_df", src_key="query_str", dest_key="query_str")
#    pipeline.add_link("benchmark_df", "output", dest_key="benchmark_df")
#    pipeline.add_link("synthesizer", "output", dest_key="response")
#    pipeline.add_link("output", "output_fr")
    print("Pipeline setup complete!")

    print("Running queries")
    benchmark_tracker = BenchmarkTracker()

    for query in queries:
        print(query)

        output = pipeline.run(
            query_str=query,
            retriever=retriever,
            pipeline_memory=pipeline_memory,
            benchmark_tracker=benchmark_tracker
        )

        response = output["response"]
        print(output)
        print(response)

        #update memory
        user_msg = ChatMessage(role="user", content=query)
        pipeline_memory.put(user_msg)
        response_msg = ChatMessage(role="assistant", content=response)
        pipeline_memory.put(response_msg)
        
        print(benchmark_tracker.memory_retrieval_tps)
        benchmark_tracker.reset()
   # return create_benchmark_df(query, memory_retrieval_time, index_retrieval_time, time_to_result) 

#looking through source code, FnComponent modules can only have one output key, so additional tracking outputs from each module has
#to be stored in some other structure in order to be accessed by other parts of the pipeline
class BenchmarkTracker:
    def __init__(self):
        self.memory_retrieval_tps = 0

    def reset(self):
        self.memory_retrieval_tps = 0

#set up module-wide variables 
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=600.0)
benchmark_df = pd.DataFrame(columns=["Query", "Composable memory retrieval tokens/sec", "Document index retrieval tokens/sec",
                                     "Time to final result", "Overall pipeline tokens/sec"])
query = ""
memory_retrieval_time = -1
index_retrieval_time = -1
time_to_result = -1