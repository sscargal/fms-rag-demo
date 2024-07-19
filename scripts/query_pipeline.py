#!/usr/bin/env python3

from llama_index.core import Settings
from llama_index.core import ChatPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import FnComponent 
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import TokenCountingHandler
import tiktoken
import pandas as pd
import time

REWRITE_PROMPT = ChatPromptTemplate([
    ChatMessage(
        role=MessageRole.SYSTEM,
        content = (
        "Write a query to a semantic search engine using context from the given chat history. "
        "Your query should capture the full intent of the user's message. Do not answer the content of the user's message yourself."
        "Merely rewrite the user's query in such a way that another engine could successfully answer what the user intends to ask."
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
            "the current one, and possibly relevant context from documents. Current session messages are listed chronologically, "
            "with the session's first user query at the top. "
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
def get_chat_history_str(query_str, pipeline_memory, benchmark_tracker):
    start_time = time.perf_counter()
    retrieved_memory = pipeline_memory.get(query_str)
    end_time = time.perf_counter()
    print(f"Composable memory context retrieval start time: {start_time}") 
    print(f"Composable memory context retrieval end time: {end_time}")
    delta = end_time - start_time
    benchmark_tracker.memory_retrieval_time = delta
    print(f"Total composable memory context retrieval time: {delta:.6f} seconds")

    retrieved_messages = [str(msg) for msg in retrieved_memory]
    composite_message = "\n".join(retrieved_messages) #joining the retrieved messages together is needed to get their token count  

    token_counter.reset_counts()
    token_count = len(token_counter.tokenizer(composite_message))
    print(f"Token count of retrieved context from composable memory: {token_count}")
    
    tps = token_count / delta
    print(f"Composable memory context retrieval tps: {tps}")
    benchmark_tracker.memory_retrieval_tps = tps
    
    lines = composite_message.split("\n") 
    #get start and end indices of the secondary memory source context 
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

def rewrite_query(query_str, chat_history_str, llm):
    rewrite_input = REWRITE_PROMPT.format(chat_history_str=chat_history_str, query_str=query_str)
    rewritten_query = llm.complete(rewrite_input)
    #rewritten_query = "COVID COVID COVID DEBUG"
    return str(rewritten_query)
    
#this function gives timing control over retrieval rather than just having the retriever itself as a module in the pipeline
def doc_index_retrieve(query_str, retriever, benchmark_tracker):
    start_time = time.perf_counter() 
    nodes = retriever.retrieve(query_str)
    end_time = time.perf_counter()
    print(f"Document context retrieval start time: {start_time}")
    print(f"Document context retrieval end time: {end_time}")
    delta = end_time - start_time
    benchmark_tracker.doc_retrieval_time = delta
    print(f"Total document context retrieval time: {delta:.6f} seconds")
    
    token_count = 0
    for node in nodes:
        token_counter.reset_counts()
        node_content = node.get_content()
        token_count += len(token_counter.tokenizer(node_content))
    print(f"Token count of retrieved context from documents: {token_count}")

    tps = token_count / delta 
    print(f"Document context retrieval tps: {tps}")
    benchmark_tracker.doc_retrieval_tps = tps 
    return nodes 

def synthesize_response(query_str, retrieved_nodes, chat_history_str, benchmark_tracker, llm):
    node_context = "" 
    
    for idx, node in enumerate(retrieved_nodes):
        node_text = node.get_content(metadata_mode="llm")
        node_context += f"Context Chunk {idx}:\n{node_text}\n\n"
    
    combined_prompt = SYSTEM_PROMPT.format() + CONTEXT_PROMPT.format(
        node_context=node_context,
        query_str=query_str,
        chat_history_str=chat_history_str
    )
    print(f"COMBINED PROMPT: {combined_prompt}")

    response = llm.complete(combined_prompt)
    pipeline_end_time = time.perf_counter()
    print(f"Pipeline ending time: {pipeline_end_time:6f}")
    pipeline_delta = pipeline_end_time - benchmark_tracker.pipeline_start_time 
    print(f"Total pipeline time: {pipeline_delta:6f} seconds")
    benchmark_tracker.pipeline_delta = pipeline_delta
    
    token_counter.reset_counts()
    token_count = len(token_counter.tokenizer(str(response)))
    print(f"Token count of pipeline response: {token_count}")
    
    tps = token_count / pipeline_delta
    print(f"Overall pipeline tps: {tps}")
    benchmark_tracker.pipeline_tps = tps

    return str(response)


def build_and_run_pipeline(doc_index, pipeline_memory, queries, model):
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
        },
        verbose=True
    )
    #pass benchmark tracker to benchmarked modules 
    pipeline.add_link("input", "memory_retriever", src_key="benchmark_tracker", dest_key="benchmark_tracker")
    pipeline.add_link("input", "query_retriever", src_key="benchmark_tracker", dest_key="benchmark_tracker")
    pipeline.add_link("input", "synthesizer", src_key="benchmark_tracker", dest_key="benchmark_tracker")

    pipeline.add_link("input", "rewriter", src_key="query_str", dest_key="query_str")
    pipeline.add_link("input", "memory_retriever", src_key="query_str", dest_key="query_str")
    pipeline.add_link("input", "memory_retriever", src_key="pipeline_memory", dest_key="pipeline_memory")
    pipeline.add_link("input", "rewriter", src_key="llm", dest_key="llm")
    pipeline.add_link("memory_retriever", "rewriter", dest_key="chat_history_str") #send context from memory to use in query rewrite 

    pipeline.add_link("input", "query_retriever", src_key="retriever", dest_key="retriever")
    pipeline.add_link("rewriter", "query_retriever", dest_key="query_str") #rewrite user query and retrieve context using that

    #rank the retrieved nodes
    pipeline.add_link("query_retriever", "reranker", dest_key="nodes")
    pipeline.add_link("input", "reranker", src_key="query_str", dest_key="query_str")

    #synthesize most relevant context and chat history into query
    pipeline.add_link("input", "synthesizer", src_key="query_str", dest_key="query_str")
    pipeline.add_link("input", "synthesizer", src_key="llm", dest_key="llm")
    pipeline.add_link("reranker", "synthesizer", src_key="nodes", dest_key="retrieved_nodes")
    pipeline.add_link("memory_retriever", "synthesizer", dest_key="chat_history_str")
    print("Pipeline setup complete!")

    print("Running queries")
    benchmark_tracker = BenchmarkTracker()
    dfs = [] 
    
    print(f"DEBUG: {model}")
    llm = Ollama(model=model, base_url="http://localhost:11434", request_timeout=600.0)

    for query in queries:
        print(f"QUERY: {query}")
        pipeline_start_time = time.perf_counter()
        print(f"Pipeline starting time: {pipeline_start_time:.6f}")
        benchmark_tracker.pipeline_start_time = pipeline_start_time 

        response = pipeline.run(
            query_str=query,
            retriever=retriever,
            pipeline_memory=pipeline_memory,
            llm=llm,
            benchmark_tracker=benchmark_tracker
        )
        print(f"PIPELINE OUTPUT: {response}")

        #update memory
        user_msg = ChatMessage(role="user", content=query)
        pipeline_memory.put(user_msg)
        response_msg = ChatMessage(role="assistant", content=response)
        pipeline_memory.put(response_msg)
        
        benchmarks = {
            "query": [query],
            "composable memory retrieval time": [benchmark_tracker.memory_retrieval_time],
            "composable memory retrieval tokens/sec": [benchmark_tracker.memory_retrieval_tps],
            "document index retrieval time": [benchmark_tracker.doc_retrieval_time],
            "document index retrieval tokens/sec": [benchmark_tracker.doc_retrieval_tps],
            "time to final result": [benchmark_tracker.pipeline_delta],
            "overall pipeline tokens/sec": [benchmark_tracker.pipeline_tps]
        } 
        benchmark_df = pd.DataFrame.from_dict(data=benchmarks)    
        dfs.append(benchmark_df)
        benchmark_tracker.reset() 
    #merge the individual dataframes from each pipeline run together
    return(pd.concat(dfs, ignore_index=True)) 

#FnComponent modules can only have one output key, so additional tracking outputs from each module have
#to be stored in some other structure in order to be accessed by other parts of the pipeline
class BenchmarkTracker:
    def __init__(self):
        self.pipeline_start_time = 0
        self.memory_retrieval_time = 0
        self.memory_retrieval_tps = 0
        self.doc_retrieval_time = 0
        self.doc_retrieval_tps = 0
        self.pipeline_delta = 0
        self.pipeline_tps = 0
        #when timing the overall pipeline, the start is considered as the pipeline.run() call
        #and the end when the model finishes its final inference within the pipeline

    def reset(self):
        self.pipeline_start_time = 0
        self.memory_retrieval_time = 0
        self.memory_retrieval_tps = 0
        self.doc_retrieval_time = 0
        self.doc_retrieval_tps = 0
        self.pipeline_delta = 0
        self.pipeline_tps = 0

#set up module-wide variables 
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
token_counter = TokenCountingHandler( 
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
Settings.callback_manager = CallbackManager([token_counter])