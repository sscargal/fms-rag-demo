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
from qdrant_client import QdrantClient
from typing import Any
from typing import Dict
from typing import List 
from typing import Optional
import os
import logging
import sys

#configure global LlamaIndex settings and program constants
DOC_INDEX_PERSIST_DIR = "doc_index"
DOC_DATA_DIR = "data"
DEFAULT_CONTEXT_PROMPT = (
    "Here is some context that may be relevant:\n"
    "-----\n"
    "{node_context}\n"
    "-----\n"
    "Please write a response to the following question, using the above context:\n"
    "{query_str}\n"
)
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

#this is the LlamaIndex example in the docs except without validation or async  
class ResponseWithChatHistory(CustomQueryComponent):
    llm: Ollama = Field(..., description="Ollama LLM")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        # NOTE: These are required inputs. If you have optional inputs please override
        # `optional_input_keys_dict`
        return {"chat_history", "nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response"}

    def _prepare_context(
        self,
        chat_history: List[ChatMessage],
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )
        user_message = ChatMessage(role="user", content=formatted_context)

        chat_history.append(user_message)

        if self.system_prompt is not None:
            chat_history = [
                ChatMessage(role="system", content=self.system_prompt)
            ] + chat_history

        return chat_history

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            chat_history, nodes, query_str
        )

        response = llm.chat(prepared_context)

        return {"response": response}


#create index for documents or load it if it already exists on disk
doc_qdrant_client = QdrantClient(host="localhost", port=6333)
doc_index = load_or_create_index(doc_qdrant_client)

#create an input component to capture the user query
input_component = InputComponent()

#use the LLM to rewrite a user query
rewrite = (
    "Please write a query to a semantic search engine using the current conversation.\n"
    "\n"
    "\n"
    "{chat_history_str}"
    "\n"
    "\n"
    "Latest message: {query_str}\n"
    'Query:"""\n'
)
rewrite_template = PromptTemplate(rewrite)
llm = Ollama(model="mistral:7b", base_url="http://localhost:11434", request_timeout=360.0)

#we will retrieve two times, so we need to pack the retrieved nodes into a single list
argpack_component = ArgPackComponent()

#with the nodes, retrieve vectors and postprocess/rerank
retriever = doc_index.as_retriever(similarity_top_k=6)
reranker = ColbertRerank(top_n=3)

#initialize custom response component
response_component = ResponseWithChatHistory(
    llm=llm,
    system_prompt=(
        "You are a Q&A system. You will be provided with the previous chat history, "
        "as well as possibly relevant context, to assist in answering a user message."
    ),
)

#link all the components into a pipeline
pipeline = QueryPipeline(
    modules={
        "input": input_component,
        "rewrite_template": rewrite_template,
        "llm": llm,
        "rewrite_retriever": retriever,
        "query_retriever": retriever,
        "join": argpack_component,
        "reranker": reranker,
        "response_component": response_component,
    },
    verbose=True,
    show_progress=True
)

# run both retrievers -- once with the hallucinated query, once with the real query
pipeline.add_link(
    "input", "rewrite_template", src_key="query_str", dest_key="query_str" #write query_str stored in input to query_str variable in rewrite_template
)
pipeline.add_link( #write chat_history_str stored in input to query_str variable in rewrite_template
    "input",
    "rewrite_template",
    src_key="chat_history_str",
    dest_key="chat_history_str",
)
pipeline.add_link("rewrite_template", "llm")
pipeline.add_link("llm", "rewrite_retriever")
pipeline.add_link("input", "query_retriever", src_key="query_str") #pass query_str to retriever to retrieve simliar vectors

# each input to the argpack component needs a dest key -- it can be anything
# then, the argpack component will pack all the inputs into a single list
pipeline.add_link("rewrite_retriever", "join", dest_key="rewrite_nodes")
pipeline.add_link("query_retriever", "join", dest_key="query_nodes")

# reranker needs the packed nodes and the query string
pipeline.add_link("join", "reranker", dest_key="nodes")
pipeline.add_link(
    "input", "reranker", src_key="query_str", dest_key="query_str"
)

# synthesizer needs the reranked nodes and query str
pipeline.add_link("reranker", "response_component", dest_key="nodes")
pipeline.add_link(
    "input", "response_component", src_key="query_str", dest_key="query_str"
)
pipeline.add_link(
    "input",
    "response_component",
    src_key="chat_history",
    dest_key="chat_history",
)


pipeline_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

#create a chat session with the messages specified in user_inputs
user_inputs = [
    "Hello!",
    "Can you explain if scientists predicted the COVID pandemic?" 
    "Can you tell me a little more about the emergence mechanisms behind it?"
    "What was the first question I asked you?"
    "Thanks, that's what I needed to know!",
]

for msg in user_inputs:
    # get memory
    chat_history = pipeline_memory.get()

    # prepare inputs
    chat_history_str = "\n".join([str(x) for x in chat_history])

    #enable debug logging
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # run pipeline
    response = pipeline.run(
        query_str=msg,
        chat_history=chat_history,
        chat_history_str=chat_history_str,
    )

    # update memory
    user_msg = ChatMessage(role="user", content=msg)
    pipeline_memory.put(user_msg)
    print(str(user_msg))

    pipeline_memory.put(response.message)
    print(str(response.message))
    print()