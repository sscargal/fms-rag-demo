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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from fmsdemo import MOCK_CHAT_HISTORY_PATH
from fmsdemo import DOC_INDEX_PERSIST_DIR
from tqdm import tqdm
import pandas as pd
import time
import subprocess
import re
import os

DOC_DB_COLLECTION_NAME = "llamaindex_doc_db"
CHAT_DB_COLLECTION_NAME = "llamaindex_chat_db"
MOCK_CHAT_HISORY_LEN = 100
#this list contains "past" messages related to the queries run in the demo that we want to be retrieved
RELEVANT_DEMO_HISTORY = [ChatMessage.from_str("Could you summarize the purpose of the article for me?", "user"),
                         ChatMessage.from_str("The purpose of the article titled \"The origin of COVID-19\" appears to be discussing the emergence of the coronavirus pandemic and highlighting the need for increased understanding, surveillance, research, and prevention efforts to prevent future pandemics. The authors emphasize that this is necessary due to the growing threat posed by emerging and reemerging infectious agents in the natural world, many of which have yet to be identified and studied. They also discuss the importance of international collaboration, expanding research, changing human behaviors that increase contact with bats and other potential virus hosts, strengthening public health infrastructure, developing effective antivirals and vaccines, and investing in the development of broadly protective vaccines and therapeutic agents against taxonomic groups likely to emerge in the future.", "assistant"),
                         ChatMessage.from_str("Can you elaborate more on the prevention methods the authors of the article outline?", "user"),
                         ChatMessage.from_str(("In the article titled \"Origin and Prevention of COVID-19,\" the authors propose various methods for preventing future coronavirus pandemics. Here are some key points they suggest:\n" 
                                            "1. Robust expansion of surveillance and research, with a focus on virologic studies related to human and animal spillover risks and means of reducing them. The article emphasizes the importance of field studies of humans and animals in disease hotspots."
                                            "2. Development of platform technologies for diagnostics, vaccines, and animal models for studies of pathogenesis and potential therapeutics. This is crucial to create tools for disease control.\n"
                                            "3. Increasing international collaboration involving many countries, especially recruiting scientists from China and other hotspot countries to these efforts.\n"
                                            "4. Aggressive surveillance of known coronavirus hotspots to learn more about the local viral ecology and identify initial human spillover events.\n"
                                            "5. Changing human behaviors that pose a risk for coming into contact with bats, such as wet markets, bat cave tourism, capturing and eating bats, and perturbing the environment in ways that alter bat habitats and habits.\n"
                                            "6. Strengthening basic public health, including hygiene and sanitation to prevent viruses from amplifying replication. Building and maintaining a strong public health infrastructure for an efficient response to pathogen emergence is essential.\n"
                                            "7. Developing effective antivirals and, ideally, broadly protective vaccines against emerging viruses. Education and communication with populations where spillover events occur are also important components of risk reduction.\n"
                                            "8. Realizing that the problem is larger than just coronaviruses; they suggest focusing on other high-risk pathogens like henipaviruses, filoviruses, etc., and developing broadly protective vaccines and antiviral/antimicrobial agents against them.\n"
                                            "The authors also stress that investing more in critical and creative laboratory, field, and behavioral research is essential to prevent future pandemics. They suggest that pandemic prevention should be a global effort on a par with chemical and nuclear weapon prevention. \n"
                                            "user: Can you elaborate more on the prevention methods the authors of the article outline?"), "assistant"),
                        ChatMessage.from_str("Can you tell me about the etymology of the term anarchism?", "user"),
                        ChatMessage.from_str(("Certainly! Based on the provided context, I can provide information on the etymology of the term \"anarchism.\""
                                            "The term \"anarchism\" originates from the Ancient Greek words \"anarkhia\" (ἀναρχία) and \"arkhos\" (ἄρχος), which mean \"without a ruler\" or \"leader,\" respectively. The suffix \"-ism\" denotes the ideological current that favors anarchy."
                                            "The first political philosopher to call himself an \"anarchist\" was Pierre-Joseph Proudhon in the mid-19th century, marking the formal birth of anarchism. However, as noted in the context, the term \"libertarianism\" has often been used as a synonym for anarchism, particularly outside the United States."
                                            "Anarchism is contrasted to socialist forms that are state-oriented or from above, and scholars of anarchism generally highlight its socialist credentials while criticizing attempts to create dichotomies between the two."
                                            "In summary, the term \"anarchism\" originates from Ancient Greek words and refers to a political philosophy that advocates for the abolition of authority and hierarchy, often in favor of voluntary free associations and stateless societies."), "assistant"),
                        ChatMessage.from_str("Can you summarize some facts about industry in Alabama?", "user"),
                        ChatMessage.from_str(("Based on the provided context, I can provide information on industry in Alabama. Alabama has a diverse industrial base, including the production of iron and steel products, paper, lumber, and wood products, mining (mainly coal), plastic products, cars, and trucks, and apparel. The state is also home to a growing automotive manufacturing industry, with several major auto manufacturers operating in the state, including Honda Manufacturing of Alabama, Hyundai Motor Manufacturing Alabama, Mercedes-Benz U.S. International, and Toyota Motor Manufacturing Alabama. In addition, the state is home to aerospace and electronic products, mostly in the Huntsville area, which is also home to NASA's George C. Marshall Space Flight Center and the U.S. Army Materiel Command."
                                            "In recent years, the state has experienced significant growth in its steel industry, with Outokumpu, Nucor, SSAB, ThyssenKrupp, and U.S. Steel operating facilities in the state and employing over 10,000 people. The Hunt Refining Company, a subsidiary of Hunt Consolidated, Inc., operates a refinery in Tuscaloosa, while JVC America, Inc. operates an optical disc replication and packaging plant in the same city."
                                            "The state has also seen an increase in tourism and entertainment, with attractions such as the Airbus A320 family aircraft assembly plant in Mobile, which was formally announced by Airbus CEO Fabrice Brégier in 2012 and began operating in 2015. The plant produces up to 50 aircraft per year by 2017."
                                            "Overall, Alabama's economy has diversified over the years, with a mix of traditional and new industries driving growth and development in the state."), "assistant")]

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
        ports = list(set(re.findall(r'(?<=:)\d+(?=->)', result.stdout)))
        ports.sort()
        return ports
    except subprocess.CalledProcessError: 
        return "Error: docker ps failed" 
     
#returns a tuple of doc_index, ingestion_time. If ingestion was skipped, ingestion_time will be None
#this sort of only exists for debug purposes. In the future, persisting to and reading from disk could be useful
#for benchmarking. But within the demo, ingestion must always be run to benchmark the ingestion time
def load_or_create_doc_index(doc_db_client, doc_data_dir):
    ingestion_time = None
    doc_vector_store = QdrantVectorStore(client=doc_db_client, collection_name=DOC_DB_COLLECTION_NAME)
    if os.path.exists(DOC_INDEX_PERSIST_DIR):
        print("Document index found on disk. Loading existing index.")
        storage_context = StorageContext.from_defaults(persist_dir=DOC_INDEX_PERSIST_DIR, vector_store=doc_vector_store)
        doc_index = load_index_from_storage(storage_context)
    else:
        print("No document index found on disk. Creating and ingesting a new index into qdrant database.")
        print("Loading data")
        documents = SimpleDirectoryReader(doc_data_dir).load_data()
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
        ingestion_time = end_time - start_time
        print(f"Total ingestion time: {ingestion_time:.6f} seconds")
        print("Persisting created index to disk.")
        doc_index.storage_context.persist(persist_dir=DOC_INDEX_PERSIST_DIR)
    return doc_index, ingestion_time

def create_chat_history(vector_memory):
    df = pd.read_csv(MOCK_CHAT_HISTORY_PATH)
    df = df[:MOCK_CHAT_HISORY_LEN] #more of a debug line. Only loads in part of the csv to reduce memory creation time for testing purposes
    for i in tqdm(range(MOCK_CHAT_HISORY_LEN), desc="Creating mock chat history"): #df.iterrows() behaves weirdly with tqdm, so manually pull the rows instead 
        row = df.iloc[i]
        msg = ChatMessage(role="user", content=row['question'])
        vector_memory.put(msg)
        unparsed_answer = row['answers'] #csv answer col contains the actual answer and other info that needs to be removed
        parsed_answer = re.search(r"(?<=\[).+?(?=\])", unparsed_answer).group(0) 
        msg = ChatMessage(role="assistant", content=parsed_answer) 
        vector_memory.put(msg)

    #add past messages relevant to the demo to memory
    for msg in RELEVANT_DEMO_HISTORY:
        vector_memory.put(msg)

#takes in the name of the model to run and a list of the queries to pass into the pipeline
def run_queries(model, queries, doc_data_dir):
    qdrant_container_ports = get_qdrant_container_ports()
    if len(qdrant_container_ports) < 2: 
        raise RuntimeError("At least two qdrant databases must be running")

    #set embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    #load data
    doc_db_client = QdrantClient(host="localhost", port=qdrant_container_ports[0])
    doc_index, ingestion_time = load_or_create_doc_index(doc_db_client, doc_data_dir)

    #create a primary memory buffer
    chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1000)

    #either create memory with past session chat history
    chat_db_client = QdrantClient(host="localhost", port=qdrant_container_ports[1])
    chat_db_vector_store = QdrantVectorStore(client=chat_db_client, collection_name=CHAT_DB_COLLECTION_NAME)
    vector_memory = VectorMemory.from_defaults(vector_store=chat_db_vector_store, 
                                               retriever_kwargs={"similarity_top_k": 4})

    chat_db_collections = chat_db_client.get_collections().collections
    if any(collection.name == CHAT_DB_COLLECTION_NAME for collection in chat_db_collections):
        print("Chat history already exists in database client. Skipping creating a mock chat history.")
    else:
        chat_db_client.create_collection(
            collection_name=CHAT_DB_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
        create_chat_history(vector_memory)

    #create composable memory
    composable_memory = SimpleComposableMemory.from_defaults(
        primary_memory=chat_memory_buffer,
        secondary_memory_sources=[vector_memory]
    )

    pipeline_memory = composable_memory 

    from query_pipeline import build_and_run_pipeline

    #model = args[1] #model name was passed as script arg from fmsdemo.py
    benchmark_df = build_and_run_pipeline(doc_index, pipeline_memory, queries, model)
    ingest_time_col = []
    #create an ingestion time column (of the right length to match the df's rows) for the query runs of this configuration
    #ingestion only happens once, so all queries under the given demo configuration will be listed with the same time
    for i in range(len(benchmark_df)):
        if ingestion_time is None:
            ingest_time_col.append(pd.NA)
        else:
            ingest_time_col.append(ingestion_time)
    benchmark_df.insert(loc=0, column="ingestion time", value=ingest_time_col)
    return benchmark_df

    #trying to make them all use the same collection name or same docker volume causes conflicts. Collection is written to 
    #to docker volume and updates in the other mounted clients storage/collections directory, so will say  
    #collection already exists, but then will say it doesn't exist later. Metadata or something important
    #maybe stored in storage/collection? This may also only be because using the same local host volume mounted
    #to each of their storage dirs