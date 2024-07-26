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
from tqdm import tqdm
import pandas as pd #user will have to manually source python VE before running script (unless there's a way to programatically do this)
import subprocess
import sys
import signal
import os
import re
import argparse
import time
import json
    
#demo-wide constants    
DEMO_SCRIPTS = ["vectordb_manager.sh", "run_queries.py"]
#this dict's values are lists that contain the command for how the dbs will be set up. Its
#keys are what part of the demo the memory configuration is for/a short description of what the set up is
DEMO_DB_CONFIGS = {"DRAM only": ["-m", "1", "-s", "1"],
                   "CXL only": ["-m", "2", "-s", "1"]}
                   #"DRAM and CXL": ["./vectordb_manager.sh", "-n", "2", "-m", "1,2", "-b", "0"]}
DOC_INDEX_PERSIST_DIR = "doc_index"
QUERIES = ["What are some of the things Achilles did in Greek mythology?",
           "I told you a while ago what my favorite color is. Do you remember what I said?",
           "Did scientists predict the COVID 19 pandemic?",
           "What was the first question I asked you at the start of the current chat session?"]
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
                        ChatMessage.from_str("Hello! How can I help you today?", "assistant"),
                        ChatMessage.from_str("My favorite color is purple.", "user"),
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


def signal_handler(sig, frame):
    print("SIGINT sent")
    time.sleep(5)
    reset_demo()
    sys.exit(0)

def run_dstat(output_file):
    #start dstat in the background to measure disk and cpu util at different points in the demo
    print(f"Starting a background dstat process and writing results to {output_file}")
    if os.path.exists(output_file):
        print(f"Warning: {output_file} already exists. dstat output will be concatenated to the end of this file")
    process = subprocess.Popen(
        ["dstat", "--cpu", "--disk", "--page", "--mem", "--time", "--epoch", "--output", output_file, "1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    created_subprocesses.append(process)
    
def run_pcm_memory(output_file):
    with open(output_file, "w") as f:
        process = subprocess.Popen(
            ["sudo", "/opt/pcm/sbin/pcm-memory", "1", "-all"],# "-csv"],
            stdout=f,
            stderr=subprocess.DEVNULL
        )
        created_subprocesses.append(process)

#need a separate function to just clear vector database containers without clearing the Ollama container
def reset_demo():
    print("Clearing vector db containers")
    for container in created_containers:
        subprocess.run(["sudo", "docker", "stop", container])
        time.sleep(3)
        subprocess.run(["sudo", "docker", "remove", container])
    created_containers.clear()
    subprocess.run(["rm", "-r", DOC_INDEX_PERSIST_DIR]) #remove persisted index to force reingestion during next configuration

    print("Removing any created background processes")
    for process in created_subprocesses:
        print(f"Terminating {process}")
        process.send_signal(signal.SIGINT) #send SIGINT specifically to let commands write their outputs to files

   
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

def check_for_ollama_container():
    try:
        #filter docker ps to only show port info of qdrant-based containers
        result = subprocess.run(
            ["sudo", "docker", "ps", "--format", "{{.Ports}}", "--filter", "ancestor=ollama/ollama"],
            capture_output=True,
            text=True,
            check=True
        )

        if not result.stdout:
            raise RuntimeError("An Ollama container must already be running!")
    except subprocess.CalledProcessError: 
        return "Error: docker ps failed" 


#small wrapper to deal with running the db script and updating created_containers
def start_vector_db(script_command, created_containers):
    print(f"Running vector db script: {script_command}")
    try:
        script_output = subprocess.run(script_command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error trying to run vectordb_manager. Make sure ports 6333 and 6334 are free, no other running containers are mounted to any doc or chat db volumes, and no previous demo containers with conflicting names exist (whether running or stopped)")
        sys.exit()
    lines = script_output.stdout.split("\n")
    container_id = (lines[0].split())[-1]
    created_containers.append(container_id)    

def load_memory_from_volume(mem_config, chat_db_volume, chat_store_collection, created_containers):
    print(f"Creating new vector db with the data in {chat_db_volume}")
    script_command = ["./vectordb_manager.sh", "--name", "FMSDemo_chat_db", "--host-port", "6334", "-v", chat_db_volume] + mem_config
    start_vector_db(script_command, created_containers)
    
    chat_db_client = QdrantClient(host="localhost", port=6334)
    chat_vector_store = QdrantVectorStore(client=chat_db_client, collection_name=chat_store_collection)
    vector_memory = VectorMemory.from_defaults(vector_store=chat_vector_store, 
                                               retriever_kwargs={"similarity_top_k": 6})
    for msg in RELEVANT_DEMO_HISTORY:
        vector_memory.put(msg)
    return vector_memory
    
def create_new_chat_memory(mem_config, chat_store_collection, created_containers):
    print(f"Creating new vector db for chat messages")
    script_command = ["./vectordb_manager.sh", "--name", "FMSDemo_chat_db", "--host-port", "6334"] + mem_config
    start_vector_db(script_command, created_containers)

    chat_db_client = QdrantClient(host="localhost", port=6334)
    chat_vector_store = QdrantVectorStore(client=chat_db_client, collection_name=chat_store_collection)
    vector_memory = VectorMemory.from_defaults(vector_store=chat_vector_store, 
                                               retriever_kwargs={"similarity_top_k": 6})
    chat_db_client.create_collection(
        collection_name=chat_store_collection,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    for msg in RELEVANT_DEMO_HISTORY:
        vector_memory.put(msg)
    return vector_memory

def load_index_from_volume(mem_config, doc_db_volume, doc_store_collection, created_containers):
    print(f"Creating new vector db with the data in {doc_db_volume}")
    script_command = ["./vectordb_manager.sh", "--name", "FMSDemo_doc_db", "--host-port", "6333", "-v", doc_db_volume] + mem_config
    start_vector_db(script_command, created_containers)

    doc_db_client = QdrantClient(host="localhost", port=6333) 
    doc_vector_store = QdrantVectorStore(client=doc_db_client, collection_name=doc_store_collection)
    return VectorStoreIndex.from_vector_store(doc_vector_store)

def create_new_doc_index(mem_config, doc_data_dir, doc_store_collection, created_containers):
    print("Creating new vector db to ingest into")
    script_command = ["./vectordb_manager.sh", "--name", "FMSDemo_doc_db", "--host-port", "6333"] + mem_config
    start_vector_db(script_command, created_containers)

    doc_db_client = QdrantClient(host="localhost", port=6333) 
    doc_vector_store = QdrantVectorStore(client=doc_db_client, collection_name=doc_store_collection)
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
#    print("Persisting created index to disk.")
#    doc_index.storage_context.persist(persist_dir=DOC_INDEX_PERSIST_DIR)
    return doc_index, ingestion_time

if __name__ == "__main__":
    check_for_ollama_container()

    created_containers = [] #keep track of what docker containers are made so they can be removed as needed (either for error cleanup or to restart databases)
    created_subprocesses = [] #keep track of what subprocesses are made for similar reasons
    signal.signal(signal.SIGINT, signal_handler)

    for script in DEMO_SCRIPTS:
        subprocess.run(["chmod", "+x", script])    

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Specify the directory containing the document data to be ingested and indexed. Default: ../data/", default="../data/")
    parser.add_argument("-m", "--model", help="Specify the model to use for inferencing. Default: llama2", choices=["mistral:7b", "llama2", "llama3"], default="llama2")
    parser.add_argument("-b", "--benchmark-dir", dest="benchmark_dir", help="Specify the directory to write benchmark data to. If the directory does not exist, it will be created. Default: ../benchmarks/", default="../benchmarks/")
    parser.add_argument("--doc-db-volume", dest="doc_db_volume", help="Specify a docker volume containing a vector store of the ingested documents. db containers will be loaded with the volume's contents to avoid reingestion.")
    parser.add_argument("--doc-store-collection", dest="doc_store_collection", help="If loading a database from a volume, specify the name of the collection within the database containing the ingested documents vector store.", default="llamaindex_doc_store")
    parser.add_argument("--chat-db-volume", dest="chat_db_volume", help="Specify a docker volume containing a vector store of past chat history. db containers will be loaded with the volume's contents to avoid recreation and reingestion of the history.")
    parser.add_argument("--chat-store-collection", dest="chat_store_collection", help="If loading a database from a volume, specify the name of the collection within the database containing the chat history vector store.", default="llamaindex_chat_store")
    args = parser.parse_args()

    #validate args
    doc_data_dir = args.data
    if not os.path.isdir(doc_data_dir):
        reset_demo()
        raise argparse.ArgumentTypeError("Invalid argument for -d. Please specify a valid directory which contains the documents to ingest. Absolute paths are needed for directories outside FMSDemo/scripts/")

    benchmark_dir = args.benchmark_dir
    if not os.path.isdir(benchmark_dir):
        print(f"No directory found at {benchmark_dir}. Creating the directory. If this was not intended, note that absolute paths are needed to specify existing directories outside FMSDemo/scripts/")
        try:
            subprocess.run(["mkdir", benchmark_dir], check=True)
        except subprocess.CalledProcessError:
            print(f"Error: mkdir failed to create a directory at {benchmark_dir}")
            reset_demo()
            sys.exit()
    if benchmark_dir[-1] != "/": #benchmark_dir needs to end with a slash because later parts assume it does  
        benchmark_dir += "/"

    doc_db_volume = args.doc_db_volume
    doc_store_collection = args.doc_store_collection
    chat_db_volume = args.chat_db_volume
    chat_store_collection = args.chat_store_collection

    model = args.model
    print(f"Starting Ollama container and {model} model")
    try:
        subprocess.run(["sudo", "docker", "exec", "-d", "ollama", "ollama", "run", model], check=True)
    except subprocess.CalledProcessError:
        print("Error: docker exec failed to start the model in the Ollama container") 
        reset_demo()
        sys.exit()

    benchmark_dfs = {}

    #run through each part of the demo
    for desc, mem_config in DEMO_DB_CONFIGS.items():
        print(f"Running demo with {desc}")

        benchmark_file_prefix = benchmark_dir + desc.replace(" ", "_") #create separate benchmark files for each demo config within the benchmark dir
        run_dstat(f"{benchmark_file_prefix}_dstat.csv")

        print("Setting up vector dbs and SimpleComposableMemory")
        print("Setting up document vector db")
        ingestion_time = None
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")     
        if doc_db_volume:
            doc_index = load_index_from_volume(mem_config, doc_db_volume, doc_store_collection, created_containers)
        else:
            doc_index, ingestion_time = create_new_doc_index(mem_config, doc_data_dir, doc_store_collection, created_containers)

        chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1000)        
        print("Setting up past chat history vector db")
        if chat_db_volume:
            vector_memory = load_memory_from_volume(mem_config, chat_db_volume, chat_store_collection, created_containers) 
        else:
            vector_memory = create_new_chat_memory(mem_config, chat_store_collection, created_containers)
        pipeline_memory = SimpleComposableMemory.from_defaults(
            primary_memory=chat_memory_buffer,
            secondary_memory_sources=[vector_memory]
        )
        
        from query_pipeline import build_and_run_pipeline
        benchmark_df = build_and_run_pipeline(doc_index, pipeline_memory, QUERIES, model)
               
        print("Resetting for next demo configuration")
        benchmark_dfs[desc] = benchmark_df
        pipeline_df_path = benchmark_file_prefix + "_benchmarks.csv"
        print(f"Writing results tof {pipeline_df_path}")
        benchmark_df.to_csv(pipeline_df_path)

        reset_demo() 

    for key, value in benchmark_dfs.items():
        print(f"{key}: {value}")
        
    configs_path = benchmark_dir + "demo_configs.json"
    print(f"Writing demo configurations to {configs_path}")
    with open (configs_path, "w") as f:
        json.dump(DEMO_DB_CONFIGS, f)
    print("Ending demo")
    reset_demo()
