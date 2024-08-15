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
from query_pipeline import build_and_run_pipeline
import pandas as pd 
import streamlit as st
import subprocess
import signal
import os
import re
import time
    
#this list contains "past" messages related to the documents in the demo that we want to be retrieved
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
                        ChatMessage.from_str("My friend James' favorite color is purple.", "user"),
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
DOC_DB_PORT = 7000
CHAT_DB_PORT = 7001

#note that dstat and pcm memory are currently not run during the demo. Call these functions
#to add that functionality, but beware that they can easily leak if the script gets interrupted
#with an unexpected error
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
    st.session_state.created_subprocesses.append(process)
    
def run_pcm_memory(output_file):
    with open(output_file, "w") as f:
        process = subprocess.Popen(
            ["sudo", "/opt/pcm/sbin/pcm-memory", "1", "-all"],# "-csv"],
            stdout=f,
            stderr=subprocess.DEVNULL
        )
        st.session_state.created_subprocesses.append(process)

def reset_demo():
    if "created_containers" in st.session_state:
        print("Clearing vector db containers")
        for container in st.session_state.created_containers:
            subprocess.run(["sudo", "docker", "stop", container])
            time.sleep(3)
            subprocess.run(["sudo", "docker", "remove", container])
        st.session_state.created_containers.clear()

    if "created_subprocesses" in st.session_state:
        print("Removing any created background processes")
        for process in st.session_state.created_subprocesses:
            print(f"Terminating {process}")
            process.send_signal(signal.SIGINT) #send SIGINT specifically to let commands write their outputs to files
            
    if "messages" in st.session_state:
        print("Clearing session state message history")
        st.session_state.messages.clear()
   
def check_for_ollama_container():
    try:
        result = subprocess.run(
            ["sudo", "docker", "ps", "--filter", "ancestor=ollama/ollama"],
            capture_output=True,
            text=True,
            check=True
        )

        if not result.stdout:
            raise RuntimeError("An Ollama container must already be running! Please start a docker container with your desired settings and refresh the page")
    except subprocess.CalledProcessError: 
        return "Error: docker ps failed" 

def start_vector_db(docker_command):
    print(f"Running the following command: {docker_command}")
    st.text("Creating database container")
    command_result = None
    try:
        command_result = subprocess.run(docker_command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Error trying to run {docker_command}. Are the host port and container name available?")
    container_id = command_result.stdout.strip()
    print(f"Created container id: {container_id}")
    st.session_state.created_containers.append(container_id)    
    #extract the host port bound to container port 6333
    command_result = subprocess.run(["sudo", "docker", "inspect",
                                     "--format={{(index (index .NetworkSettings.Ports \"6333/tcp\") 0).HostPort}}",
                                     container_id], capture_output=True, text=True)
    port = command_result.stdout.strip()
    #couldn't find a good command to use for a docker healthcheck, so check for when the database is ready by testing the client connection
    with st.spinner("Starting vector database"):
        st.text("Letting the database warm up")
        time.sleep(5)
        MAX_ATTEMPTS = 5
        ATTEMPT_DELAY = 10
        st.text("Connecting to the database")
        
        client = QdrantClient(host="localhost", port=port)
        print(f"Host port connected to container: {port}")
        for i in range(0, MAX_ATTEMPTS):
            st.text(f"Attempt {i+1}")
            try:
                client.get_collections()
                st.success("Connected to database!")
                return client
            except:
                st.text(f"Connection failed. Retrying in {ATTEMPT_DELAY} seconds")
            time.sleep(ATTEMPT_DELAY)
        
        raise RuntimeError("Couldn't connect to the created database")
    
def validate_collection(client, collection):
    collections = client.get_collections()
    if collection not in [collection.name for collection in collections.collections]:
        reset_demo()
        raise RuntimeError(f"The collection {collection} wasn't found in the database!")

#the similarity_top_k args passed in both chat memory creation functions will determine the final
#amount of messages retrieved from the db by the pipeline (ie there is no reranking to a smaller number)
def load_chat_memory_from_volume(mem_config, chat_db_volume, chat_store_collection):
    st.text(f"Loading the data in {chat_db_volume} into a database container")
    docker_command = ["sudo", "docker", "run", "-d", "--name", "FMSDemo_chat_db", "-p", f"{CHAT_DB_PORT}:6333", "-v", f"{chat_db_volume}:/qdrant"] + mem_config + ["qdrant/qdrant"]
    chat_db_client = start_vector_db(docker_command)
    validate_collection(chat_db_client, chat_store_collection)
    chat_vector_store = QdrantVectorStore(client=chat_db_client, collection_name=chat_store_collection)
    vector_memory = VectorMemory.from_defaults(vector_store=chat_vector_store, 
                                               retriever_kwargs={"similarity_top_k": 2})
    st.text("Putting relevant demo messages into database")
    for msg in RELEVANT_DEMO_HISTORY:
        vector_memory.put(msg)
    st.success("Loading complete!")
    return vector_memory
    
def create_new_chat_memory(mem_config):
    st.text(f"Creating new vector database for chat messages")
    docker_command = ["sudo", "docker", "run", "-d", "--name", "FMSDemo_chat_db", "-p", f"{CHAT_DB_PORT}:6333"] + mem_config + ["qdrant/qdrant"]
    chat_db_client = start_vector_db(docker_command)
    #just like create_new_doc_index, the collection name is a nonfactor when creating from scratch
    chat_vector_store = QdrantVectorStore(client=chat_db_client, collection_name="llamaindex_chat_store")
    vector_memory = VectorMemory.from_defaults(vector_store=chat_vector_store, 
                                               retriever_kwargs={"similarity_top_k": 2})
    chat_db_client.create_collection(
        collection_name="llamaindex_chat_store",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    for msg in RELEVANT_DEMO_HISTORY:
        vector_memory.put(msg)
    st.success("Creation complete!")
    return vector_memory

def load_doc_index_from_volume(mem_config, doc_db_volume, doc_store_collection):
    st.text(f"Loading the data in {doc_db_volume} into a database container")
    docker_command = ["sudo", "docker", "run", "-d", "--name", "FMSDemo_doc_db", "-p", f"{DOC_DB_PORT}:6333", "-v", f"{doc_db_volume}:/qdrant"] + mem_config + ["qdrant/qdrant"]
    doc_db_client = start_vector_db(docker_command)
    validate_collection(doc_db_client, doc_store_collection)
    doc_vector_store = QdrantVectorStore(client=doc_db_client, collection_name=doc_store_collection)
    st.success("Loading complete!")
    return VectorStoreIndex.from_vector_store(doc_vector_store)

#returns a tuple of the created doc index and ingestion time to create it
def create_new_doc_index(mem_config, doc_data_dir):
    st.text("Creating new vector database to ingest documents into")
    docker_command = ["sudo", "docker", "run", "--name", "FMSDemo_doc_db", "-p", f"{DOC_DB_PORT}:6333", "-d"] + mem_config + ["qdrant/qdrant"]
    doc_db_client = start_vector_db(docker_command)
    #when creating the collection from scratch, just give it a random name. It won't affect the index creation
    doc_vector_store = QdrantVectorStore(client=doc_db_client, collection_name="llamaindex_doc_store")
    storage_context = StorageContext.from_defaults(vector_store=doc_vector_store)
    with st.spinner("Loading documents"):
        documents = SimpleDirectoryReader(doc_data_dir).load_data()

    start_time = time.perf_counter()
    print(f"Ingestion starting time: {start_time}")
    with st.spinner("Ingesting documents into database and creating index"):
        doc_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

    end_time = time.perf_counter()
    print(f"Ingestion ending time: {end_time}")
    ingestion_time = end_time - start_time
    st.success(f"Ingestion complete! Total ingestion time: {ingestion_time:.6f} seconds")
    return doc_index, ingestion_time

#function that deals with all the logic of deciding how to set up the doc and chat dbs based
#on the options entered by the user (ie preload vs create from scratch for each)
def set_up_databases(doc_data_dir, doc_db_volume, doc_store_collection, 
                     chat_db_volume, chat_store_collection, 
                     mem_config):
 
    st.session_state.ingestion_time = None
    with st.status(":orange[**Setting up document vector database**]", expanded=True) as status:
        if doc_db_volume:
            st.session_state.doc_index = load_doc_index_from_volume(mem_config, doc_db_volume, doc_store_collection)
        else:
            st.session_state.doc_index, st.session_state.ingestion_time = create_new_doc_index(mem_config, doc_data_dir) 
    time.sleep(2)
    status.update(label=":green[Set up document vector database]", expanded=False, state="complete")
    
    with st.status(":orange[**Setting up chat vector database**]", expanded=True) as status:
        if chat_db_volume:
            past_chat_memory = load_chat_memory_from_volume(mem_config, chat_db_volume, chat_store_collection)
        else:
            past_chat_memory = create_new_chat_memory(mem_config)
    time.sleep(2)
    status.update(label=":green[Set up chat vector database]", expanded=False, state="complete")

    chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1000)
    st.session_state.pipeline_memory = SimpleComposableMemory.from_defaults(
        primary_memory=chat_memory_buffer,
        secondary_memory_sources=[past_chat_memory]
    )
    print("Doc index and composable memory object saved to session state")

@st.dialog("Save benchmark results")
def write_benchmarks():
    benchmark_dir = st.text_input("Directory to write results to", "../benchmarks/")
    if not os.path.isdir(benchmark_dir):
        print(f"{benchmark_dir} doesn't exist. Creating a new directory at that path")
        try:
            subprocess.run(["mkdir", benchmark_dir], check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Error: mkdir failed to create a directory at {benchmark_dir}")
    if benchmark_dir[-1] != "/":
        benchmark_dir += "/"

    if st.button("Save results"):
        for label, df in st.session_state.benchmark_dfs.items():
            csv_path = benchmark_dir + label.replace(" ", "_") + ".csv"
            print(f"Writing to {csv_path}")
            df.to_csv(csv_path)
        st.session_state.write_benchmarks = False
        st.rerun()

def set_up_write_benchmarks():
    st.session_state.write_benchmarks = True

#run the chat interface as a fragment to prevent restarting and rerunning the script every time
#the user updates the component by inputting a query
@st.fragment
def run_queries(doc_index, pipeline_memory, model, task_label): 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #while benchmark_dfs holds csvs for individual demo configurations (so they can be compared later),
    #benchmark_df_rows holds the rows that will eventually be combined into one dataframe for one configuration
    if "benchmark_df_rows" not in st.session_state:
        st.session_state.benchmark_df_rows = []
    
    chat_window = st.container(border=True)
    chat_input_container = st.container() #create containers to order elements differently than how they're declared in the script
    #display the previous messages from the current session
    for message in st.session_state.messages:
        chat_window.chat_message(message["role"]).write(message["content"])

    if st.button("Stop demo and show data"): #only give the user the option to input a query if the demo is running (ie they did not press the stop butotn last run)
        reset_demo()
        if st.session_state.benchmark_df_rows: #create a new dataframe for the data with this demo config's queries
            benchmark_df = pd.concat(st.session_state.benchmark_df_rows, ignore_index=True)
            #create an ingestion time column (of the right length to match the df's rows) for the query runs of this configuration
            #ingestion only happens once, so all queries under the given demo configuration will be listed with the same time
            ingestion_time_col = []
            for i in range(len(benchmark_df)):
                if st.session_state.ingestion_time:
                    ingestion_time_col.append(st.session_state.ingestion_time)
                else:
                    ingestion_time_col.append(pd.NA)
            benchmark_df.insert(loc=0, column="ingestion time", value=ingestion_time_col)
            st.session_state.benchmark_dfs[task_label] = benchmark_df
            st.session_state.benchmark_df_rows.clear() #after merging the data for each query into one df, clear the session state so the demo can be rerun
        st.session_state.messages.clear() #clear message history in case demo is rerun with a new configuration

        st.markdown("#### Pipeline data")
        for label, df in st.session_state.benchmark_dfs.items(): #display this demo config's dataframe, and data from any other configs in this session
            print(f"Dataframe: {label} \n {df}")
            st.text(label)
            st.dataframe(df, hide_index=True)
        st.button("Save benchmark data to disk", on_click=set_up_write_benchmarks)
        #even when this nested button is clicked, the outer if statement evaluates to false when the fragment reruns since
        #the outer button wasn't pressed. The work around is to change the session state so the outer scope's elif can catch that the
        #button was pressed 
        st.markdown("#### Change any of the data and model or database settings above to rerun the demo with new settings.")
    elif st.session_state.write_benchmarks:
        write_benchmarks()
    else: 
        if query := chat_input_container.chat_input("Enter query"):
            chat_window.chat_message("user").write(query) #display current query by user and save to session history for future reruns
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.spinner("Thinking"):
                response, benchmark_df_row = build_and_run_pipeline(query, doc_index, pipeline_memory, model)
            chat_window.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            print(f"Benchmark data from the pipeline run of this query: {benchmark_df_row}")
            st.text("Data from the last query:")
            st.dataframe(benchmark_df_row, hide_index=True) 
            st.session_state.benchmark_df_rows.append(benchmark_df_row)

if __name__ == "__main__":
    reset_demo() #make sure no containers were left hanging before this rerun
    st.title("LlamaIndex RAG pipeline demo :llama:")
    check_for_ollama_container()

    if "created_containers" not in st.session_state:
        st.session_state.created_containers = [] #keep track of what docker containers are made so they can be removed as needed (either for error cleanup or to restart databases)
    if "created_subprocesses" not in st.session_state:
        st.session_state.created_subprocesses = [] #keep track of what subprocesses are made for similar reasons
    if "benchmark_dfs" not in st.session_state:
        st.session_state.benchmark_dfs = {}
    if "write_benchmarks" not in st.session_state:
        st.session_state.write_benchmarks = False

    with st.expander("**Data and model set up**", True):
        model = st.radio("Model", ["llama2", "mistral:7b"])
        doc_data_dir = None
        doc_db_volume = None
        doc_store_collection = None
        load_doc_db = st.toggle("Load document vector db from a docker volume instead of creating a new one")
        if load_doc_db:
            doc_db_volume = st.text_input("Document database volume name :red[*]")
            doc_store_collection = st.text_input("Document vector store collection name :red[*]")
        else:
            doc_data_dir = st.text_input("Data directory :red[*]", "../data/")
            if not os.path.isdir(doc_data_dir):
                raise ValueError("Invalid data directory. Please specify an existing directory which contains the documents to ingest. Absolute paths are needed for directories outside FMSDemo/scripts/")

        chat_db_volume = None
        chat_store_collection = None
        load_chat_db = st.toggle("Load chat history vector db from a docker volume instead of creating a new one")
        if load_chat_db:
            chat_db_volume = st.text_input("Chat database volume name :red[*]")
            chat_store_collection = st.text_input("Chat vector store collection name :red[*]")

        task_label = st.text_input("Task label (used to label the benchmark results) :red[*]")
        task_label = task_label.strip()
        if re.match(r'^(?=.*\w)[\w\s]+$', task_label) is None: #this will only match the entire string (if it's valid) or nothing
            raise ValueError("Please enter a valid task label (at least one letter or digit, no special characters)")
        elif task_label in st.session_state.benchmark_dfs.keys():
            st.markdown(f":orange[Warning: the label {task_label} was used previously. The data attached to it will be overwritten]")

    with st.expander("**Database setup**", True):
        bound_cpus = st.text_input("CPU(s) to bind databases to")
        bound_nodes = st.text_input("NUMA node(s) to bind databases to")
        mem_config = [] #this list's memory settings will eventually be passed to docker run
        if bound_cpus:
            mem_config += ["--cpuset-cpus", bound_cpus]
        if bound_nodes:
            mem_config += ["--cpuset-mems", bound_nodes]

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")     
    if st.button("Launch demo"): 
        set_up_databases(
            doc_data_dir, doc_db_volume, doc_store_collection,
            chat_db_volume, chat_store_collection,
            mem_config
        )

        print(f"Starting {model} model in Ollama container")
        try:
            subprocess.run(["sudo", "docker", "exec", "-d", "ollama", "ollama", "run", model], check=True)
        except subprocess.CalledProcessError:
            print("Error: docker exec failed to start the model in the Ollama container") 
            reset_demo()

        run_queries(st.session_state.doc_index, st.session_state.pipeline_memory, model, task_label)