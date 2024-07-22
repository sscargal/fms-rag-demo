#!/usr/bin/env python3
import pandas as pd #user will have to manually source python VE before running script (unless there's a way to programatically do this)
import subprocess
import sys
import signal
import os
import argparse
import time
    
#demo-wide constants    
DEMO_SCRIPTS = ["vectordb_manager.sh", "run_queries.py"]
MOCK_CHAT_HISTORY_PATH = "mock_chat_history.csv"
#this dict's values are lists that contain the command for how the dbs will be set up. Its
#keys are what part of the demo the memory configuration is for/a short description of what the set up is
DEMO_DB_CONFIGS = {"DRAM only": ["./vectordb_manager.sh", "-n", "2"],
                   "DRAM and CXL": ["./vectordb_manager.sh", "-n", "2", "-m", "1,2", "-s", "1,2"]}
DOC_INDEX_PERSIST_DIR = "doc_index"
QUERIES = ["What do the article's authors say about if scientists predicted the COVID pandemic?",
           "I asked you before about what pandemic prevention methods the authors discuss. Could you answer that question again?",
           "What was the first question I asked you today?"]

def signal_handler(sig, frame):
    print("SIGINT sent")
    time.sleep(5)
    demo_cleanup()
    sys.exit(0)

def demo_cleanup():
    print("Demo cleanup called. Removing all containers created throughout the demo's execution")
    for container in created_containers:
        subprocess.run(["sudo", "docker", "stop", container])
        time.sleep(3)
        subprocess.run(["sudo", "docker", "remove", container])
    if os.path.isdir(DOC_INDEX_PERSIST_DIR):
        subprocess.run(["rm", "-r", DOC_INDEX_PERSIST_DIR])

#need a separate function to just clear vector database containers without clearing the Ollama container
def clear_databases():
    print("Clearing vector db containers")
    i = 0
    while i < len(created_containers):
        container = created_containers[i]
        if container != "ollama":
            subprocess.run(["sudo", "docker", "stop", container])
            time.sleep(3)
            subprocess.run(["sudo", "docker", "remove", container])
            created_containers.remove(container)
        else:
            i += 1
    subprocess.run(["rm", "-r", DOC_INDEX_PERSIST_DIR]) #remove persisted index to force reingestion during next configuration

if __name__ == "__main__":
    created_containers = [] #keep track of what docker containers are made so they can be removed as needed (either for error cleanup or to restart databases)
    signal.signal(signal.SIGINT, signal_handler)

    for script in DEMO_SCRIPTS:
        subprocess.run(["chmod", "+x", script])    

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", dest="use_cpu", help="Specify Ollama to use cpu for inferencing (by default, Ollama will attempt to use a gpu)", default=False)
    parser.add_argument("-d", "--data", help="Specify the directory containing the document data to be ingested and indexed", default="../data")
    parser.add_argument("-m", "--model", help="Specify the model to use for inferencing", choices=["mistral:7b", "llama2", "llama3"], default="llama2")
    parser.add_argument("-b", "--benchmark-dir", dest="benchmark_dir", help="Specify the directory to write benchmark data to. If the directory does not exist, it will be created", default="../benchmarks/")
    args = parser.parse_args()

    doc_data_dir = args.data
    if not os.path.isdir(doc_data_dir):
        demo_cleanup()
        raise RuntimeError("Invalid argument for -d. Please specify a valid directory which contains the documents to ingest. Absolute paths are needed for directories outside FMSDemo/scripts/")

    benchmark_dir = args.benchmark_dir
    if not os.path.isdir(benchmark_dir):
        print(f"No directory found at {benchmark_dir}. Creating the directory. If this was not intended, note that absolute paths are needed to specify existing directories outside FMSDemo/scripts/")
        try:
            subprocess.run(["mkdir", benchmark_dir], check=True)
        except subprocess.CalledProcessError:
            print(f"Error: mkdir failed to create a directory at {benchmark_dir}")
            demo_cleanup()
            sys.exit()
    if benchmark_dir[-1] != "/": #benchmark_dir needs to end with a slash because later parts assume it does  
        benchmark_dir += "/"

    model = args.model
    print(f"Starting Ollama container and {model} model")
    try:
        if args.use_cpu:
            subprocess.run(["sudo", "docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", 
                            "ollama/ollama"],
                            check=True)
        else:
            subprocess.run(["sudo", "docker", "run", "-d", "--gpus=all", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", 
                            "ollama/ollama"], check=True)
        subprocess.run(["sudo", "docker", "exec", "-d", "ollama", "ollama", "run", model], check=True)
        created_containers.append("ollama")
    except subprocess.CalledProcessError:
        print("Error: docker run failed for the Ollama container") 
        demo_cleanup()
        sys.exit()

    benchmark_dfs = {}

    #run through each part of the demo
    for desc, args in DEMO_DB_CONFIGS.items():
        print(f"Running demo with {desc}")
        print(f"Starting vector database containers with the following args: {args}")

        try:
            vectordb_manager_result = subprocess.run(args, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            print("Error trying to execute vectordb_manager.sh")
            demo_cleanup()
            sys.exit()

        for line in vectordb_manager_result.stdout.split("\n"):
            print(line)
            words = line.split()
            if words: #script will output an empty line that we can't split and negatively index
                created_containers.append(words[-1])

        print("Running query pipeline using the created databases")
        from run_queries import run_queries
        benchmark_file_prefix = benchmark_dir + desc.replace(" ", "_") #want to create separate benchmark files for each demo config within the benchmark dir
        benchmark_df = run_queries(model, QUERIES, doc_data_dir, benchmark_file_prefix)
           
        print("Resetting for next demo configuration")
        benchmark_dfs[desc] = benchmark_df
        pipeline_df_path = benchmark_file_prefix + "_benchmarks.csv"
        print(f"Writing results to {pipeline_df_path}")
        benchmark_df.to_csv(pipeline_df_path)

        clear_databases()
    
    for key, value in benchmark_dfs.items():
        print(f"{key}: {value}")
        
    print("Ending demo")
    demo_cleanup()
