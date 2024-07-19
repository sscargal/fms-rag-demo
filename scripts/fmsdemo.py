#!/usr/bin/env python3
import pandas as pd #user will have to manually source python VE before running script (unless there's a way to programatically do this)
import subprocess
import sys
import argparse
    
#demo-wide constants    
DEMO_SCRIPTS = ["vectordb_manager.sh", "run_queries.py", "demo_cleanup.sh", "clear_qdrant_containers.sh"]
CHAT_HISTORY_PATH = "mock_chat_history.csv"
#this dict's values are lists that contain the command for how the dbs will be set up. Its
#keys are what part of the demo the memory configuration is for/a short description of what the set up is
DEMO_DB_CONFIGS = {"DRAM only": ["./vectordb_manager.sh", "-n", "2"],
                   "DRAM and CXL": ["./vectordb_manager.sh", "-n", "2", "-m", "1,2", "-s", "1,2"]}
DOC_INDEX_PERSIST_DIR = "doc_index"
BENCHMARK_DF_PATH = "pipeline_benchmarks.csv"

if __name__ == "__main__":
    for script in DEMO_SCRIPTS:
        subprocess.run(["chmod", "+x", script])    

    parser = argparse.ArgumentParser()
    #NOTE: -d is not fully implemented yet
    parser.add_argument("-c", "--cpu", action="store_true", dest="use_cpu", help="Specify Ollama to use cpu for inferencing (by default, Ollama will attempt to use a gpu)", default=False)
    parser.add_argument("-d", "--data", help="Specify the directory containing the document data to be ingested and indexed")
    parser.add_argument("-m", "--model", help="Specify the model to use for inferencing", choices=["mistral:7b", "llama2", "llama3"], default="llama2")
    args = parser.parse_args()

    model = args.model
    print(f"Starting Ollama container and {model} model")
    try:
        if args.use_cpu:
            subprocess.run(["sudo", "docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", 
                            "--health-cmd=\"ollama list | grep {model}\"", "--health-interval", "30s", "--health-timeout", "3s", "--health-retries", "10", "--health-start-period", "90s",
                            "ollama/ollama"],
                            check=True)
        else:
            subprocess.run(["sudo", "docker", "run", "-d", "--gpus=all", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", 
                            "--health-cmd=\"ollama list | grep {model}\"", "--health-interval", "30s", "--health-timeout", "3s", "--health-retries", "10", "--health-start-period", "90s",
                            "ollama/ollama"], check=True)
        subprocess.run(["sudo", "docker", "exec", "-d", "ollama", "ollama", "run", model], check=True)
    except subprocess.CalledProcessError:
        print("Error: docker run failed for the Ollama container") 
        sys.exit()
        
    benchmark_dfs = {}
        
    #run through each part of the demo
    for desc, args in DEMO_DB_CONFIGS.items():
        print(f"Running demo with {desc}")
        print(f"Starting vector database containers with the following args: {args}")

        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError:
            print("Error trying to execute vectordb_manager.sh")
            sys.exit()

        print("Running query pipeline using the created databases")
        try:
            subprocess.run(["./run_queries.py", model], check=True)
        except subprocess.CalledProcessError:
            print("Error trying to execute run_queries.py")
            sys.exit()
            
        print("Resetting for next demo configuration")
        benchmark_df = pd.read_csv(BENCHMARK_DF_PATH)
        benchmark_dfs[desc] = benchmark_df
        desc = desc.replace(" ", "_")
        benchmark_df.to_csv(f"../benchmarks/{desc}_benchmarks.csv")

        subprocess.run(["./clear_qdrant_containers.sh"])
        subprocess.run(["rm", "-r", DOC_INDEX_PERSIST_DIR])
    
    for key, value in benchmark_dfs.items():
        print(f"{key}: {value}")
        
    print("Ending demo")
    subprocess.run(["./demo_cleanup.sh"])
