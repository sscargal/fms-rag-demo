#!/usr/bin/env python3
import subprocess
import sys
    
#demo-wide constants    
DOC_INDEX_PERSIST_DIR = "doc_index"
DOC_DATA_DIR = "data"
DOC_DB_COLLECTION_NAME = "llamaindex_doc_db"
CHAT_INDEX_PERSIST_DIR = "chat_index"
CHAT_DB_COLLECTION_NAME = "llamaindex_chat_db"

#Create ollama container
if __name__ == "main":
    print("Starting Ollama container and mistral model")
    try:
        subprocess.run(["sudo", "docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", "ollama/ollama"],
                       check=True)
        subprocess.run(["sudo", "docker", "exec", "-d", "ollama", "ollama", "run", "mistral:7b"], check=True)
    except subprocess.CalledProcessError:
        print("Error: docker run failed for the Ollama container") 
        sys.exit()

    #Create qdrant containers 
    print("Starting vector database containers")
    try:
        subprocess.run(["./vectordb_manager.sh", "-n", "1"], check=True)
    except subprocess.CalledProcessError:
        print("Error trying to execute vectordb_manager.sh")
        sys.exit()

    #Ingest data into qdrant containers 
    print("Ingesting data into databases")
    try:
        subprocess.run(["./ingest_data.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error trying to execute ingest_data.py")
        sys.exit()
        
    #DEBUG
    subprocess.run(["./clear_containers.sh"])