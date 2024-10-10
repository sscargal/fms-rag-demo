# Retrieval Augmented Generation (RAG) Using Compute Express Link (CXL) Memory

This is a demonstration project for a RAG (Retrieval-Augmented Generation) pipeline that combines document context, current session chat history, and cumulative chat history to enhance LLM responses.

## Prerequisites: 

This repository requires the following:

- Docker
- python3
- pip
- Ubuntu 22.04 (minimum), 24.04 running Kernel 6.9 or newer is required for Kernel Weighted Interleaving.

## Installation

1. Install the Python dependencies

```bash
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev
```

2. Create and activate a virtual environment called `llmdemo`:

```bash
python3 -m venv llmdemo
source llmdemo/bin/activate
```

3. Install Docker following the [official documentation](https://docs.docker.com/engine/install/ubuntu/).

4. Install the project dependencies

```bash
pip install -r requirements.txt
```

5. Add your username to the `docker` group

If you run the application as a non-root user, your username must be added to the `docker` group to allow socket communication to the containers.

- **Check if Your User is in the Docker Group:**
Docker requires that the user running Docker commands be part of the docker group. You can check if your user is part of the docker group with this command:
```bash
groups
```

Look for docker in the list of groups. If you don't see it, you’ll need to add your user to the Docker group.

- **Add Your User to the Docker Group:**
If your user is not in the docker group, you can add the user to the group using the following command:

```bash
sudo usermod -aG docker $USER
```

After adding the user to the docker group, you’ll need to log out and back in for the changes to take effect. Alternatively, you can run:

```bash
newgrp docker
```
This will apply the changes to your current session without needing to log out.

## Quick Start

1. Activate the virtual environment:

```bash
source llmdemo/bin/activate
```

2. Start the we UI:

```bash
cd scripts/
streamlit run fmsdemo.py
```

Note: It make take a minute or more to initialize and fully start the UI before it becomes available for use.

3. Refer to the **Examples** section for step-by-step instructions to use the UI to run various demos. 
    - Streamlit will print how to access it to the terminal. If you'd like to only have to ingest your data once while using it multiple times, create a Qdrant container with the argument `-v volume_name:/qdrant`. Then, ingest the data into the container. The container's state will be saved into volume_name, so simply pass that (and the name of the collection created within the container) into the appropriate fields of the UI.

**Notes**

- When specifying CPU and NUMA nodes, use single numbers, hyphens for ranges, or commas for multiple (eg 0 or 7-10 or 3,5)
- Note that LlamaIndex will attempt to ingest *everything* located in the directory specified in the Data directory field. The data directory specified must exist and should only contain textual data (but text can be in a variety of file types, eg .txt and .pdf)
- Where the streamlit command is run affects how paths entered into the demo are interpreted. Eg if the command is run within the scripts/ directory, entering data/ as the data directory in the UI will look for a directory called data/ within scripts/ (i.e a path like $HOME/scripts/data/)
- The demo should automatically clear any containers it's created. However, bugs or certain actions (eg refreshing the page in the middle of the demo) may prevent this from happening. In this case, you'll have to manually remove the containers created by the script (FMSDemo_doc_db and FMSDemo_chat_db)
- The `RELEVANT_DEMO_HISTORY` constant in fmsdemo.py contains a list of LlamaIndex chat messages. The script will always insert these messages into the chat vector database
    - The purpose of these messages is to ensure that the chat database always contains at least some messages relevant to the documents, even if the rest of the database is filler or garbage used as padding to increase the overall database size and stress the system. These messages are what are ideally retrieved by the pipeline as context to enhance the LLM response
    - If the database is being created by the demo from scratch, it will solely consist of these messages 
    - If the database is being loaded from a volume containing a container state, the messages will be added into the loaded vector store
    - If you want to query from your own documents, you may want to change this constant. However, it's not necessary to have relevant messages in the secondary vector store

## Project architecture

The pipeline features some prompt chaining and pulls context from three sources: documents that have been ingested, the current session's chat history, and a cumulative chat history including messages from many past sessions. These last two sources are the components of the simple composable memory being used: the simple composable memory's primary source is the current chat buffer while its secondary source is a vector store containing a (potentially massive) amount of messages.

The goal is for this to simulate a RAG pipeline that goes beyond enhancing inference with document context. Imagine a cumulative database containing every past message sent by a user (or every user) from every past session. Not only can the model see past messages from its current session, it can also pull from this past message database to answer questions. So, a user could successfully ask something like "Can you summarize the results of our company's quarterly report?" (document context); "You mentioned x earlier. Can you explain that again differently?" (primary memory context); or "I remember you told me y about x yesterday. Could you elaborate more on that?" (secondary memory context).

Documents are queried through an index connected to one Qdrant docker container. The past chat history vector store is connected to another Qdrant container. The current chat history is managed by LlamaIndex. Ollama runs in its own container. When the demo runs, context is retrieved from the corresponding containers and combined into one final prompt which is then passed to the model running inside the Ollama container.

![Image](./FMSDemo_structure.png)

## Examples

All these examples assume the virtual environment has already been activated.

### 1. Using the Provided Sample Data (Wikipedia)

The demo repository comes with some sample data and chat messages to use. To use it, do the following in the UI:

1. Navigate to the `scripts` directory
2. Run `streamlit run fmsdemo.py`, and open the UI in a web browser
3. Enter the task label name (eg "demo")
4. Press the `Launch demo` button
- That's it! You can now input queries into the pipeline, for example "Why is the sky blue?". The documents that were ingested by the script are a few Wikipedia articles and a paper on the origins of COVID (you can read the original documents yourself within the `data` directory)
- The terminal contains additional program information. Notably, it contains more verbose output from the RAG pipeline
- When you're done, press the `Stop demo and show data` button. Optionally, you can then press `Save benchmark data to disk` to write measurements about the pipeline to a file

### 2. Preloading Data to Avoid Reingestion

This example demonstrates how to ingest a document dataset once using LlamaIndex. The sample LlamaIndex program and process as a whole can easily be reused to fit your own use case with minimal modificaitons:

1. Navigate to `examples/preload/` and run the `llamaindex_ingest.py` script
```bash
cd examples/preload
chmod +x llamaindex_ingest.py
sudo docker run -d --name wiki_db -p 7000:6333 -v wiki_db_vol:/qdrant qdrant/qdrant
./llamaindex_ingest.py
sudo docker stop wiki_db
```
    - The articles to be ingested are located in the `wiki/` directory
    - LlamaIndex will now create the vector database from the documents passed into it. Because this database is in a Docker container mounted to the wiki_db_vol volume, all the data will be stored in the volume for future use

2. Start the UI:
```bash
cd ../../scripts
streamlit run fmsdemo.py
```

3. Type in any random task label (eg "preload example")
4. Toggle the `Load document vector db from a docker volume` option. Enter "wiki_db_vol" in the `volume name` field and "llamaindex_wiki_store" in the `collection name` field
5. Press `Launch demo`. Whenever you want, you can stop the demo, refresh the browser, or change the settings and relaunch the demo with the same document loading options. Instead of having to reingest the articles between each of these restarts, the demo will load the preloaded database, saving a lot of time

### 3. Using the provided larger chat history

This example shows how to use a provided dataset and script to create a vector store that simulates having a larger chat history than what is created when the second database is spun up from scratch. Similarly to with the document database, it also shows how you can load the chat history database from preexisting data to save time.

1. Navigate to `examples/chat_history/`.
2. Create and populate the chat history database:
```bash
chmod +x create_chat_db_container.py
sudo docker run -d --name chat_db -p 7000:6333 -v chat_db_vol:/qdrant qdrant/qdrant
./create_chat_db_container.py
sudo docker stop chat_db
```
3. Type in any random task label (eg "preload example")
4. Toggle the `Load document vector db from a docker volume` option. Enter "wiki_db_vol" in the `volume name` field and "llamaindex_wiki_store" in the `collection name` field
5. Press `Launch demo`. Whenever you want, you can stop the demo, refresh the browser, or change the settings and relaunch the demo with the same document loading options. Instead of having to reingest the articles between each of these restarts, the demo will load the preloaded database, saving a lot of time
  - Note that, when using the option to load the chat history database from a volume, any new queries sent to the pipeline and the pipeline's response will be persisted into the volume. This is because they are put into the vector memory which is connected to the container which is connected to the volume. That means that, as long as the same volume and collection are loaded, messages from past sessions will carry over and be loaded

### 4. Using the memory configuration options

To compare DRAM and CXL performance, the process is:
- Set up databases as desired (see above)
- Run the demo with CPUs X-Y on NUMA node 0 (DRAM).
- Stop the demo and change settings to NUMA node 2 (CXL).
- Run the demo again and compare results.

1. Set up the document and chat databases as you want (both from scratch, both preloaded, or one from scratch and one preloaded). See previous examples.
2. In the UI, Enter "dram only" as the task label
3. Enter 32-40 in the `CPU(s) to bind databases to` field and 0 in the `NUMA node(s) to bind databases to` field
4. Press `Launch demo` and enter some queries
5. Press `Stop demo and show data`. Then, change the `NUMA node(s) to bind databases to` field to 2 and the task label to "cxl only"
6. Press `Launch demo` and enter your queries
7. Press `Stop demo and show data`. Optionally, save the results to disk

## Troubleshooting

If you encounter errors when installing Python or starting the demo UI with `streamlit run fmsdemo.py`, reinstall Python inside your virtual environment. For example, to reinstall Python 3.11.9:

```bash
pyenv uninstall 3.11.9
pyenv install 3.11.9
```

This should compile Python with all the necessary modules. If you still encounter issues, you may need to install additional dependencies. Use the errors to identify which dependencies are required, then use `apt` to install them.

After installing these dependencies, reinstall Python:

```bash
pyenv uninstall 3.11.9
pyenv install 3.11.9
```

There should be no new errors.

---

Error: "ModuleNotFoundError: No module named '_ctypes'"

Solution: Activate the Python environment using `source llmdemo/bin/activate`