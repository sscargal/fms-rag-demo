### Installation

*Optional: If you're planning to use Kernel TPP, install Ubuntu 24.04 and Kernel 6.9*

Prerequisites: python3 and pip
Create a new virtual environment using `python3 -m venv`
Activate the VE using `source`
If not already installed, install docker as shown [here](https://docs.docker.com/engine/install/ubuntu/)
Install dependencies with `pip install -r requirements.txt`

### Running the Demo

With the virtual environment activated, `cd` into the scripts directory and run `chmod +x fmsdemo.py`

Run `./fmsdemo.py <args>` to run the actual demo. Use `./fmsdemo.py -h` or `./fmsdemo.py --help` to display a help menu for the arguments

#### A few notes:

- By default, -d is set to micron_FMSDemo/data, which contains a small paper on COVID 19, and -b is set to micon_FMSDemo/benchmarks. 
- Note that LlamaIndex will attempt to index and ingest *everything* located in the directory specified by -d into the vector databases. The data directory passed into -d must exist and should only contain textual data (but text can be in a variety of file types, eg .txt and .pdf)
- Because fmsdemo.py must currently be run within script directory, use absolute paths to specify anything outside the script directory
- If everything works correctly, the demo will automatically clear any containers it's created. However, a bug may prevent this from happening. In this case, any docker containers that cause conflicts with the ones the demo tries to create must be stopped and removed for the script to work

#### Project architecture
There are four files involved in the demo (clear_qdrant_containers.sh is for debugging purposes). fmsdemo.py is the master script that coordinates all the other parts. run_queries.py deals with data ingestion into databases and sets up the pipeline memory. query_pipeline.py builds and runs the actual pipeline. vectordb_manager.sh configures the vector database memory settings.

The pipeline features some prompt chaining and pulls context from three sources: documents that have been ingested, the current session's chat history, and a cumulative chat history including messages from many past sessions. These last sources are the components of the simple composable memory being used: the simple composable memory's primary source is the current chat buffer while its secondary source is a vector store containing a (potentially massive) amount of messages.

The goal is for this to simulate a RAG pipeline that goes beyond enhancing inference with document context. Imagine a cumulative database containing every past message sent by a user (or every user) from every past session. Not only can the model see past messages from its current session, it can also pull from this past message database to answer questions. So, a user could successfully ask something like "Can you summarize the results of our company's quarterly report?" (document context), "You mentioned x earlier. Can you explain that again differently?" (primary memory context), or "I remember you told me y about x yesterday. Could you elaborate more on that?" (secondary memory context).

Documents are ingested into an index connected to one Qdrant docker container. The past chat history vector store is connected to another Qdrant container. The current chat history is managed by LlamaIndex. Ollama runs in its own container. When the demo runs, context is retrieved from the corresponding containers and combined into one final prompt which is then passed to the model running inside the Ollama container.

Benchmark data for various parts of the pipeline is written to the directory specified by -b.

![Image](./FMSDemo_structure.png)

#### Important constants

Several of the files have constants that, if you want to modify, currently have to be changed directly within the source file itself.
- fmsdemo.py
    - `DEMO_DB_CONFIGS` specifies what arguments to pass into the vectordb_manager.sh script. vectordb_manager.sh will start Qdrant databases according to the arguments passed into it. Information on what each argument does can be found by running `./vectordb_manager.sh -h` or in the source code itself. `DEMO_DB_CONFIGS` is a dict. Each value is an array that will be passed into `subprocess.run()` to run vectordb_manager as you would like. Each key is a description or identifier for what the configuration does. Changing the arguments passed to vectordb_manager.sh is ultimately how memory settings for the pipeline are changed 
    - `QUERIES` is a list containing the queries that will be run through the pipeline during each demo configuration 
- run_queries.py
    - `MOCK_CHAT_HISTORY_LEN` is an integer that specifies how many lines of the mock chat history to read into the vector database. The actual content of the mock chat history shouldn't matter; a large vector memory of random messages just needs to be created to demonstrate retrieval of actually relevant information. The source code can be further modified to change what is read into the secondary vector database entirely (it's currently some random dataset from huggingface). If you'd like to continue using the current database, you'll probably want to drastically increase the size of this variable
    - `RELEVANT_DEMO_HISTORY` is a list that will end up containing ChatMessages relevant to the document data that's been indexed. This simulates past conversations relevant to the ingested document data that you want to be retrieved by the pipeline's memory to enhance context
- query_pipeline.py
    - `REWRITE_PROMPT` instructs a model to rewrite the user's query so that it's been enhanced by chat history. This enhanced query is what will be used to retrieve document context from the index
    - `CONTEXT_PROMPT` is the final prompt that prompts the final inference of the pipeline. It will end up containing chat history from the current session, past sessions, and documnet context

#### Example
- Say a system has DRAM on node 0 and CXL on node 1 and a text file containing a book's content in a directory called book_data. We might want to ask the following questions to the pipeline using DRAM only then CXL only:
    - "Can you summarize chapter 2 of the book for me?"
    - "A few days ago, I asked you about one of the characters. Tell me if you prefer them over character Y and explain why."
- Change `DEMO_DB_CONFIGS` to have two entries, one which passes -m 0 and/or -s 0 to vectordb_manager.sh and one which passes -m 1 and/or -s 1.
- Change `RELEVANT_DEMO_HISTORY` to include some messages relating to the book and one referring to a specific character. Ideally, the pipeline's similarity search will retrieve this message (and a few others related to the book) as opposed to the other unrelated messages in the secondary memory.
- Increase `MOCK_CHAT_HISTORY_LEN`
- Change `QUERIES` to have our two questions
- In the scripts directory, run `fmsdemo.py -d book_data`
- The pipeline will now run four times in total: query 1 with Qdrant datbases on DRAM, query 2 with databases on DRAM, query 1 with databases on CXL, query 2 with databases on CXL
    - Note that, currently data must be reingested between each memory configuration. The DRAM databases will be removed and new ones on CXL will be started. The book will have to be reingested into the new CXL databases
        - Currently working on using docker volumes to avoid this reingestion
