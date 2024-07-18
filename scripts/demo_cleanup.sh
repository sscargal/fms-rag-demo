#!/bin/bash

sudo docker stop ollama
sudo docker remove ollama
./clear_qdrant_containers.sh
rm -r doc_index