#!/bin/bash

#create dockerfile
cat > fmsdemo.Dockerfile <<EOF
#syntax=docker/dockerfile:1
#ensure latest dockerfile syntax (which supports --mount) is used

#start with Ollama base image
FROM ollama/ollama:latest

#Install Python
RUN --mount=type=cache,target=/var/cache/apt \ #--mount caches installed packages between runs
    apt-get update && apt-get install -y python3 python3-pip

#Install script dependencies
WORKDIR /ollama
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install llama-index-embeddings-huggingface && \
    pip install llama-index-vector-stores-qdrant

#Copy python script into container
COPY ./ingest_data.py
EOF

DOCKER_BUILDKIT=1 docker build
