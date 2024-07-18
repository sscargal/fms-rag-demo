#!/bin/bash

echo "Clearing all qdrant docker containers"
sudo docker ps --format {{.ID}} --filter ancestor=qdrant/qdrant | \
xargs -I {} sudo docker stop {} | xargs -I {} sudo docker remove {}
sudo docker volume remove vectordb_data