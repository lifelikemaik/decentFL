#!/bin/bash

docker exec -i container1 python3 discovery.py &
docker exec -i container2 python3 discovery.py & 
docker exec -i container3 python3 discovery.py & 