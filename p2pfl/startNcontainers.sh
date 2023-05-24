#!/bin/bash

NUM_CONTAINERS=${NUM_CONTAINERS:-$1}

# Start the containers using Docker Compose
docker-compose up -d --scale container=$NUM_CONTAINERS

# Generate and assign dynamic hostnames and container names
for ((i=1; i<=NUM_CONTAINERS; i++)); do
   # Generate unique hostname
   HOSTNAME="mycontainer$i"

   # Assign custom container name
   CONTAINER_NAME="mycontainer_$i"

   # Set the hostname and container name
   docker exec -it $CONTAINER_NAME hostname $HOSTNAME
   docker rename $CONTAINER_NAME $CONTAINER_NAME-$i
done
