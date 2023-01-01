#!/bin/sh

# python3 server.py (/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)

poetry install
poetry shell

# python3 server.py 172.24.33.61