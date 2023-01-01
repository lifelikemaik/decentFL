#!/bin/sh

python3 server.py (/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)