import re
import os
import time
import signal
import zmq
import sys
import json
import threading
import socket
import collections
from datetime import datetime, timedelta

# 2023-01-04 19:04:13.378536
format_data = "%Y-%m-%d %H:%M:%S.%f"
## macht den algo über id, hostname sollte gehen

# https://courses.engr.illinois.edu/cs425/fa2021/assets/slides/lect8-leader-election-final.pdf


def leader():
    print("you never walk alone")


def client():
    print("clientel")


def get_vm_dict():
    f = open("/Users/maik/Dev/decentralizedfl/decentFL/leader/test.json", "r")
    f = f.read().replace('"', "").replace("'", '"')
    return json.loads(f)


def triggerElection():
    # read file
    # get all hostnames from list, in-memory mapping to file

    now = datetime.now()
    # check if heartbeat heard in 6 + 1 seconds
    checktime = now - timedelta(seconds=20)
    print(checktime)


if __name__ == "__main__":
    print("lol")
