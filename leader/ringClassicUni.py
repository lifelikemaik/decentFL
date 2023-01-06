import re
import os
import time

# import zmq
import sys
import json
import socket
import collections
from datetime import datetime, timedelta

# Ring Election
# [Chang & Roberts’79]
# unidirecitonal mit Pub Sub
# https://courses.engr.illinois.edu/cs425/fa2021/assets/slides/lect8-leader-election-final.pdf


def leader():
    f = open("/Users/maik/Dev/decentralizedfl/decentFL/leader/test.json", "r")
    f = f.read().replace('"', "").replace("'", '"')
    f = json.loads(f)
    tupleList = sorted(
        f.items(), key=lambda item: socket.inet_aton(item[0])
    )  # sort ip address - OUTPUT TUPLE LIST
    vm_dict = {}
    vm_dict = convertTupleListToDict(tupleList, vm_dict)

    now = datetime.now()
    # check if heartbeat heard in 6 + 1 seconds
    checktime = now - timedelta(seconds=20)
    print(checktime)
    # datetime.strptime(elem[1], format_data)


def convertTupleListToDict(tup, di):
    di = dict(tup)
    return di


def client():
    print("clientelRing")


def triggerElection():
    # read file
    print("")


if __name__ == "__main__":
    leader()
