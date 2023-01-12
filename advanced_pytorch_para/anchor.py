import server
import client
import zmq
import json
import re
import os
import time

os.environ["WANDB_API_KEY"] = "a2d90cdeb8de7e5e4f8baf1702119bcfee78d1ee"

def connectToAllNodes():
    # pub / sub pattern with grpc
    print("pain")

    # server.main(4) # start server on port 8080, ENTER round number


def clientToServer():
    print("clientToServer")
    # get parameter to spawn new server
    # notify all server

def serverToClient():
    print("serverToClient")
    print("starting Client again")
    # announce new server from file or algo


if __name__ == "__main__":

    # read test.json file

    f = open("/Users/maik/Dev/decentralizedfl/decentFL/leader/test.json", "r")
    # f = open("/root/test.json", "r")
    f = f.read().replace('"', "").replace("'", '"')
    vm_json = json.dumps(json.loads(f), sort_keys=True)
    vm_load = json.loads(vm_json)
    print(len(vm_json))
    # print(vm_load[str(len(vm_load) - 1)])  # get info of last node in list
    print(vm_load)

    # if(re.findall(r"\d+", os.uname()[1])[0] == str(len(vm_load) - 1)):
    #     server.main(4)
    # else:
    #     time.sleep(180)
    #     ipServer = vm_load[str(len(vm_load) - 1)][0]
    #     client.main(ipServer)

    # for n in range(len(vm_load) - 1, -1, -1):
    #     # last at first .. duh
    #     print(vm_load[str(n)])

    print(vm_load[str(len(vm_load) - 1)][0])
    connectToAllNodes()
