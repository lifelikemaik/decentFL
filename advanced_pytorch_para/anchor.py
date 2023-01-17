import signal
import sys
import server2
import client2
import zmq
import json
import re
import os
import time
import flwr
# from flwr import NDArray, NDArrays, Parameters

os.environ["WANDB_API_KEY"] = "a2d90cdeb8de7e5e4f8baf1702119bcfee78d1ee"

class CurrentParameters:
    def __init__(self, age = 0):
         self._age = age
      
    # getter method
    def get_currentParameter(self):
        return self._age
      
    # setter method
    def set_currentParameter(self, x):
        self._age = x


def connectToAllNodes(vm_load):
    # pub / sub pattern with grpc macht eher weniger sinn

    # zeroMQ pub / sub

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % 5556)
    # versuchen wir mal einen Port; 5556

    contextsub = zmq.Context()
    socketsub = contextsub.socket(zmq.SUB)
    topicfilter = "LEADER"
    socketsub.setsockopt(zmq.SUBSCRIBE, topicfilter)
    for n in range(len(vm_load) - 1, -1, -1):
        # last at first .. duh
        # print(vm_load[str(n)]) # all info
        # print(vm_load[str(n)][0]) # just ips
        socket.connect("tcp://" + str(vm_load[str(n)][0]) + ":%s" % 5556)

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


def getVMlist():
    # read test.json file
    f = open("/Users/maik/Dev/decentralizedfl/decentFL/leader/test.json", "r")
    # f = open("/root/test.json", "r")
    f = f.read().replace('"', "").replace("'", '"')
    vm_json = json.dumps(json.loads(f), sort_keys=True)
    return json.loads(vm_json)


if __name__ == "__main__":
    vm_load = getVMlist()
    # print(vm_load[str(len(vm_load) - 1)])  # get info of last node in list
    print(vm_load)
    print(vm_load[str(len(vm_load) - 1)][0])

    # if(re.findall(r"\d+", os.uname()[1])[0] == str(len(vm_load) - 1)):
    #     server.main(4)
    # else:
    #     time.sleep(180)
    #     ipServer = vm_load[str(len(vm_load) - 1)][0]
    #     client.main(ipServer)

    currentLeader = ""
    connectToAllNodes(vm_load)

    # while True:
    #     try:
    #         print("TODO")
    #
    #
    #
    #
    #     except KeyboardInterrupt or signal.SIGTERM:
    #         print("program break")
    #         sys.exit(1)
