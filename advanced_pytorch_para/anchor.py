import signal
import sys
import server2
import client2
import zmq
import json
import re
import os
import time
import argparse
import flwr
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchvision.datasets
import torch
import utils
# from flwr import NDArray, NDArrays, Parameters

os.environ["WANDB_API_KEY"] = "a2d90cdeb8de7e5e4f8baf1702119bcfee78d1ee"

class CurrentParameters:
    def __init__(self, current = flwr.common.Parameters([], "")):
         self._current = current
      
    # getter method
    def get_currentParameter(self):
        return self._current
      
    # setter method
    def set_currentParameter(self, x):
        self._current = x



class CifarClient(flwr.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def get_parameters(self, config):
        print("ALARM alarm!")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = utils.load_efficientnet(classes=10)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        results = utils.train(model, trainLoader, valLoader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        print(len(parameters))
        # print(fl.common.ndarrays_to_parameters(parameters))
        
        model = self.set_parameters(parameters)
        
        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = utils.test(model, testloader, steps, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}



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
    # time.sleep(120)
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

def clientConnectNewServer():
    print("Client stays client")


def getVMlist():
    # read test.json file
    f = open("/Users/maik/Dev/decentralizedfl/decentFL/leader/test.json", "r")
    # f = open("/root/test.json", "r")
    f = f.read().replace('"', "").replace("'", '"')
    vm_json = json.dumps(json.loads(f), sort_keys=True)
    return json.loads(vm_json)


if __name__ == "__main__":
    # vm_load = getVMlist()
    # print(vm_load[str(len(vm_load) - 1)])  # get info of last node in list
    # print(vm_load)
    # print(vm_load[str(len(vm_load) - 1)][0])

    # if(re.findall(r"\d+", os.uname()[1])[0] == str(len(vm_load) - 1)):
    #     server.main(4)
    # else:
    #     time.sleep(180)
    #     ipServer = vm_load[str(len(vm_load) - 1)][0]
    #     client.main(ipServer)

    currentLeader = ""
    # connectToAllNodes(vm_load)

    current = CurrentParameters()
    model = utils.load_efficientnet(classes=10)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initi = flwr.common.ndarrays_to_parameters(model_parameters)

    current.set_currentParameter(initi) # atomic?
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--toy",
    #     type=bool,
    #     default=True,
    #     required=False,
    #     help="Set to true to use only 10 datasamples for validation. \
    #         Useful for testing purposes. Default: False",
    # )
    # args = parser.parse_args()
    # strategy = flwr.server.strategy.FedAvg(
    #     fraction_fit=0.5,
    #     fraction_evaluate=0.5,
    #     min_fit_clients=2,
    #     min_evaluate_clients=5,
    #     min_available_clients=9,
    #     evaluate_fn=server2.get_evaluate_fn(model, args.toy),
    #     on_fit_config_fn=server2.fit_config,
    #     on_evaluate_config_fn=server2.evaluate_config,
    #     initial_parameters=current.get_currentParameter(),
    # )
    # flwr.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     config=flwr.server.ServerConfig(num_rounds=1),
    #     strategy=strategy,
    # )
    # nach jeder Runde checken ob neue Participants

    ipaddress = localhost

    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    device = torch.device("cpu")

    specialClient = CifarClient(trainset, testset, device)
    CifarClient.set_parameters(CurrentParameters.get_currentParameter())

    flwr.client.start_numpy_client(server_address=ipaddress + ":8080", client=specialClient)










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
