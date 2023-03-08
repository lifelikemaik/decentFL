from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import glob
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    # self = SaveModelStrategy
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )
            print("para")
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), f"newModel_round_{server_round}.pth")
            # for index, x in enumerate(aggregated_parameters.tensors):
            #     with open(
            #         f"/home/ubuntu2/Desktop/decentFL/advanced_pytorch_para/bytes/bytesPart_{index}",
            #         "wb",
            #     ) as f:
            #         f.write(x)
            print(f"Saving round {server_round} aggregated_ndarrays...")
        return aggregated_parameters, aggregated_metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = SaveModelStrategy(evaluate_metrics_aggregation_fn=weighted_average)
net = Net().to(DEVICE)
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

# reloadBytes: List[bytes] = []
# for infile in sorted(glob.glob("/bytes/*")):
#     f = open(infile, "rb")
#     reloadBytes.append(f.read())
#     f.close()
# print(reloadBytes)

print(net)
# Net(
#   (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
#   (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )
print("ufff")
# print([val.cpu().numpy() for _, val in net.state_dict().items()])
# array list mit werten

# net.load_state_dict(torch.load("/home/ubuntu2/Desktop/decentFL/newModel_round_2.pth"))

print(net)
# genauso geblieben, ein Glueck

# parametersp = torch.load("/home/ubuntu2/Desktop/decentFL/newModel_round_2.pth")


# parametersp = np.load("/home/ubuntu2/Desktop/decentFL/newModel_round_2.pth")
# params_dict = zip(net.state_dict().keys(), parametersp)
# state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict}) # FEHLER
# net.load_state_dict(state_dict, strict=True)

# strategyini = SaveModelStrategy(
#     evaluate_metrics_aggregation_fn=weighted_average,
#     initial_parameters=[val.cpu().numpy() for _, val in net.state_dict().items()],
# )

# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategyini,
# )
