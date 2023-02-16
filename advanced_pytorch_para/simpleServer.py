from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from typing import Optional
import numpy as np


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ) -> Optional[fl.common.Parameters]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {rnd} weights...")
            np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = SaveModelStrategy(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
