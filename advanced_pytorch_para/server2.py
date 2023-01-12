from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch
import os
import utils
import wandb
import warnings

# warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = utils.load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 5k training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    valLoader = DataLoader(valset, batch_size=16)
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        print(parameters)
        print(params_dict)
        print(fl.common.ndarrays_to_parameters(parameters))
        
        loss, accuracy = utils.test(model, valLoader)
        wandb.log({"loss": loss})
        wandb.log({"accuracy": accuracy})
        wandb.watch(model)
        return loss, {"accuracy": accuracy}

    return evaluate


def main(rounds):
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    os.environ["WANDB_API_KEY"] = "a2d90cdeb8de7e5e4f8baf1702119bcfee78d1ee"
    configO = {
        "dataset": "CIFAR10",
        "machine": os.uname()[1],
        "model": "CNN",
        "learning_rate": 0.01,
        "batch_size": 128,
    }
    config = {
        "dataset": "CIFAR10",
        "machine": os.uname()[1],
    }
    wandb.init(config=config, project="dfl", entity="lifelikemaik")
    wandb.run.name = os.uname()[1]

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=True,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )

    args = parser.parse_args()

    model = utils.load_efficientnet(classes=10) # inital parameters are necessary, otherwise fail


    wandb.watch(model) # 5 sec run, than new model appears
    # ohne parameter: Requesting initial parameters from one random client
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=5,
        min_available_clients=9,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main(10)
