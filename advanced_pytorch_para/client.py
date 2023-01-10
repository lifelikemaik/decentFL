import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
import wandb
import os

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
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
        model = self.set_parameters(parameters)
        wandb.watch(model)
        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = utils.test(model, testloader, steps, self.device)
        wandb.log({"loss": loss})
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cpu"):
    """Weak tests to check whether all client methods are working as
    expected."""

    model = utils.load_efficientnet(classes=10)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main(ipaddress) -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--ip",
    #     type=str,
    #     default="172.24.33.61",
    #     required=False,
    #     help="Get IP from server",
    # )
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=True,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

    args = parser.parse_args()
    # print(vars(args))

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
    wandb.run.name = str(args.partition) + "_client"

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        trainset, testset = utils.load_partition(args.partition)

        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        client = CifarClient(trainset, testset, device)

        fl.client.start_numpy_client(server_address=ipaddress + ":8080", client=client)


if __name__ == "__main__":
    main("localhost")
