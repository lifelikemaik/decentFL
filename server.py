"""Flower server example."""


import flwr as fl
import sys

ipaddressServer = sys.argv[1]

if __name__ == "__main__":
    fl.server.start_server(
        server_address = ipaddressServer + ":8080",
        config = {"num_rounds": 3},
    )
