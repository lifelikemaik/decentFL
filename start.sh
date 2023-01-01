#!/bin/bash

apt-get update
# apt-get upgrade -y 
# dauert 5 min + interactive shell von sshd

apt-get -y install python3-pip

pip3 install poetry
