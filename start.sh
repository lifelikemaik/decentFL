#!/bin/bash

apt-get update
apt-get upgrade

apt-get install python3-pip

pip3 install -r requirements.txt