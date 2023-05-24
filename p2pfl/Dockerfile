FROM python:3.8
RUN apt-get -y update

RUN apt-get install -y git iproute2 nano build-essential wget

RUN git clone https://github.com/lifelikemaik/federated_learning_p2p
RUN pip3 install -r federated_learning_p2p/requirements.txt

# get anchor, discovery and ping service



EXPOSE 8888 8887

# run / start systemctl services