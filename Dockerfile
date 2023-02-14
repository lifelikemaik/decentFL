FROM ubuntu:20.04

# FROM python:3.8.16-bullseye

# Download latest listing of available packages:
RUN apt-get -y update
# Upgrade already installed packages:


RUN apt-get install -y git rabbitmq-server iproute2 python3 python3-pip python-dev build-essential wget

RUN git clone https://github.com/lifelikemaik/decentFL

# install requirements.txt
RUN pip3 install -r decentFL/advanced_pytorch_para/requirements.txt

# RUN apt-get -y upgrade urllib3 requests

ADD initRabbit.sh /initRabbit.sh
# Copy files from Discovery
ADD discovery.py /discovery.py
# systemd doesnt exist in container
# ADD discovery.service /discovery.service
RUN chmod +x /initRabbit.sh



EXPOSE 5007
EXPOSE 15672

CMD ["/initRabbit.sh"]

CMD [ "python3", "./discovery.py"]

