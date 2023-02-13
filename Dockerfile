FROM ubuntu:20.04

# FROM python:3.8.16-bullseye

# Download latest listing of available packages:
RUN apt-get -y update
# Upgrade already installed packages:


RUN apt-get install -y git rabbitmq-server python3 python3-pip python-dev python-is-python3 build-essential wget


# install requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get -y upgrade urllib3 requests

ADD initRabbit.sh /initRabbit.sh
RUN chmod +x /initRabbit.sh

EXPOSE 15672

CMD ["/initRabbit.sh"]

# Copy files from Discovery