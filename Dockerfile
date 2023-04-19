FROM rabbitmq:3.11.9-management

RUN apt-get -y update

RUN apt-get install -y git iproute2 nano python3 python3-pip python-dev build-essential wget

RUN git clone https://github.com/lifelikemaik/decentFL
RUN pip3 install -r decentFL/advanced_pytorch_para/requirements.txt

ADD initRabbit.sh /initRabbit.sh
ADD discovery.py /discovery.py
RUN chmod +x /initRabbit.sh

RUN rabbitmq-plugins enable --offline rabbitmq_mqtt

EXPOSE 5007

# # RUN /initRabbit.sh

# CMD [ "python3", "./discovery.py"]
# RUN python3 discovery.py &
CMD ["rabbitmq-server"]

