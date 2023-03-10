#!/bin/sh

# rabbitmq-server start -detached
# sleep 3
# rabbitmqctl start_app

# activate rabbitmq plugins
rabbitmq-plugins enable rabbitmq_mqtt
sleep 5
# create a user
rabbitmqctl add_user test test
sleep 5
# tag the user with "administrator" for full management UI and HTTP API access
rabbitmqctl set_user_tags test administrator