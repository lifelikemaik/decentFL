#!/bin/sh

rabbitmq-server -detached
rabbitmqctl start_app

# activate rabbitmq plugins
rabbitmq-plugins enable rabbitmq_mqtt
# create a user
rabbitmqctl add_user test test
# tag the user with "administrator" for full management UI and HTTP API access
rabbitmqctl set_user_tags test administrator