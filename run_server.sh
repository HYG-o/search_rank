#!/bin/bash

SCRIPT_NAME=server.py

echo "run file: " $SCRIPT_NAME
LOG_NAME=${SCRIPT_NAME}".log"

ps -ef | grep $SCRIPT_NAME | grep -v grep | awk '{print $2}' | xargs kill -9

nohup python $SCRIPT_NAME &

#nohup python $SCRIPT_NAME > $LOG_NAME 2>&1&
#tail -f $LOG_NAME
