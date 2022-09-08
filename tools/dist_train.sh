#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python  \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
