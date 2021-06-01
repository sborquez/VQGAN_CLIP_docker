#!/bin/bash
HOST_MODELS="$HOME/vqgan/models"
HOST_OUTPUT="$HOME/vqgan/outputs"
HOST_PORT=8888
#HOST_PORT=8081
sudo docker run --name vqgan_clip --gpus all --rm -it -p $HOST_PORT:8888 \
        -v "$(pwd):/tf/src" \
        -v "$HOST_MODELS:/tf/models" \
        -v "$HOST_OUTPUT:/tf/outputs" \
        sborquez/vqgan_clip:latest\
        bash