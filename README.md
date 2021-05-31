# VQGAN_CLIP_docker
Docker for VQGAN+CLIP (z+quantize method)

## Run Container
```bash
HOST_MODELS = ""
HOST_OUTPUT = ""
sudo docker run --name vqgan_clip --gpus all --rm -it -p 8888:8888 \
        -v "$HOST_MODELS:/tf/models"
        -v "$HOST_OUTPUT:/tf/outputs"
        sborquez/vqgan_clip:latest
```

## Build Locally
```bash
sudo docker run --name vqgan_clip --gpus all --rm -it -p 8888:8888 -v "$(pwd):/tf/src" sborquez/vqgan_clip:latest
```