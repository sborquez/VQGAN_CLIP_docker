# VQGAN_CLIP_docker
Docker for VQGAN+CLIP (z+quantize method)

# A. Run Container
```bash
HOST_MODELS="<models absolute path>"
HOST_OUTPUT="<outputs absolute path>"
HOST_PORT=8888
sudo docker run --name vqgan_clip --gpus all --rm -it\
        -p $HOST_PORT:8888 \
        -v "$HOST_MODELS:/tf/models"\
        -v "$HOST_OUTPUT:/tf/outputs"\
        sborquez/vqgan_clip:latest\
        bash
```

## 1. Download models


```bash
python get_models.py --help
usage: get_models.py [-h] [-o OUTPUT]
        (-m {vqgan_imagenet_f16_1024,vqgan_imagenet_f16_16384,coco,faceshq,wikiart,sflckr} [{vqgan_imagenet_f16_1024,vqgan_imagenet_f16_16384,coco,faceshq,wikiart,sflckr} ...] | --ls | --all)
```
**Examples**
```bash
# List models
python get_models.py --ls
```
```bash
# Download vqgan_imagenet_f16_1024 model
python get_models.py -m vqgan_imagenet_f16_1024
```
```bash
# Download all models
python get_models.py --all
```
## 2. Start Jupyter Server
```bash
./run_jupyter_notebook.sh
```


Go to [localhost:8888](http://localhost:8888) and enter to the `VQGAN_CLIP_docker` folder.

## 3. Run script

```bash
python generate_images.py --help
usage: generate_images.py [-h]
                          [-m {vqgan_imagenet_f16_1024,vqgan_imagenet_f16_16384,coco,faceshq,wikiart,sflckr}]
                          (-i PROMPTS [PROMPTS ...] | --it)
                          [-o OUTPUTS_FOLDER] [-M MODELS_FOLDER]
                          [--iterations ITERATIONS]
                          [--image_prompts [IMAGE_PROMPTS [IMAGE_PROMPTS ...]]]
                          [--noise_prompt_seeds [NOISE_PROMPT_SEEDS [NOISE_PROMPT_SEEDS ...]]]
                          [--noise_prompt_weights [NOISE_PROMPT_WEIGHTS [NOISE_PROMPT_WEIGHTS ...]]]
                          [--size SIZE SIZE] [--init_image INIT_IMAGE]
                          [--init_weight INIT_WEIGHT]
                          [--clip_model CLIP_MODEL] [--step_size STEP_SIZE]
                          [--cutn CUTN] [--cut_pow CUT_POW]
                          [--display_freq DISPLAY_FREQ] [--seed SEED]
                          [--overwrite] [--save_video]
                          [--video_seconds VIDEO_SECONDS]

Generate Images with VQGAN+CLIP.

optional arguments:
  -h, --help            show this help message and exit
  -m {vqgan_imagenet_f16_1024,vqgan_imagenet_f16_16384,coco,faceshq,wikiart,sflckr}, --model {vqgan_imagenet_f16_1024,vqgan_imagenet_f16_16384,coco,faceshq,wikiart,sflckr}
                        Pretrained model. Check get_models.py.
  -i PROMPTS [PROMPTS ...], --prompts PROMPTS [PROMPTS ...]
                        Input text.
  --it                  Use prompt from user input.
  -o OUTPUTS_FOLDER, --outputs_folder OUTPUTS_FOLDER
                        Outputs folder.
  -M MODELS_FOLDER, --models_folder MODELS_FOLDER
                        Models folder.
  --iterations ITERATIONS
                        Number of iterations.
  --image_prompts [IMAGE_PROMPTS [IMAGE_PROMPTS ...]]
                        Input images.
  --noise_prompt_seeds [NOISE_PROMPT_SEEDS [NOISE_PROMPT_SEEDS ...]]
                        Noise prompt seeds.
  --noise_prompt_weights [NOISE_PROMPT_WEIGHTS [NOISE_PROMPT_WEIGHTS ...]]
                        Noise prompt weights.
  --size SIZE SIZE      Resulting image size.
  --init_image INIT_IMAGE
                        Input initial image.
  --init_weight INIT_WEIGHT
                        Input initial weight.
  --clip_model CLIP_MODEL
                        CLIP model.
  --step_size STEP_SIZE
                        Step size.
  --cutn CUTN           cutn.
  --cut_pow CUT_POW     cut_pow.
  --display_freq DISPLAY_FREQ
                        Display frequency.
  --seed SEED           Seed.
  --overwrite
  --save_video
  --video_seconds VIDEO_SECONDS
                        Lenght of video
```


**Examples**
```bash
python generate_images.py --it  --save_video --iterations 120

```

```bash
python generate_images.py -i "made with love"  --save_video --overwrite 
```

```bash
python generate_images.py -i "noche estrellada en el desierto de atacama" -m "vqgan_imagenet_f16_16384" --size 280 650 --save_video
```

## 4. Enqueue run script with `task-spooler`

`task-spooler` is a simple job queue manager. 

```bash
./enqueue_generate_images.sh <generate_images.py arguments>
```

```bash
# Help 
tsp -h
# List tasks
tsp 
# Cat running task
tsp -c 
# Change number of concurrent jobs (default=1)
tsp -N <number of jobs> 
```

**Note:** It will send the task without checking the arguments. The `--it` argument won't work.

# B. Build Locally

Replace `<models absolute path>` and `<outputs absolute path>` with your host paths.

```bash
git clone "https://github.com/sborquez/VQGAN_CLIP_docker"
cd VQGAN_CLIP_docker
sudo docker build sudo docker build . --tag sborquez/vqgan_clip:latest
HOST_MODELS="<models absolute path>"
HOST_OUTPUT="<outputs absolute path>"
HOST_PORT=8888
sudo docker run --name vqgan_clip --gpus all --rm -it -p 8888:8888 \
        -p $HOST_PORT:8888 \
        -v "$(pwd):/tf/src"\
        -v "$HOST_MODELS:/tf/models"\
        -v "$HOST_OUTPUT:/tf/outputs"\
        sborquez/vqgan_clip:latest\
        bash
```