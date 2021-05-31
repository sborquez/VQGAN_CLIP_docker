FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN git clone https://github.com/openai/CLIP
RUN git clone https://github.com/CompVis/taming-transformers
RUN pip install ftfy regex tqdm omegaconf pytorch-lightning
RUN apt install task-spooler
RUN git clone https://github.com/sborquez/VQGAN_CLIP_docker.git
