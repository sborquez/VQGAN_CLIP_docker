FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt update -y
RUN apt install -y ffmpeg
RUN git clone https://github.com/openai/CLIP
RUN git clone https://github.com/CompVis/taming-transformers
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ftfy regex tqdm omegaconf pytorch-lightning 
RUN pip install imageio imageio-ffmpeg pandas seaborn kornia einops
RUN apt install -y task-spooler
RUN git clone https://github.com/sborquez/VQGAN_CLIP_docker.git
RUN chmod u+x VQGAN_CLIP_docker/start_jupyter_notebook.sh
RUN chmod u+x VQGAN_CLIP_docker/enqueue_generate_images.sh
RUN mkdir -p "/root/.cache/torch/hub/checkpoints"
RUN curl "https://download.pytorch.org/models/vgg16-397923af.pth" -o "/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
WORKDIR "/tf/VQGAN_CLIP_docker"
