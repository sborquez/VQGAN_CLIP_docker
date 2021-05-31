# Utils
import os
import sys
import math
import warnings
from pathlib import Path
from tqdm import tqdm 
from tqdm.notebook import tqdm as tqdm_notebook
from base64 import b64encode
import imageio
from IPython.core.display import display, HTML
#import wandb as wandb TODO: add Weight and Biases
import datetime
__utils__ = [
    "os", "sys", "Path", "tqdm", "tqdm_notebook", "display", "HTML",
    "datetime", "imageio", "b64encode", "warnings", "math"
]

# DataScience-CPU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
__cpu_all__ = [
    "plt", "sns", "np", "pd", "Image"
]

# DataScience-GPU
try:
    import cupy as cp
    import cudf
    import cuml
    __gpu_all__ = ["cp", "cudf", "cuml"]
except:
    __gpu_all__ = []
    #warnings.warn("GPU skipped!")

# Distributed DataScience
try:
    import dask_cudf
    __dist_all__ = ["dask_cudf"]
except:
    __dist_all__ = []
    #warnings.warn("Dask skipped!")

# VQGAN+CLIP
try:
    from omegaconf import OmegaConf
    sys.path.append('..')
    sys.path.append('../taming-transformers')
    from taming.models import cond_transformer, vqgan
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    from CLIP import clip
    import kornia.augmentation as K
    __vqgan_clip__ = [
        "OmegaConf", "clip", "K", "TF", "F", "transforms",
        "nn", "optim", "torch", "cond_transformer", "vqgan",
    ]
except:
    __vqgan_clip__ = []
    raise OSError("Can't import VQGAN+CLIP requirements.")

def reset_kernel(): 
    os._exit(00)

__all__ = [
    "reset_kernel" 
]
__all__ += __utils__
__all__ += __gpu_all__
__all__ += __dist_all__
__all__ += __vqgan_clip__