import os
import json
from pathlib import Path
import requests


MODELS = dict()
def create_folder(output_folder):
    """Create model folder."""
    output_path = Path(output_folder)
    if output_path.exists():
        print(f"{output_path} already exists.")
    else:
        output_path.mkdir(parents=True, exist_ok=True)

def list_models(return_list=False):
    """List models and their info."""
    global MODELS
    print("Models:", len(MODELS))
    if return_list:
        return list(MODELS.keys())
    else:
        for k in MODELS:
            v = MODELS[k]
            print(f"- {k}:\n{v['info']}")

def download_models(models=[], output=".", force=False):
    """Download models."""
    global MODELS
    print(f"Downloading {len(models)} models.")
    if isinstance(models, str): models = [models]
    for i, model in enumerate(models):
        print(f"\t{i} - downloading '{model}' ...", end=" ")
        if force or not (Path(output)/model).exists():
            MODELS[model]["download"](output)
            print("ok!")
        else:
            print("skipped")

def add_model(func):
    """Register a model."""
    global MODELS
    model_name = func.__name__
    MODELS[model_name] = {
        "download" : func,
        "name": model_name,
        "info": func.__doc__
    }
    return func

def __download(src, dst):
    r = requests.get(src, allow_redirects=True)
    with open(dst, "wb") as f:
        f.write(r.content)
    return dst

@add_model
def vqgan_imagenet_f16_1024(output_folder):
    """vqgan_imagenet_f16_1024.
    src yaml: 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
    src ckpt: 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
    """
    filename = "vqgan_imagenet_f16_1024"
    yaml_file = 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
    ckpt_file = 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))


@add_model
def vqgan_imagenet_f16_16384(output_folder):
    """vqgan_imagenet_f16_16384.
    src yaml: 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
    src ckpt: 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
    """
    filename = "vqgan_imagenet_f16_16384"
    yaml_file = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
    ckpt_file = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))

@add_model
def coco(output_folder):
    """coco.
    src yaml: 'https://dl.nmkd.de/ai/clip/coco/coco.yaml'
    src ckpt: 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt'
    """
    filename = "coco"
    yaml_file = 'https://dl.nmkd.de/ai/clip/coco/coco.yaml'
    ckpt_file = 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))

@add_model
def faceshq(output_folder):
    """faceshq.
    src yaml: 'https://app.koofr.net/links/a04deec9-0c59-4673-8b37-3d696fe63a5d?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fconfigs%2F2020-11-13T21-41-45-project.yaml'
    src ckpt: 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt'
    """
    filename = "faceshq"
    yaml_file = 'https://app.koofr.net/links/a04deec9-0c59-4673-8b37-3d696fe63a5d?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fconfigs%2F2020-11-13T21-41-45-project.yaml'
    ckpt_file = 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))

@add_model
def faceshq(output_folder):
    """faceshq.
    src yaml: 'https://app.koofr.net/links/a04deec9-0c59-4673-8b37-3d696fe63a5d?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fconfigs%2F2020-11-13T21-41-45-project.yaml'
    src ckpt: 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt'
    """
    filename = "faceshq"
    yaml_file = 'https://app.koofr.net/links/a04deec9-0c59-4673-8b37-3d696fe63a5d?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fconfigs%2F2020-11-13T21-41-45-project.yaml'
    ckpt_file = 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))

@add_model
def wikiart(output_folder):
    """wikiart.
    src yaml: 'http://dl.nmkd.de/ai/clip/wikiart-vqgan/WikiArt_augmented_Steps_7mil_finetuned_1mil.yaml'
    src ckpt: 'http://dl.nmkd.de/ai/clip/wikiart-vqgan/WikiArt_augmented_Steps_7mil_finetuned_1mil.ckpt'
    """
    filename = "wikiart"
    yaml_file = 'http://dl.nmkd.de/ai/clip/wikiart-vqgan/WikiArt_augmented_Steps_7mil_finetuned_1mil.yaml'
    ckpt_file = 'http://dl.nmkd.de/ai/clip/wikiart-vqgan/WikiArt_augmented_Steps_7mil_finetuned_1mil.ckpt'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))

@add_model
def sflckr(output_folder):
    """sflckr.
    src yaml: 'http://dl.nmkd.de/ai/clip/wikiart-vqgan/WikiArt_augmented_Steps_7mil_finetuned_1mil.yaml'
    src ckpt: 'http://dl.nmkd.de/ai/clip/wikiart-vqgan/WikiArt_augmented_Steps_7mil_finetuned_1mil.ckpt'
    """
    filename = "sflckr"
    yaml_file = 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1'
    ckpt_file = 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'
    output_yaml_file = Path(output_folder)/ f"{filename}.yaml"
    output_ckpt_file = Path(output_folder)/ f"{filename}.ckpt"
    os.makedirs(Path(output_folder), exist_ok=True)
    return (__download(yaml_file, output_yaml_file), __download(ckpt_file, output_ckpt_file))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download models.")
    ap.add_argument('-o', '--output', type=str,
                            help="Output folder.", default=".")
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument('-m', '--models', nargs="+", type=str, choices=list(MODELS.keys()))
    gp.add_argument('--ls', dest='list_models', action='store_true',
                            help="List available models.")
    gp.add_argument('--all', dest='download_all', action='store_true',
                            help="Download all models.")
    args = vars(ap.parse_args())
    
    output = args["output"]
    # List Models
    if args["list_models"]:
        list_models()
    # Download Models
    elif args["download_all"]:
        models = list(MODELS.keys())
        download_models(models, output)
    elif len(args["models"]) > 0:
        models = args["models"]
        download_models(models, output)

