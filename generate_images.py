# -*- coding: utf-8 -*-
# Licensed under the MIT License

# Copyright (c) 2021 Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
VQGAN+CLIP
Original file is located at
    https://colab.research.google.com/drive/17dZl__DaU2kYelWwbtygn-jSeepzO0Bw

Simplified version is located at
    https://colab.research.google.com/drive/17dZl__DaU2kYelWwbtygn-jSeepzO0Bw?usp=sharing

**Download pre-trained models:**
get_models.py (--list | --all | -d (<model name>)+)
"""

from extra import *
from subprocess import Popen, PIPE

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return ReplaceGrad.apply(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * ReplaceGrad.apply(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),)
        self.noise_fac = 0.1


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch



def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def synth(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)


def create_video(last_frame, experiment_folder, seconds=15):
    """Generar video del proceso resultante"""
    init_frame = 1 #Este es el frame donde el vídeo empezará
    #last_frame = i #Puedes cambiar i a el número del último frame que quieres generar. It will raise an error if that number of frames does not exist.
    experiment_folder = Path(experiment_folder)
    min_fps = 10
    max_fps = 30

    total_frames = last_frame-init_frame

    length = seconds #Tiempo deseado del vídeo en segundos

    frames = []
    for i in tqdm(range(init_frame,last_frame), total=total_frames, desc="Procesando frames"): #
        frames.append(Image.open(experiment_folder / "steps" / f"{str(i).zfill(3)}.png"))
    #fps = last_frame/10
    fps = np.clip(total_frames/length,min_fps,max_fps)
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r',\
        str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt',\
        'yuv420p', '-crf', '17', '-preset', 'veryslow', experiment_folder / 'video.mp4'],\
        stdin=PIPE)
    for im in tqdm(frames, desc="Asignando frames"):
        im.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()

def to_experiment_name(prompts):
    return " ".join(prompts).strip().replace(".", " ")\
              .replace("-", " ").replace(",", " ")\
              .replace(" ", "_")

def generate_images(
        prompts, model, outputs_folder, models_folder, iterations=300, image_prompts=[], 
        noise_prompt_seeds=[], noise_prompt_weights=[], size=[300, 300],
        init_image=None, init_weight=0., clip_model='ViT-B/32', 
        step_size=0.1, cutn=64, cut_pow=1., display_freq=5, seed=None,
        overwrite=False
    ):
    model_name = model
    experiment_name = to_experiment_name(prompts)
    experiment_folder = Path(outputs_folder) / experiment_name
    os.makedirs(experiment_folder, exist_ok=overwrite)
    os.makedirs(experiment_folder / "steps", exist_ok=overwrite)

    vqgan_config = Path(models_folder)/f'{model_name}.yaml'
    vqgan_checkpoint = Path(models_folder)/f'{model_name}.ckpt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = size[0] // f, size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if seed is not None:
        torch.manual_seed(seed)

    if init_image:
        pil_image = Image.open(init_image).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    @torch.no_grad()
    def checkin(z, i, losses, iterations, experiment_folder):
        out = synth(z, model)
        TF.to_pil_image(out[0].cpu()).save(experiment_folder / 'progress.png')

    def ascend_txt(pMs, z, i, init_weight, experiment_folder):
        out = synth(z, model)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
        result = []
        if init_weight:
            result.append(F.mse_loss(z, z_orig) * init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(experiment_folder / 'steps'/  f"{str(i).zfill(3)}.png", np.array(img))
        return result

    def train(pMs, z, i, init_weight, display_freq, iterations, experiment_folder):
        opt.zero_grad()
        lossAll = ascend_txt(pMs, z, i, init_weight, experiment_folder)
        if i % display_freq == 0: checkin(z, i, lossAll, iterations, experiment_folder)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))
    try:
        for i in tqdm(range(iterations), total=iterations, desc="Training"):
            train(pMs, z, i, init_weight, display_freq, iterations, experiment_folder)
    except KeyboardInterrupt:
        print("Aborted")
    return i

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate Images with VQGAN+CLIP.")
    ap.add_argument("-m", "--model", default="vqgan_imagenet_f16_1024", type=str,
                    choices=['vqgan_imagenet_f16_1024', 'vqgan_imagenet_f16_16384', 'coco', 'faceshq', 'wikiart', 'sflckr'],
                    help="Pretrained model. Check get_models.py.")
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument("-i", "--prompts", nargs="+", type=str, 
                    help="Input text.")
    gp.add_argument('--it', dest='interactive', action='store_true',
                    help="Use prompt from user input.")
    ap.add_argument("-o", "--outputs_folder", required=False, type=str,
                    help="Outputs folder.", default="/tf/outputs")
    ap.add_argument("-M", "--models_folder", required=False, type=str,
                    help="Models folder.", default="/tf/models")
    ap.add_argument("--iterations", required=False, type=int, 
                    help="Number of iterations.", default=200)
    ap.add_argument("--image_prompts", nargs="*", type=str, 
                    help="Input images.", default=[])
    ap.add_argument("--noise_prompt_seeds", nargs="*", type=int, 
                    help="Noise prompt seeds.", default=[])
    ap.add_argument("--noise_prompt_weights", nargs="*", type=float, 
                    help="Noise prompt weights.", default=[])
    ap.add_argument("--size", nargs=2, type=int, 
                    help="Resulting image size.", default=[300, 300])
    ap.add_argument("--init_image", type=str, 
                    help="Input initial image.", default=None)
    ap.add_argument("--init_weight", type=float, 
                    help="Input initial weight.", default=0.)
    ap.add_argument("--clip_model", type=str, 
                    help="CLIP model.", default='ViT-B/32')
    ap.add_argument("--step_size", type=float, 
                    help="Step size.", default=0.1)
    ap.add_argument("--cutn", type=int, 
                    help="cutn.", default=64)
    ap.add_argument("--cut_pow", type=float, 
                    help="cut_pow.", default=1.)
    ap.add_argument("--display_freq", type=int, 
                    help="Display frequency.", default=5)
    ap.add_argument("--seed", type=int, 
                    help="Seed.", default=None)                
    ap.add_argument('--overwrite', dest='overwrite', action='store_true')
    ap.add_argument('--save_video', dest='save_video', action='store_true',
                    help="Save video.")
    ap.add_argument('--video_seconds', type=int, 
                    help="Lenght of video", default=15)    
    kwargs = vars(ap.parse_args())
    # Options
    if kwargs["interactive"]:
        prompts = []
        while True:
            txt = input("Add prompt or press enter to continue: ")
            if not txt: break
            prompts.append(txt)
        if len(prompts) == 0: raise ValueError("Empty prompts.")
        kwargs["prompts"] = prompts
    del kwargs["interactive"]
    # Save video parameters
    save_video = kwargs["save_video"]
    video_seconds = kwargs["video_seconds"]
    del kwargs["save_video"]
    del kwargs["video_seconds"]
    # Run GAN
    last_frame_index = generate_images(**kwargs)
    # Save Video
    if save_video:
        outputs_folder = kwargs["outputs_folder"]
        prompts = kwargs["prompts"]
        experiment_name = to_experiment_name(prompts)
        experiment_folder = Path(outputs_folder) / experiment_name
        create_video(last_frame_index, Path(experiment_folder), video_seconds)
