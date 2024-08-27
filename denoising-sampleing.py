import imghdr
import os,sys
import math
sys.path.append('denoising-diffusion-pytorch')
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer # type: ignore
from USUtils.USLoader import *
from accelerate import Accelerator # type: ignore
from pathlib import Path
from torchvision import transforms as T,  utils # type: ignore
from PIL import Image # type: ignore
from tqdm.auto import tqdm # type: ignore
from einops import rearrange, reduce, repeat # type: ignore

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

milestone=86
print('got in')
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4,8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    #image_size = (158,56),
    image_size = (56,160),
    timesteps = 1000,           # number of steps
    sampling_timesteps = 2    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

#accelerator = Accelerator()
#better?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('starting load')
data = torch.load(str('results' '/' f'model-{milestone}.pt'), map_location=device)

diffusion.to(device)
diffusion.load_state_dict(data['model'])
diffusion.step = data['step']

print('ending load')

num_samples = 2
batch=1
shape=(1,3,56,160)
image_size=(56,160)

eta=0.
total_timesteps=1000
sampling_timesteps=250

test_image_path='/n/holyscratch01/howe_lab_seas/dperrin/MAE-data/docker-data/fixedsize-torch/validation/91d22630-fdb1-11ee-a39e-0242ac110004.png'
test_img = Image.open(test_image_path)
transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

t=transform(test_img)
in_image=rearrange(repeat(t, '1 h w -> c h w', c=3),'c h w -> 1 c h w')

with torch.inference_mode():
    
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # type: ignore # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        #print('shape', img.shape)
        imgs = [img]
      
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = None

            model_output = diffusion.model(img, time_cond, None)
            v = model_output
            
            alpha = diffusion.alphas_cumprod[time]
            
            x_start=x_t=alpha.sqrt()*img-(1-alpha).sqrt()*v
            
            alpha_r= 1. / diffusion.alphas_cumprod[time]
            pred_noise = (alpha_r.sqrt()*img-x_start) / (alpha_r -1).sqrt()

    
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            
            alpha_next = diffusion.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

#add in_image*apha.sqrt() + c?  
            img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            imgs.append(img)

        #      ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
            ret=img
            ret = diffusion.unnormalize(ret)


all_images = torch.cat([ret], dim = 0)

utils.save_image(all_images, str('results/' f'sample-new-{milestone}.png'), \
                    nrow = int(math.sqrt(num_samples)))