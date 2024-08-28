import imghdr
import os,sys
import math
import numpy as np
sys.path.append('denoising-diffusion-pytorch')
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer # type: ignore
from USUtils.USLoader import *

from pathlib import Path
from torchvision import transforms as T,  utils # type: ignore
from PIL import Image # type: ignore
from tqdm.auto import tqdm # type: ignore
from einops import rearrange, reduce, repeat # type: ignore


milestone=86
print('got in')
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4,8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = (56,160),
    timesteps = 1000,           # number of steps
    sampling_timesteps = 2    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('starting load')
data = torch.load(str('results' '/' f'model-{milestone}.pt'), map_location=device)

diffusion.to(device)
diffusion.load_state_dict(data['model'])
diffusion.step = data['step']

print('ending load')

num_samples = 4
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
t=t.to(torch.device("cuda:0"))

ref_image=rearrange(repeat(t, '1 h w -> c h w', c=3),'c h w -> 1 c h w')

mask=np.zeros(image_size)
mask[0::2]=1
#mask[10:20,:]=1

mask=torch.from_numpy(mask)
mask=mask.reshape(t.shape)
mask=rearrange(repeat(mask, '1 h w -> c h w', c=3),'c h w -> 1 c h w')


mask=mask.to(torch.float32)
mask=mask.to(device)

all_images= torch.empty((0,3,56,160), dtype=torch.float32)
all_images=all_images.to(device)
# first dimension should be zero.
 
test=[]

with torch.inference_mode():
    for li in range(num_samples):    
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # type: ignore # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_noizier = torch.randn(shape, device = device)
        ref_noize = x_noizier
        ref_image=diffusion.normalize(ref_image)
        imgs = [x_noizier]
      
        x_current = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            alpha = diffusion.alphas_cumprod[time]
            
            ref_img_noize=diffusion.q_sample(x_start =ref_image, t = time_cond, noise = ref_noize)

            x_noizier=x_noizier*(1-mask)+ref_img_noize*(mask)
            
            model_output = diffusion.model(x_noizier, time_cond, None)
            v = model_output
            
            
            x_current=x_t=alpha.sqrt()*x_noizier-(1-alpha).sqrt()*v
            
            alpha_r= 1. / diffusion.alphas_cumprod[time]
            pred_noise = (alpha_r.sqrt()*x_noizier-x_current) / (alpha_r -1).sqrt()

    
            if time_next < 0:
                x_noizier = x_current
                imgs.append(x_noizier)
                continue

            
            alpha_next = diffusion.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_noizier)

            x_noizier = x_current * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            imgs.append(x_noizier)

        #   ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret=x_noizier
        ret = diffusion.unnormalize(ret)
        #utils.save_image(ret, str('results/' f'sample-new-{li}-{milestone}.png'))
       
        all_images = torch.cat((all_images, ret), 0)
    print(all_images.shape)
    #all_images = torch.cat(all_images, ret, dim = 0)

    utils.save_image(all_images, str('results/' f'sample-newAll-{milestone}.png'), \
                    nrow = int(math.sqrt(num_samples)))