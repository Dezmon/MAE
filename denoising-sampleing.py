import os,sys
import math
sys.path.append('denoising-diffusion-pytorch')
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer # type: ignore
from USUtils.USLoader import *
from accelerate import Accelerator # type: ignore
from pathlib import Path
from torchvision import  utils # type: ignore


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def exists(x):
    return x is not None


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
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

accelerator = Accelerator()

print('starting load')
data = torch.load(str('results' '/' f'model-{milestone}.pt'), map_location=accelerator.device)

#model = self.accelerator.unwrap_model(self.model)
diffusion.load_state_dict(data['model'])
 
 
diffusion.step = data['step']

if 'version' in data:
    print(f"loading from version {data['version']}")


print('ending load')

num_samples = 2
shape=(1,3,56,160)
with accelerator.autocast():  #does this do anything for inference speed or just learing? 
    with torch.inference_mode():
        all_images_list = [diffusion.ddim_sample(shape) for _ in range(num_samples)]

all_images = torch.cat(all_images_list, dim = 0)

utils.save_image(all_images, str('results/' f'sample-new-{milestone}.png'), \
                    nrow = int(math.sqrt(num_samples)))