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
#better?
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('starting load')
data = torch.load(str('results' '/' f'model-{milestone}.pt'), map_location=accelerator.device)

#model = self.accelerator.unwrap_model(self.model)
diffusion.to(accelerator.device)
diffusion.load_state_dict(data['model'])
 
 
diffusion.step = data['step']

if 'version' in data:
    print(f"loading from version {data['version']}")


print('ending load')

num_samples = 2
batch=1
shape=(1,3,56,160)
image_size=(56,160)
#with accelerator.autocast():  #does this do anything for inference speed or just learing? 
#    with torch.inference_mode():
#        all_images_list = [diffusion.ddim_sample(shape) for _ in range(num_samples)]

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


exit()
with torch.inference_mode():
    with accelerator.autocast():
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # type: ignore # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = accelerator.device)
        print('shape', img.shape)
        imgs = [img]
      
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = accelerator.device, dtype = torch.long)
            self_cond = None
            pred_noise, x_start, *_ = diffusion.model_predictions(img, time_cond, self_cond, clip_x_start = True, \
                                                                rederive_pred_noise = True)
            
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = diffusion.alphas_cumprod[time]
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