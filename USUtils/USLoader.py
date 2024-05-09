import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import Normalize
import glob
import os
class USImages(torch.utils.data.Dataset,):  # type: ignore
    
    def __init__(self, path: str,noise_var=0):
        super().__init__()
        self.files = [file for file in glob.glob(path+'*.png')]
        
        self.noise_var=noise_var

    def __getitem__(self, index):
        image=torchvision.io.read_image(self.files[index]).to(torch.float)
        sd,mean=torch.std_mean(image)
        image=Normalize(mean,sd)(image)
        noise = torch.randn(image.size()) * self.noise_var
        image=image+noise
        
        return  image, 'noLabel'

    def __len__(self):
        return len(self.files)