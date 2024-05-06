from ast import arg
import os
import argparse
import math
import torch  # type: ignore
import torch.cuda
import torchvision
from torchvision.io import read_image
import glob
from torch.utils.tensorboard.writer import SummaryWriter  
from torchvision.transforms import Lambda, ToTensor, Compose, Normalize
from einops import repeat, rearrange

from tqdm import tqdm

from model import *
from utils import setup_seed
import wandb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=192+64)
    parser.add_argument('--base_learning_rate', type=float, default=3.0e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--project', type=str, default='MAE')
    parser.add_argument('--embeding_dim', type=int, default=90)
    parser.add_argument('--loging', type=bool, default=False)
    parser.add_argument('--train_noise', type=float, default=0.2)
    parser.add_argument('--encoder_layer', type=int, default=12)
    parser.add_argument('--decoder_layer', type=int, default=4)

    args = parser.parse_args()
    args = parser.parse_args()

    setup_seed(args.seed)
    if args.loging:
        wandb.login()
        wandb.init(config=args) # type: ignore

    batch_size = args.batch_size

    #train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    #val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    
    class Images(torch.utils.data.Dataset,):  # type: ignore
        
        def __init__(self, path: str, transform=None,noise_var=0):
            super().__init__()
            self.files = [file for file in glob.glob(path+'*.png')]
            self.transform = transform
            self.noise_var=noise_var

        def __getitem__(self, index):
            image=torchvision.io.read_image(self.files[index]).to(torch.float)/255.0
            noise = torch.randn(image.size()) * self.noise_var
            image=image+noise
            if self.transform:
                image = self.transform(image)
            return  image, 'noLabel'

        def __len__(self):
            return len(self.files)
        
    train_dataset = Images(args.data_path+'train/',noise_var=args.train_noise,transform=Compose([ Normalize(0.5, 0.5)]))
    val_dataset = Images(args.data_path+'validation/',transform=Compose([Normalize(0.5, 0.5)]))
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=3)  # type: ignore
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=3)  # type: ignore
    
    writer = SummaryWriter(os.path.join('logs', 'US', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(image_width=158,image_height=56,patch_height=1,patch_width=158,
                    emb_dim=args.embeding_dim,mask_ratio=args.mask_ratio,
                    encoder_layer=args.encoder_layer,decoder_layer=args.decoder_layer,
                    image_channels=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)




    #import pytorch_ssim
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_loss = StructuralSimilarityIndexMeasure(kernel_size=3).to('cuda')    
    
    crossover=10000
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        model.set_mask_ratio(args.mask_ratio)
        
        losses = []
#       for img, label in tqdm(iter(dataloader)):
        for img, label in iter(dataloader):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            if e < crossover:
                #loss = torch.mean((torch.square(predicted_img - img)) * mask) / args.mask_ratio
                loss = torch.mean((torch.square(predicted_img * mask + img * (1 - mask) - img)) )
            else:
                loss=1-ssim_loss(img,predicted_img * mask + img * (1 - mask))
  
            loss.backward() #gradient calculated per per batch
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())

        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)

        metrics={
           "Training Loss": avg_loss
        }
        
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        model.eval()

        val_metrics={}
        if e % 2 == 1:
            with torch.no_grad():
                if e % 10 == 1  or e < 20:
                    ''' visualize the first 16 predicted images on val dataset'''
                    print('sent example')
                    model.set_mask_ratio(0.5)
                    val_img = torch.stack([val_dataset[i][0] for i in [1, 3]])
                    val_img = val_img.to(device)
                    predicted_val_img, mask = model(val_img)
                    #predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
                    #predicted_val_img = predicted_val_img * mask 
                    img = torch.cat([predicted_val_img, val_img * (1 - mask), val_img ], dim=0)
                    img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=3)
                    if args.loging:
                        wandb.log({"examples": [wandb.Image(img)]}) 
                
                '''log val loss'''
                model.set_mask_ratio(args.mask_ratio)
                val_losses = []      
                SSIM_val_losses = []      
                for img, label in iter(val_dataloader):
                    img = img.to(device)
                    predicted_img, mask = model(img)
                    #val_loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                        #val_loss = torch.mean((torch.square(predicted_img - img)) * mask) / args.mask_ratio
                    val_loss = torch.mean((torch.square(predicted_img * mask + img * (1 - mask) - img)) )
                    SSIM_val_loss=1-ssim_loss(img,predicted_img * mask + img * (1 - mask))

                    
                    
                    SSIM_val_losses.append(SSIM_val_loss.item())            
                    val_losses.append(val_loss.item())            
                    

                avg_val_loss = sum(val_losses) / len(val_losses)
                SSIM_avg_val_loss = sum(SSIM_val_losses) / len(SSIM_val_losses)
               
                val_metrics={
                    "SSIM Val Loss": SSIM_avg_val_loss,
                    "Val Loss": avg_val_loss,
                    "Loss Delta": avg_val_loss-avg_loss
                }
        
        if args.loging:
            wandb.log({**metrics,**val_metrics})  
                
        ''' save model '''
        torch.save(model, args.model_path)
    
    if args.loging:
        wandb.finish()