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
    parser.add_argument('--max_device_batch_size', type=int, default=192+64)
    parser.add_argument('--base_learning_rate', type=float, default=3.0e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--data_path', type=str, default='docker-data/torch-test/')
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')

    args = parser.parse_args()

    setup_seed(args.seed)
    wandb.login()
    wandb.init(config=args)

    wandb.init(
      # Set the project where this run will be logged
      project="MAE-US", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      #name=f"experiment_{run}", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": args.base_learning_rate,
      "architecture": "MAE",
      "dataset": "SL-US",
      "epochs": args.total_epoch,
      })
  
 


    
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    #train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    #val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    
    class Images(torch.utils.data.Dataset,):  # type: ignore
        def __init__(self, path: str, transform=None):
            super().__init__()
            self.files = [file for file in glob.glob(path+'*.png')]
            self.transform = transform

        def __getitem__(self, index):
            #image=read_image(self.files[index])
            #image=rearrange(torchvision.io.read_image(self.files[index]), 'b h w -> h w b')
            image=torchvision.io.read_image(self.files[index]).to(torch.float)/255.0
            
            if self.transform:
                image = self.transform(image)
            return  image, 'noLabel'

        def __len__(self):
            return len(self.files)
        
    #train_dataset = Images(args.data_path+'train/',   transform=Compose([Lambda(lambda x: x.repeat(1, 1, 3) ), Normalize(0.5, 0.5)]))
    #val_dataset = Images(args.data_path+'validation/',   transform=Compose([Lambda(lambda x: x.repeat(1, 1, 3) ), Normalize(0.5, 0.5)]))
    train_dataset = Images(args.data_path+'train/',transform=Compose([ Normalize(0.5, 0.5)]))
    val_dataset = Images(args.data_path+'validation/',transform=Compose([Normalize(0.5, 0.5)]))
    
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=3)  # type: ignore
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=True, num_workers=3)  # type: ignore
    
    writer = SummaryWriter(os.path.join('logs', 'US', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(image_width=158,image_height=56,patch_height=1,patch_width=158,emb_dim=90,mask_ratio=args.mask_ratio,image_channels=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
#        for img, label in tqdm(iter(dataloader)):
        for img, label in iter(dataloader):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            #is there a NaN in the predicted images?causeing this to fail 
            #loss = torch.mean(torch.sqrt(torch.square(predicted_img - img)) * mask) / args.mask_ratio
            loss = torch.mean((torch.square(predicted_img - img)) * mask) / args.mask_ratio
 #           loss = torch.mean(abs(predicted_img - img) * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
#        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        metrics={
           "Training Loss": avg_loss
        }
        wandb.log(metrics)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        if e % 2 == 1:
            with torch.no_grad():
                if e % 10 == 1 :
                    print('sent example')
                    val_img = torch.stack([val_dataset[i][0] for i in range(4)])
                    val_img = val_img.to(device)
                    predicted_val_img, mask = model(val_img)
                    predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
                    img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
                    img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                    wandb.log({"examples": [wandb.Image(img)]}) 
                
                val_losses = []
                val_losses_MSE = []
                val_losses_total_mae = []
                
                for img, label in iter(val_dataloader):
                    img = img.to(device)
                    predicted_img, mask = model(img)
                    loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                    val_losses_MSE.append(loss.item())

                    loss = torch.mean(abs(predicted_img - img) * mask) / args.mask_ratio
                    val_losses.append(loss.item())
                    
                    loss = torch.mean(abs((predicted_img * mask + img * (1 - mask)) - img))
                    val_losses_total_mae.append(loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_loss_MSE = sum(val_losses_MSE) / len(val_losses_MSE)
                avg_val_loss_total_mae = sum(val_losses_total_mae) / len(val_losses_total_mae)
    #            writer.add_scalar('mae_val_loss', avg_loss, global_step=e)
                val_metrics={
                    "Val Loss MAE": avg_val_loss,
                    "Val Loss MSE": avg_val_loss_MSE,
                    "Val Loss total MAE": avg_val_loss_total_mae,
                    "Loss Delta": avg_val_loss_MSE-avg_loss
                }
                wandb.log({**metrics,**val_metrics})  
                
            ''' save model '''
            torch.save(model, args.model_path)
    wandb.finish()