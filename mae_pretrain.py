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
from utils import *
from model import *
from USUtils.USLoader import *
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

    batch_size = args.batch_size

        
    train_dataset = USImages(args.data_path+'train/',noise_var=args.train_noise)
    #train_dataset = USImages(args.data_path+'train/')
    val_dataset = USImages(args.data_path+'validation/')
    
    #train_dataset = USImages(args.data_path+'train/',noise_var=args.train_noise)
    #val_dataset = USImages(args.data_path+'validation/')
    
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

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    if args.loging:
        wandb.login()
        wandb.init(config=args,notes='model_size: '+ str(params)) # type: ignore

    
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

            loss = torch.mean(torch.square((predicted_img *mask + (1-mask) * img) - img))
        
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
        if e % 2 == 0:
            with torch.no_grad():
                if e % 20 == 0  or e < 20:
                    ''' visualize the first 16 predicted images on val dataset'''
                    print('sent example')
                    model.set_mask_ratio(0.5)
                    val_img = torch.stack([val_dataset[i][0] for i in [1, 3]])
                    tst_img = torch.stack([train_dataset[i][0] for i in [1]])
                    val_img = val_img.to(device)
                    tst_img = tst_img.to(device)
                    predicted_val_img, mask = model(val_img)
                    
                    predicted_tst_img, tst_mask = model(tst_img)
                    
                    img = torch.cat([(predicted_val_img * (mask)) + (1-mask)*-2,
                                    val_img * (1 - mask)+(mask*-2),
                                    (predicted_val_img * (mask)) + (1-mask)*val_img ], dim=0)
                    img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=3)
                    
                    org_img = torch.cat([val_img,
                                    (predicted_val_img * (mask)) + (1-mask)*val_img ], dim=0)
                    org_img = rearrange(org_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=2)
                    
                    
                    t_img = torch.cat([(predicted_tst_img * (tst_mask)) + (1-tst_mask)*-2,
                                       tst_img * (1 - tst_mask)+(tst_mask*-2),
                                       (predicted_tst_img * (tst_mask)) + (1-tst_mask)*tst_img ], dim=0)
                    t_img = rearrange(t_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=3)
                   
                    err_img = torch.cat([torch.square((predicted_val_img *mask + (1-mask) * val_img) - val_img)
                                    ], dim=0)
                    #err_img = rearrange(org_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=1)
                    
                    if args.loging:
                        img_sd,img_mean=torch.std_mean(val_img)
                        sd,mean=torch.std_mean(predicted_val_img)
                        wandb.log({"val examples": [wandb.Image(img)],
                                   "val Error ": [wandb.Image(err_img)],
                                   "training examples": [wandb.Image(t_img)],
                                   "orginal vs predict": [wandb.Image(org_img)],
                                   "predict_mean":mean,
                                   "predict_sd":sd,
                                   "img_mean":img_mean,
                                   "img_sd":img_sd}) 
                
                '''log the val loss'''
                model.set_mask_ratio(args.mask_ratio)
                val_losses = []      
                for img, label in iter(val_dataloader):
                    img = img.to(device)
                    predicted_img, mask = model(img)
                    val_loss = torch.mean(torch.square((predicted_img *mask + (1-mask) * img) - img))
                    val_losses.append(val_loss.item())            
                    
                avg_val_loss = sum(val_losses) / len(val_losses)
               
                val_metrics={
                    "Val Loss": avg_val_loss,
                    "Loss Delta": avg_val_loss-avg_loss
                }
        
        if args.loging:
            wandb.log({**metrics,**val_metrics})  
                
        ''' save model '''
        torch.save(model, args.model_path)
        wandb.save(args.model_path)
    if args.loging:
        wandb.finish()