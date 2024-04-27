import re
import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    if np.random.random() < 0.5:
        forward_indexes=[i for i in forward_indexes if i%2==1]+[i for i in forward_indexes if i%2==0]
    else:
        forward_indexes=[i for i in forward_indexes if i%2==0]+[i for i in forward_indexes if i%2==1]
        
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    #gather reorders repeat is used to give indexes the right size repeting the index in to the c dimention
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor,indexes=None):
        T, B, C = patches.shape #what are T, B C (total h*w batch/instances chanels?)
        remain_T = int(T * (1 - self.ratio))
        if np.random.random() < 0.5:
            remain_T=remain_T+12
        else:
            remain_T=remain_T
        indexes = [random_indexes(T) for _ in range(B)] #indexes are just linierindexes into the flaatened h*w image
        #print(indexes)
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        #print('patches dim:', patches.shape)
        patches = patches[:remain_T]
        #print('patches after remain_T :', patches.shape)

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_width=32,
                 image_height=32,
                 image_channels=1,
                 patch_width=2,
                 patch_height=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=4, 
                 mask_ratio=0.75
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_width // patch_width * image_height // patch_height , 1, emb_dim)))
        self.shuffle = PatchShuffle(mask_ratio)
        
        
        self.patchify =torch.nn.Conv2d(image_channels, emb_dim, (1, patch_width),(1,patch_width))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, indexes=None):
        patches = self.patchify(img) 
        
        #print('patches dim after conv:', patches.shape)
        patches = rearrange(patches, 'b c h w -> (h w) b c')

        #print('patches dim after rearange:', patches.shape)
        patches = patches + self.pos_embedding
        patches, forward_indexes, backward_indexes = self.shuffle(patches,indexes)
        #print('indexes dim:', forward_indexes.shape,' ', backward_indexes.shape)
        #adds a cls token with out a pos_embeding to the end? begining?
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        #print('patches after cat dim:', patches.shape)
        patches = rearrange(patches, 't b c -> b t c')
        #print('patches after rearrange:', patches.shape)
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_width=32,
                 image_height=32,
                 image_channels=1,
                 patch_width=2,
                 patch_height=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=4
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_width // patch_width * image_height // patch_height +1 , 1, emb_dim)))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, image_channels * patch_width *patch_height)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', c=image_channels, p2=patch_width, p1=patch_height, w=image_width//patch_width, h=image_height//patch_height)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 patch_width,
                 patch_height,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75
                 ) -> None:
        super().__init__()
                 

        self.encoder = MAE_Encoder(image_width, image_height, image_channels, patch_width, patch_height, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_width, image_height, image_channels, patch_width, patch_height, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask


if __name__ == '__main__':
    #shuffle = PatchShuffle(0.75)
    #a = torch.rand(16, 2, 10)
    #print('a: ',a.shape)
    #b, forward_indexes, backward_indexes = shuffle(a)
    #print('b: ',b.shape)
    #print('forward index: ',forward_indexes.shape)

    #example, chanel(color),w,h
    img = torch.rand(1, 1, 32, 64)
    sz=img.shape
    print('img dim:', sz)

    #patch=torch.nn.Conv2d(image_channels, emb_dim, patch_width, patch_height)(img)
    #patch=torch.nn.Conv2d(1, 196, 2,2)(img)
    #print(patch.shape)
    #patch=torch.nn.Conv2d(1, 196, (1, 64),(1,64))(img)
    #print(patch.shape)
    #patch=torch.nn.Conv2d(1, 196, (2, 2),(2,2))(img)
    #print(patch.shape)
    encoder = MAE_Encoder( image_width=64,image_height=32,patch_height=1,patch_width=64)
    decoder = MAE_Decoder( image_width=64,image_height=32,patch_height=1,patch_width=64)
    features, backward_indexes = encoder(img,'bob')
    #print('backward index: ',backward_indexes.shape)
    #print(backward_indexes)
    #predicted_img, mask = decoder(features, backward_indexes)
    #print('mask:', mask.shape)
    #print('predicted shape:',predicted_img.shape)
    #loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    #print(loss)
