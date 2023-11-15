from distutils.errors import LibError
from re import X
from tkinter.messagebox import NO
from turtle import forward
from cv2 import sepFilter2D, transpose
from requests import head
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from vit_transformer import ViT, ViT_2, TransformerDecoder
from einops import rearrange, reduce, repeat
import math
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, default='./config/train.json', help='json path')
args = parser.parse_args()

json_path = args.json_path
with open(json_path, 'r') as f:
    args = json.load(f)


class CAST(nn.Module):
    def __init__(self, num_of_bands, num_of_class, patch_size, dataset, head, dim_head):
        super(CAST, self).__init__()
        self.num_of_bands = num_of_bands 
        self.num_of_class = num_of_class
        self.head = head
        self.dim_head = dim_head
        self.transformer_C = ViT(
                                image_size=patch_size,   
                                num_patches= num_of_bands * 2,  
                                num_classes=args["model"]["class_number"],
                                dim = patch_size ** 2,
                                depth=1,
                                heads=patch_size,        
                                mlp_dim=8,
                                dim_head=patch_size,                        
                                dropout=0.1,
                                emb_dropout=0.1,
                                mode=args["model"]["mode"] ,
                                MLP_model =args["model"]["MLP_mode"] 
                                )

        self.transformer_S = ViT_2(       
                                image_size=patch_size,   
                                num_patches= patch_size ** 2,    
                                num_classes=args["model"]["class_number"],
                                dim= num_of_bands * 2,
                                depth=1,                                     
                                heads=head,
                                dim_head=dim_head,
                                mlp_dim=8,
                                dropout=0.1,
                                emb_dropout=0.1,
                                mode=ViT, 
                                MLP_model =args["model"]["MLP_mode"] 
                                )

        
        self.transformer_decoder = TransformerDecoder(dim = num_of_bands * 2, depth = 1, heads = head,
                                                      dim_head = dim_head,mlp_dim = 8, dropout = 0. )    
    
    def forward(self, x1, x2=None):
        b, n,h,w = x1.shape
        x2_embedded = rearrange(x1, 'b n h w -> b n (h w)')  # [b,n,hw]
        x2_embedded = x2_embedded.transpose(1,2)      #[b,hw,n]
        x2_transformer = self.transformer_S(x2_embedded)     # [b,hw,n]
        x2_decoder = self.transformer_decoder(x2_embedded, x2_transformer)
        x1_embedded = x2_decoder.transpose(1,2)  #[b,n,hw]         
        x1_transformer = self.transformer_C(x1_embedded)
        return x1_transformer[2], x2_embedded, x2_decoder 