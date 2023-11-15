from modulefinder import Module
from re import X
import re
from statistics import mode
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat, reduce
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x,*args, **kwargs):
        return self.fn(self.norm(x),*args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim), 
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)



class FeedForward_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_1d = nn.Conv2d(1, 1, [3, 1], 1,padding=(1, 0))

    def forward(self,x):
        x_cls = x[:,0]
        x_cls = rearrange(x_cls,'b dim -> b 1 dim')
        x_feature_t = x[:, 1:]
        b,n,dim1 = x_feature_t.shape
        x_feature_t = rearrange(x_feature_t, 'b n dim -> b 1 n dim')
        x_feature_t = self.Conv_1d(x_feature_t)
        # x = reduce(x_feature_t,'b 1 n dim -> b n dim',n = n, dim = dim1)
        x = torch.squeeze(x_feature_t)
        x = torch.cat((x_cls, x), dim=1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):  
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv) 

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

       
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[
                -1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

      
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout,num_channel, mode, MLP_model):
        super().__init__()

        self.layers = nn.ModuleList([])
        if MLP_model == 'MLP':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                        Residual(PreNorm(dim,Attention(dim,heads=heads,dim_head=dim_head,dropout=dropout))),
                        Residual(PreNorm(dim, FeedForward(dim,mlp_head,dropout=dropout)))
                    ]))

        if MLP_model == 'Convolution':  
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                        Residual(PreNorm(dim,Attention(dim, heads=heads,dim_head=dim_head,dropout=dropout))),
                        Residual(PreNorm(dim,FeedForward_Conv()))
                    ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(
                nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))
                
                    


    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](torch.cat(
                        [x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)],
                        dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))  
            ]))
    def forward(self, x, m, mask=None):
        for attn, ff in self.layers:
            x = attn(x, m)
            x = ff(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        # self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attn = None
        
        
        

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads  
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
            
       
        self.attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class ViT(nn.Module):
    def __init__(self,
                 image_size,
                 num_patches,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 MLP_model,
                 mode,
                 dim_head,
                 pool='cls',
                 channels=1,
                 dropout=0.,
                 emb_dropout=0.
                 ):
        super().__init__()

        patch_dim = image_size**2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1,dim)) 
        self.patch_to_embedding = nn.Linear(dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout, num_patches, mode, MLP_model)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )

    def forward(self, x, mask=None): 
    
        x000 = x
        x = self.patch_to_embedding(x)  #[b,n,dim]  
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  #[b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  #[b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.dropout(x)
        out = self.transformer(x, mask)

        # classification: using cls_token output
        x_cls_token = self.to_latent(out[:, 0])
        x_feature = self.to_latent(out[:, 1:])

        # MLP classification layer
        class_x = self.mlp_head(x_cls_token)
        return x_cls_token, x_feature, class_x,out

class ViT_2(nn.Module):  
    def __init__(self,
                 image_size,
                 num_patches,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 MLP_model,
                 mode,
                 dim_head,
                 pool='cls',
                 channels=1,
                 dropout=0.,
                 emb_dropout=0.
                 ):
        super().__init__()
        patch_dim = image_size**2
    
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  
        self.patch_to_embedding = nn.Linear(dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout, num_patches, mode, MLP_model)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )

    def forward(self, x, mask=None):  
        x000 = x
        x = self.patch_to_embedding(x)  #[b,n,dim]  
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)
        out = self.transformer(x, mask)
        return out  