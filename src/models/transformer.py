from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(
        self,n_features:int,d_model:int=96,n_heads:int=4,n_layers:int=3,
        n_classes:int=3,dropout:float=0.1,max_seq_len:int=256,
    ):
        super().__init__()
        self.proj=nn.Linear(n_features,d_model)
        self.position=nn.Parameter(torch.zeros(1,max_seq_len,d_model))
        nn.init.normal_(self.position,std=0.02)
        layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dim_feedforward=4*d_model,
            dropout=dropout,batch_first=True,norm_first=True,
            activation="gelu"
        )
        self.encoder=nn.TransformerEncoder(layer,num_layers=n_layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,n_classes)
        self.max_seq_len=max_seq_len

    def forward(self,x):
        if x.shape[1]>self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        z=self.proj(x)+self.position[:,:x.shape[1],:]
        z=self.encoder(z)
        return self.head(self.norm(z[:,-1,:]))


def build_transformer(n_features:int,**kwargs)->TemporalTransformer:
    return TemporalTransformer(n_features,**kwargs)
