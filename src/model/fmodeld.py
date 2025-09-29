import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fgcn_conv import FBatchGCNConv
from model.gcn_conv import BatchGCNConv



class Basic_Model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = FBatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], 2,1,bias=True, gcn=False).to(args.device)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.args = args

    def forward(self, dx, adj):
        N = adj.shape[0]
        if isinstance(dx,torch.Tensor):
            x = dx.reshape((-1, N, self.args.gcn["in_channel"]))    # [bs, N, feature]
        else:
            x = dx.x.reshape((-1, N, self.args.gcn["in_channel"]))
            fadj = dx.L.reshape((-1, N,2, self.args.gcn["in_channel"],1,1))
        x = F.relu(self.gcn1(x, fadj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        if isinstance(dx,torch.Tensor):
            x = x + dx   # [bs, N, feature]
        else:
            x = x + dx.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

 
    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        fadj = data.L.reshape((-1, N,2, self.args.gcn["in_channel"],1,1))
        x = F.relu(self.gcn1(x, fadj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x
