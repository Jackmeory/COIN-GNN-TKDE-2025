import math
import pdb
import torch
import torch.nn as nn

class FBatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,k_hop,in_channels,bias=True, gcn=True):
        super(FBatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_hop = k_hop
        self.in_channels = in_channels
        self.weight_neigh = nn.Linear(k_hop*in_features*in_channels, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def sgnroot(self,x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self,x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


    def forward(self, x, L):
        # x: [bs, N, in_features], adj: [N, N]
        #fadj = self.featureCal(x,L)
        fadj = L
        x = x.unsqueeze(-1)
        input_x = torch.einsum('bnkcaf,bncf->bnkca', fadj, x)            # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(input_x.view(x.size(0),x.size(1),self.k_hop*self.in_features*self.in_channels))
        x = x.squeeze(-1)  
        if self.weight_self is not None:
            output += self.weight_self(x)               # [bs, N, out_features]
        return output          # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]


    """
    def featureCal(self,x,L):
        device = torch.device('cuda', 0)
        x = x.unsqueeze(-1)
        fadj = torch.stack([torch.stack([torch.stack([torch.einsum('ca,ncb->cab', x[i][j], L[i][j][k]) for k in range(self.k_hop)]) for j in range(x.size(1))]) for i in range(x.size(0))]).to(device)
        fadj += fadj.transpose(-2, -1).to(device)
        fadj = self.row_normalize(self.sgnroot(fadj)).to(device)

        return fadj
    """