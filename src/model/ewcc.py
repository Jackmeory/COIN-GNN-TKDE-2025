import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from math import log2
from scipy.spatial import distance
import math
import numpy as np
from numpy.linalg import norm
import logging
import pdb
from src.model.JSD import JSD
from torch_geometric.data import Data
from utils.graph_completion import graph_completion

class EWC(nn.Module):

    def __init__(self, args, model, sub_adj,adj, adjp, subgraph, neighbors, sliceidx, warn_list, ewc_lambda = 0, ewc_type = 'ewc'):
        super(EWC, self).__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type
        self.adj = adj
        self.adjp = adjp
        self.sub_adj = sub_adj
        self.subgraph = subgraph
        self.neighbors = neighbors
        self.jsd = JSD()
        self.warn_list = warn_list
        self.warn_id = sliceidx
        self.args =args

    def completionc(self,x,warn_list,neighbors):
        train_x= x.clone()
        for i in warn_list:
            index = neighbors[i]
            train_x[:,i,:] = torch.mean(x[:,index,:],dim=1)
        
        return train_x

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, loaderC, lossfunc, device):
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        for i, data in enumerate(loaderC):
            data = data.to(device, non_blocking=True)
            pred = self.model.forward(data, self.adj)
            log_likelihood = lossfunc(data.y, pred, reduction='mean')
            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                est_fisher_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])
    
    def _update_contrastive_params(self, loaderC, device):
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        est_contrastive_info = {name: 0.0 for name in _buff_param_names}
        for i, data in enumerate(loaderC):
            data = data.to(device, non_blocking=True)
            x = data.x.reshape((-1, self.adj.size(0), 12))
            buf_data = self.args.buf_data.reshape((-1, self.adj.size(0), 12))
            buf_data = buf_data[i:i+x.size(0),:,:].float()
            fx = self.completionc(x,self.warn_list, self.neighbors)
            fbuf_data = self.completionc(buf_data,self.warn_list, self.neighbors)
            negative = torch.nonzero(torch.isin(data.idx, self.warn_id), as_tuple=True)[0]
            postive = torch.nonzero(~torch.isin(data.idx, self.warn_id), as_tuple=True)[0] 
            fx = fx[:,self.args.subgraph.numpy(),:].reshape((-1,12))
            x = x[:,self.args.subgraph.numpy(),:].reshape((-1,12))
            fbuf_data = fbuf_data[:,self.args.subgraph.numpy(),:].reshape((-1,12))
            buf_data = buf_data[:,self.args.subgraph.numpy(),:].reshape((-1,12))

            x = self.model.forward(x, self.sub_adj)
            fx = self.model.forward(fx, self.sub_adj)
            buf_data = self.model.forward(buf_data, self.sub_adj)
            fbuf_data = self.model.forward(fbuf_data, self.sub_adj)
            fx = fx.reshape((-1,self.sub_adj.size(0),12))
            x = x.reshape((-1,self.sub_adj.size(0),12))
            fbuf_data = fbuf_data.reshape((-1,self.sub_adj.size(0),12))
            buf_data = buf_data.reshape((-1,self.sub_adj.size(0),12))
            negativeR = torch.cat((x[negative],buf_data),dim=0)
            negativeF = torch.cat((fx[negative],fbuf_data),dim=0)
            postiveR = x[postive]
            postiveF = fx[postive]
            N = negative.size(0)
            M = postive.size(0)          
            negativeR = torch.mean(torch.mean(negativeR,dim=2),dim=1)
            negativeF = torch.mean(torch.mean(negativeF,dim=2),dim=1)
            postiveR = torch.mean(torch.mean(postiveR,dim=2),dim=1)
            postiveF = torch.mean(torch.mean(postiveF,dim=2),dim=1)
            cos = torch.nn.CosineSimilarity(dim=0)
            jsd = self.jsd
            SimN = torch.sum(torch.exp(jsd(negativeR,negativeF)/(self.args.tao)),dim=0)
            loss = (torch.sum(torch.log(torch.exp(jsd(postiveF,postiveR)/(self.args.tao))/SimN.item())))/M
            grad_log_liklihood = autograd.grad(loss, self.model.parameters())
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                est_contrastive_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            self.register_buffer(name + '_estimated_contrastive', est_contrastive_info[name])
        


    def register_ewc_params(self, loaderP, loaderC,  lossfunc, device):
        self._update_fisher_params(loaderC, lossfunc, device)
        self._update_topological_params(loaderP, loaderC, device)
        self._update_contrastive_params(loaderC, device)
        self._update_mean_params()

    def compute_consolidation_loss(self,args):
        tlosses = []
        flosses = []
        closses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            estimated_topological = getattr(self, '{}_estimated_topological'.format(_buff_param_name))
            estimated_contrast = getattr(self, '{}_estimated_contrastive'.format(_buff_param_name))
            if estimated_fisher == None:
                flosses.append(0)
            elif self.ewc_type == 'l2':
                flosses.append((10e-6 * (param - estimated_mean) ** 2).sum())
            else:
                tlosses.append((estimated_topological * (param - estimated_mean) ** 2).sum())
                flosses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
                closses.append((estimated_contrast * (param - estimated_mean) ** 2).sum())
                
        lambdaf =  (self.ewc_lambda / 2)*args.ratio
        lambdat = (1-args.ratio)* (self.ewc_lambda / 2) 
        return lambdat* sum(tlosses) + lambdaf * sum(flosses)+args.cl*sum(closses)
    
    def forward(self, data, adj): 
        return self.model(data, adj)
    
    def contras_lost(self,args, x, fx, adj, buf_idx, idx):
        fx = fx.reshape((-1,12))
        negative = torch.nonzero(torch.isin(idx, buf_idx), as_tuple=True)[0]
        postive = torch.nonzero(~torch.isin(idx, buf_idx), as_tuple=True)[0]
        x = self.model.forward(x, adj)
        fx = self.model.forward(fx, adj)
        fx = fx.reshape((-1,adj.size(0),12))
        x = x.reshape((-1,adj.size(0),12))
        negativeR = torch.cat((x[negative],x[128:,:,:]),dim=0)
        negativeF = torch.cat((fx[negative],fx[128:,:,:]),dim=0)
        postiveR = x[postive]
        postiveF = fx[postive]
        N = negative.size(0)
        M = postive.size(0)
        negativeR = torch.mean(torch.mean(negativeR,dim=2),dim=1)
        negativeF = torch.mean(torch.mean(negativeF,dim=2),dim=1)
        postiveR = torch.mean(torch.mean(postiveR,dim=2),dim=1)
        postiveF = torch.mean(torch.mean(postiveF,dim=2),dim=1)
        cos = torch.nn.CosineSimilarity(dim=0)
        jsd = self.jsd
        SimN = torch.sum(torch.exp(jsd(negativeR,negativeF)/(args.tao)),dim=0)
     
        loss = -args.cl*(torch.sum(torch.log(torch.exp(jsd(postiveF,postiveR)/(args.tao))/SimN.item())))
        return loss.item()
        
        

    def _update_topological_params(self, loaderP, loaderC, device):
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        est_topological_info = {name: 0.0 for name in _buff_param_names}
        for batch_idx, (pre_data,cur_data) in enumerate(zip(loaderP,loaderC)):
            pre_data = pre_data.to(device, non_blocking=True)
            cur_data = cur_data.to(device, non_blocking=True)
            pre_adj = self.adjp
            cur_adj = self.adj
            L = self.subgraph
            L = [x for x in L if x<pre_adj.size(0)]
            pre_feature = self.model.forward(pre_data, pre_adj)
            pre_feature = pre_feature.view(pre_adj.size(0),-1,12)
            cur_feature = self.model.forward(cur_data, cur_adj)
            cur_feature = cur_feature.view(cur_adj.size(0),-1,12)
            cur_feature = (cur_feature-torch.min(cur_feature))/(torch.max(cur_feature)-torch.min(cur_feature))
            pre_feature = (pre_feature-torch.min(pre_feature))/(torch.max(pre_feature)-torch.min(pre_feature))
            pre_feature[pre_feature == 0] = 1e-5
            cur_feature[cur_feature == 0] = 1e-5
            nodes= torch.tensor([]).to(device, non_blocking=True)
            for i in L:
                score = torch.tensor([]).to(device, non_blocking=True)
                for j in self.neighbors[i]:
                    if j <pre_adj.size(0):
                        scorei = (self.jsd(pre_feature[i],cur_feature[i])*self.jsd(pre_feature[j],cur_feature[j]))/(self.jsd(cur_feature[i],cur_feature[j])*self.jsd(pre_feature[i],pre_feature[j]))     
                        scorei = scorei.view(1)
                        score = torch.cat([scorei,score]) 
                nodes = torch.cat([nodes,torch.norm(score).view(1)])
            grad_log_liklihood = autograd.grad(torch.norm(nodes), self.model.parameters())
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                est_topological_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            self.register_buffer(name + '_estimated_topological', est_topological_info[name])

   

