# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.buffer import Buffer
from model.continual_model import ContinualModel
from torch_geometric.utils import to_dense_batch, k_hop_subgraph
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
import torch.nn.functional as func
import higher
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers
from torch import autograd
from utils.graph_completion import graph_completion
import numpy as np
from src.model.JSD import JSD


class DM(ContinualModel):

    def __init__(self,args):
        self.buffer = Buffer(args.buffer_size, 'cuda')
        self.currentbuffer = Buffer(args.buffer_size, 'cuda')
        self.transform = None
    
    def completionc(self,x,y,args):
        train_x,train_y= x.clone(),y.clone()
        for i in args.warn_list:
            index = args.neighbors[i]
            train_x[:,i,:] = torch.mean(x[:,index,:],dim=1)
            train_y[:,i,:] = torch.mean(y[:,index,:].float(),dim=1)
        
        return train_x,train_y
    


    def observe(self, data, args, model,lossfunc):
        if args.year == args.begin_year:
            N = args.sub_adj.size(0)
            dx = data.x
            dy = data.y
            dnodeID = data.nodeID
            dyear = data.year
            didx = data.idx
            real_batch_size = int(dx.size(0)/N)
            args.optimizer.zero_grad()
            if not self.currentbuffer.is_empty():
                buf_inputs, buf_labels, buf_nodeID, buf_year , buf_idx= self.currentbuffer.get_data(
                    real_batch_size, transform=self.transform)
                dx = torch.cat((dx, buf_inputs.reshape((-1,12))))
                dy = torch.cat((dy, buf_labels.reshape((-1,12))))
                dnodeID = torch.cat((dnodeID, buf_nodeID.reshape((-1))))
                dyear = torch.cat((dyear, buf_year.reshape((-1))))
                didx = torch.cat((didx, buf_idx))
            pred = model(dx, args.sub_adj)
            x = dx.reshape((-1, N, 12))
            y = dy.reshape((-1, N, 12))
            nodeID = dnodeID.reshape((-1, N))
            year = dyear.reshape((-1, N))
            loss = lossfunc(dy, pred, reduction="mean")
            self.currentbuffer.add_data(examples=x[:real_batch_size],
                             labels=y[:real_batch_size],
                             nodeID = nodeID[:real_batch_size],
                             year = year[:real_batch_size],
                             idx = didx[:real_batch_size])
        else:
            if args.Nepoch<args.start_selection:
                N = args.adj.size(0)
                M = args.sub_adj.size(0)
                dx = data.x
                dy = data.y
                dnodeID = data.nodeID
                dyear = data.year.reshape((-1,N))
                dyear = dyear[:,args.subgraph.numpy()]
                dyear = dyear.reshape((-1))
                didx = data.idx
                real_batch_size = int(dx.size(0)/N)
                args.optimizer.zero_grad()
                if not self.currentbuffer.is_empty():
                    buf_inputs, buf_labels, buf_nodeID, buf_year , buf_idx, score = self.currentbuffer.get_data(
                        real_batch_size, transform=self.transform)
                    dx = torch.cat((dx, buf_inputs.reshape((-1,12))))
                    dy = torch.cat((dy, buf_labels.reshape((-1,12))))
                    dnodeID = torch.cat((dnodeID, buf_nodeID.reshape((-1))))
                    dyear = torch.cat((dyear, buf_year.reshape((-1))))
                    didx = torch.cat((didx, buf_idx))
                
                x = dx.reshape((-1, N, 12))
                y = dy.reshape((-1, N, 12))
                nodeID = dnodeID.reshape((-1, M))
                year = dyear.reshape((-1, M))

                self.currentbuffer.add_data(examples=x[:real_batch_size],
                             labels=y[:real_batch_size],
                             nodeID = nodeID[:real_batch_size],
                             year = year[:real_batch_size],
                             idx = didx[:real_batch_size],
                             score = torch.zeros(args.buffer_size))
                dxF, dyF, = self.completionc(x,y,args)
                dxF = dxF[:,args.subgraph.numpy(),:]

                x = x[:,args.subgraph.numpy(),:]
                x = x.reshape((-1,12))
                y = y[:,args.subgraph.numpy(),:]
                y = y.reshape((-1,12))
                pred = model(x, args.sub_adj)

                if not self.currentbuffer.is_empty():
                    if real_batch_size == args.batch_size:
                        used_batch = args.Fbatch1
                    else:
                        used_batch = args.Fbatch2
                    pred, _ = to_dense_batch(pred, batch=used_batch)
                    y, _ = to_dense_batch(y, batch=used_batch)
                    pred = pred[:, args.mapping, :]
                    y = y[:, args.mapping, :]
                    loss = lossfunc(y, pred, reduction="mean")
                else:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    y, _ = to_dense_batch(y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    y = y[:, args.mapping, :]
                    loss = lossfunc(y, pred, reduction="mean")
                check = loss.clone()
                loss += model.compute_consolidation_loss(args)
            else:
                N = args.adj.size(0)
                M = args.sub_adj.size(0)
                dx = data.x
                dy = data.y
                dnodeID = data.nodeID
                dyear = data.year.reshape((-1,N))
                dyear = dyear[:,args.subgraph.numpy()]
                dyear = dyear.reshape((-1))
                didx = data.idx
                real_batch_size = int(dx.size(0)/N)
                args.optimizer.zero_grad()
                buf_inputs, buf_labels, buf_nodeID, buf_year , buf_idx = self.buffer.get_data(
                        real_batch_size, transform=self.transform)
                dx = torch.cat((dx, buf_inputs.reshape((-1,12))))
                dy = torch.cat((dy, buf_labels.reshape((-1,12))))
                dnodeID = torch.cat((dnodeID, buf_nodeID.reshape((-1))))
                dyear = torch.cat((dyear, buf_year.reshape((-1))))
                didx = torch.cat((didx, buf_idx))
                x = dx.reshape((-1, N, 12))
                y = dy.reshape((-1, N, 12))
                nodeID = dnodeID.reshape((-1, M))
                year = dyear.reshape((-1, M))

                dxF, dyF, = self.completionc(x,y,args)
                dxF = dxF[:,args.subgraph.numpy(),:]

                x = x[:,args.subgraph.numpy(),:]
                x = x.reshape((-1,12))
                y = y[:,args.subgraph.numpy(),:]
                y = y.reshape((-1,12))
                pred = model(x, args.sub_adj)               

                if real_batch_size==args.batch_size:
                    used_batch = args.Fbatch1
                else:
                    used_batch = args.Fbatch2
                pred, _ = to_dense_batch(pred, batch=used_batch)
                y, _ = to_dense_batch(y, batch=used_batch)
                pred = pred[:, args.mapping, :]
                y = y[:, args.mapping, :]
                loss = lossfunc(y, pred, reduction="mean")
                loss += model.compute_consolidation_loss(args)

                subsample = args.validation_size
                bx, by, b_nodeID, b_year , b_idx= self.buffer.get_data(
                        subsample, transform=self.transform)
                nx, ny, n_nodeID, n_year , n_idx, score= self.currentbuffer.get_data(
                        subsample, transform=self.transform)


                dxbF, dybF, = self.completionc(bx,by,args)
                dxbF = dxbF[:,args.subgraph.numpy(),:]
                dxnF, dynF, = self.completionc(nx,ny,args)
                dxnF = dxnF[:,args.subgraph.numpy(),:]
                bx = bx[:,args.subgraph.numpy(),:]
                bx = bx.reshape((-1,12))
                by = by[:,args.subgraph.numpy(),:]
                by = by.reshape((-1,12)).float()
                nx = nx[:,args.subgraph.numpy(),:]
                nx = nx.reshape((-1,12))
                ny = ny[:,args.subgraph.numpy(),:]
                ny = ny.reshape((-1,12)).float()


                iteration = 1
                with higher.innerloop_ctx(model, args.optimizer) as (meta_model, meta_opt):
                    base1 = torch.ones((y.shape[0],y.shape[1],12), device=args.device)
                    eps1 = torch.zeros((y.shape[0],y.shape[1],12), requires_grad=True, device=args.device)
                    for i in range(iteration):
                        meta_train_outputs = meta_model(x,args.sub_adj)
                        meta_train_outputs, _ = to_dense_batch(meta_train_outputs, batch=used_batch)
                        meta_train_outputs = meta_train_outputs[:, args.mapping, :]
                        meta_train_loss = lossfunc(y, meta_train_outputs, reduction="none")
                        meta_train_loss += meta_model.compute_consolidation_loss(args)
                        meta_train_loss = (torch.sum(eps1 * meta_train_loss) + torch.sum(base1 * meta_train_loss)) / torch.tensor(y.shape[0]*y.shape[1]*12)
                        meta_opt.step(meta_train_loss)
                    meta_val1_outputs = meta_model(bx,args.sub_adj)
                    meta_val1_outputs, _ = to_dense_batch(meta_val1_outputs, batch=args.Fbatch3)
                    by, _ = to_dense_batch(by, batch=args.Fbatch3)
                    meta_val1_outputs = meta_val1_outputs[:, args.mapping, :]
                    by = by[:, args.mapping, :]
                    meta_val1_loss = lossfunc(by, meta_val1_outputs, reduction="mean")
                    meta_val1_loss = meta_val1_loss.float()
                    meta_val1_loss += meta_model.compute_consolidation_loss(args)
                    eps1 = eps1.float()
                    eps_grads1 = autograd.grad(meta_val1_loss, eps1)[0].detach()

                with higher.innerloop_ctx(model, args.optimizer) as (meta_model2, meta_opt2):
                    base2 = torch.ones((y.shape[0],y.shape[1],12), device=args.device)
                    eps2 = torch.zeros((y.shape[0],y.shape[1],12), requires_grad=True, device=args.device)
                    for i in range(iteration):
                        meta_train_outputs2 = meta_model2(x,args.sub_adj)
                        meta_train_outputs2, _ = to_dense_batch(meta_train_outputs2, batch=used_batch)
                        meta_train_outputs2 = meta_train_outputs2[:, args.mapping, :]                       
                        meta_train_loss2 = lossfunc(y, meta_train_outputs2, reduction="none")
                        meta_train_loss2 += meta_model2.compute_consolidation_loss(args)
                        meta_train_loss2 = (torch.sum(eps2 * meta_train_loss2) + torch.sum(base2 * meta_train_loss2)) / torch.tensor(y.shape[0]*y.shape[1]*12)
                        meta_opt2.step(meta_train_loss2)
                    meta_val2_outputs = meta_model2(nx,args.sub_adj)
                    meta_val2_outputs, _ = to_dense_batch(meta_val2_outputs, batch=args.Fbatch3)
                    ny, _ = to_dense_batch(ny, batch=args.Fbatch3)
                    meta_val2_outputs = meta_val2_outputs[:, args.mapping, :]
                    ny = ny[:, args.mapping, :]
                    meta_val2_loss = lossfunc(ny, meta_val2_outputs, reduction="mean")
                    meta_val2_loss = meta_val2_loss.float()
                    meta_val2_loss += meta_model2.compute_consolidation_loss(args)
                    eps2 = eps2.float()
                    eps_grads2 = autograd.grad(meta_val2_loss, eps2)[0].detach()
                gn = gradient_normalizers([eps_grads1, eps_grads2], [meta_val1_loss.item(), meta_val2_loss.item()], "ours")
                for gr_i in range(len(eps_grads1)):
                    eps_grads1[gr_i] = eps_grads1[gr_i] / gn[0]
                for gr_i in range(len(eps_grads2)):
                    eps_grads2[gr_i] = eps_grads2[gr_i] / gn[1]
                # compute gamma
                sol, min_norm = MinNormSolver.find_min_norm_element([eps_grads1, eps_grads2])
                # fused influence
                w_tilde = sol[0] * eps_grads1 + (1 - sol[0]) * eps_grads2
                w_tilde = w_tilde.reshape((-1,len(args.mapping),12))
                w_tilde = w_tilde.mean(dim=2)
                w_tilde = w_tilde.mean(dim=1)
                if args.inf:
                    self.currentbuffer.replace_by_score(args, examples=dx.reshape((-1, N, 12)),
                                labels=dx.reshape((-1, N, 12)),
                                nodeID = nodeID,
                                year = year,
                                idx = didx,
                                score = w_tilde)
        return loss





    def observeV(self, data, args, model,lossfunc):
        dx = data.x
        dy = data.y
        pred = model(dx, args.sub_adj)
          
        if args.strategy == "incremental" and args.year > args.begin_year:
            pred, _ = to_dense_batch(pred, batch=data.batch)
            dy, _ = to_dense_batch(dy, batch=data.batch)
            pred = pred[:, args.mapping, :]
            dy = dy[:, args.mapping, :]
        loss = masked_mae_np(dy.cpu().data.numpy(), pred.cpu().data.numpy(), 0)

        return loss
    
    def observeT(self, data, args, model,lossfunc):
        dx = data.x
        dy = data.y
        pred = model(dx, args.adjf)
        loss = func.mse_loss(dy, pred, reduction="mean")

        return loss, pred
    

