import numpy as np
import torch
class graph_completion():
    def __init__(self):
        pass
    def completion(self,input,args):
        select = [args.fsubgraph.numpy()[i] for i in list(range(args.fsubgraph.size(0))) if args.fsubgraph.numpy()[i] < args.graph_size]
        unselect = [args.fsubgraph.numpy()[i] for i in list(range(args.fsubgraph.size(0))) if args.fsubgraph.numpy()[i] >= args.graph_size]
        D = len(unselect)

        train_x = np.concatenate((input['train_x'][:,:,select],np.zeros((input['train_x'].shape[0],input['train_x'].shape[1],D))),axis =-1)
        train_y = np.concatenate((input['train_y'][:,:,select],np.zeros((input['train_y'].shape[0],input['train_y'].shape[1],D))),axis =-1)
        val_x = np.concatenate((input['val_x'][:,:,select],np.zeros((input['val_x'].shape[0],input['val_x'].shape[1],D))),axis =-1)
        val_y = np.concatenate((input['val_y'][:,:,select],np.zeros((input['val_y'].shape[0],input['val_y'].shape[1],D))),axis =-1)
        for i in range(len(select),args.fsubgraph.numpy().shape[0]):
            usage = [x for x in args.fneighbors[args.fsubgraph.numpy()[i]] if x < args.graph_size]
            index = [x for x in usage]
            train_x[:,:,i] = np.mean(input['train_x'][:,:,index],axis=-1)
            train_y[:,:,i] = np.mean(input['train_y'][:,:,index],axis=-1)
            val_x[:,:,i] = np.mean(input['val_x'][:,:,index],axis=-1)
            val_y[:,:,i] = np.mean(input['val_y'][:,:,index],axis=-1)
        
        return train_x,train_y,val_x,val_y
    
    def completionpro(self,input,args):
        select = [args.subgraph.numpy()[i] for i in list(range(args.subgraph.size(0))) if args.subgraph.numpy()[i] not in args.warn_list]
        unselect = [args.subgraph.numpy()[i] for i in list(range(args.subgraph.size(0))) if args.subgraph.numpy()[i] in args.warn_list]
        D = len(unselect)

        train_x = np.concatenate((input['train_x'][:,:,select],np.zeros((input['train_x'].shape[0],input['train_x'].shape[1],D))),axis =-1)
        train_y = np.concatenate((input['train_y'][:,:,select],np.zeros((input['train_y'].shape[0],input['train_y'].shape[1],D))),axis =-1)
        val_x = np.concatenate((input['val_x'][:,:,select],np.zeros((input['val_x'].shape[0],input['val_x'].shape[1],D))),axis =-1)
        val_y = np.concatenate((input['val_y'][:,:,select],np.zeros((input['val_y'].shape[0],input['val_y'].shape[1],D))),axis =-1)
        for i in range(len(select),args.subgraph.numpy().shape[0]):
            usage = [x for x in args.fneighbors[args.subgraph.numpy()[i]] if x < args.graph_size]
            index = [x for x in usage]
            train_x[:,:,i] = np.mean(input['train_x'][:,:,index],axis=-1)
            train_y[:,:,i] = np.mean(input['train_y'][:,:,index],axis=-1)
            val_x[:,:,i] = np.mean(input['val_x'][:,:,index],axis=-1)
            val_y[:,:,i] = np.mean(input['val_y'][:,:,index],axis=-1)
        
        return train_x,train_y,val_x,val_y   

    def node_average(node, args, node_feature):
        pass

    def clean(self,args):
        index = []
        for i in range(args.fsubgraph.size(0)):
            if min(args.fneighbors[args.fsubgraph[i].item()])>=args.graph_size and args.fsubgraph[i].item()>=args.graph_size:
                index.append(i)

        clean_list = [x for x in range(args.fsubgraph.size(0)) if x not in index]
        map_clean = [args.fmapping[i].item() for i in range(args.fmapping.size(0)) if args.fmapping[i].item() not in index]
        map_value = [args.fsubgraph[i].item() for i in map_clean]        
        args.fsubgraph = args.fsubgraph[clean_list]
        args.fmapping = torch.tensor([torch.where(args.fsubgraph == x)[0].item() for x in map_value])
        mask = torch.zeros(args.sub_adj.size(0), dtype = torch.bool)
        mask[clean_list]=True
        args.sub_adj = args.sub_adj[mask][:,mask]

    def cleanpro(self,args):
        index = []
        for i in range(args.subgraph.size(0)):
            if min(args.cneighbors[args.subgraph[i].item()]) in args.warn_list and args.subgraph[i].item() in args.warn_list:
                index.append(i)

        clean_list = [x for x in range(args.subgraph.size(0)) if x not in index]
        map_clean = [args.mapping[i].item() for i in range(args.mapping.size(0)) if args.mapping[i].item() not in index]
        map_value = [args.subgraph[i].item() for i in map_clean]        
        args.subgraph = args.subgraph[clean_list]
        args.mapping = torch.tensor([torch.where(args.subgraph == x)[0].item() for x in map_value])
        mask = torch.zeros(args.sub_adj.size(0), dtype = torch.bool)
        mask[clean_list]=True
        args.sub_adj = args.sub_adj[mask][:,mask]



                
