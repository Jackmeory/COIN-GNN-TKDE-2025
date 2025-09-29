import numpy as np
import torch 
from torch_geometric.data import Data, Dataset

class COINDataset(Dataset):
    def __init__(self, inputs, split, nodeID,year,x='', y='',idx = '',edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] # [T, Len, N]
            self.y = inputs[split+'_y'] # [T, Len, N]
        else:
            self.x = x
            self.y = y
        self.idx = idx
        self.nodeID = nodeID
        self.year = year
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        idx = torch.tensor(self.idx[index])
        nodeID = torch.Tensor(self.nodeID)
        year = torch.ones(x.size(0))*torch.tensor([self.year])

        return Data(x=x, y=y, nodeID = nodeID, year = year,idx = idx)  
    def get(self):
        pass
    
    def len(self):
        pass
    
class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs # [T, Len, N]
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return Data(x=x)
    def get(self):
        pass
    
    def len(self):
        pass
    

    
