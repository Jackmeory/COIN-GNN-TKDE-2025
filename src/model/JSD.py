
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from math import log2
from scipy.spatial import distance

import numpy as np
import logging
import pdb

from torch_geometric.data import Data
class JSD1(nn.Module):
    def __init__(self):
        super(JSD1, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

class JSD2(nn.Module):
    """
    Forward KL Divergence: KL(P||Q)
    P is the target distribution, Q is the predicted distribution
    Also known as the "I-projection" or "information projection"
    Tends to find a Q that covers all modes of P
    """
    def __init__(self):
        super(JSD2, self).__init__()
        
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        
        # Ensure proper probability distributions and avoid numerical issues
        p = F.softmax(p, dim=1) + 1e-10
        q = F.softmax(q, dim=1) + 1e-10
        
        # KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        return torch.sum(p * torch.log(p / q), dim=1).mean()

class JSD(nn.Module):
    """
    Backward KL Divergence: KL(Q||P)
    P is the target distribution, Q is the predicted distribution
    Also known as the "M-projection" or "moment projection"
    Tends to find a Q that focuses on the major mode of P
    """
    def __init__(self):
        super(JSD, self).__init__()
        
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        
        # Ensure proper probability distributions and avoid numerical issues
        p = F.softmax(p, dim=1) + 1e-10
        q = F.softmax(q, dim=1) + 1e-10
        
        # KL(Q||P) = Σ Q(x) * log(Q(x)/P(x))
        return torch.sum(q * torch.log(q / p), dim=1).mean()