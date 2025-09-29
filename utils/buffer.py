# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms


def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if type(dataset.train_loader.dataset.data) == torch.Tensor else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            refold_transform = lambda x: x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                refold_transform = lambda x: (x.cpu()*255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                refold_transform = lambda x: (x.cpu()*255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
            ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
            ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
                ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
                ])

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ('ring', 'reservoir')
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'nodeID', 'year','idx', 'score']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     nodeID: torch.Tensor, year: torch.Tensor,idx: torch.Tensor,score: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, nodeID=None, year=None, idx=None, score=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, nodeID, year,idx,score)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if nodeID is not None:
                    self.nodeID[index] = nodeID[i].to(self.device)
                if year is not None:
                    self.year[index] = year[i].to(self.device)
                if idx is not None:
                    self.idx[index] = idx[i].to(self.device)
                if score is not None:
                    self.score[index] = score[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

        return ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple
    
    def get_index(self, transform: transforms=None) -> Tuple:
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        attr = getattr(self, 'idx')
        ret_tuple += (attr,)
        return ret_tuple


    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
    
    def shrink(self, node_list) -> None:
        """
        Only preserve the nodes in node_list.
        """
        for attr_str in ['examples', 'labels', 'nodeID', 'year']:
            if hasattr(self, attr_str):
                if attr_str in ['examples', 'labels']:
                    setattr(self, attr_str, getattr(self, attr_str)[:,node_list,:].to(self.device))
                else:
                    setattr(self, attr_str, getattr(self, attr_str)[:,node_list].to(self.device))

    def merge(self, examples, labels, nodeID, year) -> None:
        """
        Select those new nodes for merging.
        """

        pre_nodeID = getattr(self, 'nodeID')
        merge_nodeID = torch.cat([nodeID, pre_nodeID], dim=1)
        merge_nodeID, indices = torch.sort(merge_nodeID, dim=1)
        setattr(self, 'nodeID', merge_nodeID.to(self.device))
        N = merge_nodeID.size(1)
        merge_examples = torch.zeros((1000,N,12))
        merge_labels = torch.zeros((1000,N,12))
        merge_years = torch.zeros((1000,N))
        for i in range(N):
            if indices[0][i].item() >= nodeID.size(1):
                merge_examples[:,i,:] = self.examples[:,indices[0][i].item()-nodeID.size(1),:]
                merge_labels[:,i,:] = self.labels[:,indices[0][i].item()-nodeID.size(1),:]
                merge_years[:,i] = self.year[:,indices[0][i].item()-nodeID.size(1)]
            else:
                merge_examples[:,i,:] = examples[:,indices[0][i].item(),:]
                merge_labels[:,i,:] = labels[:,indices[0][i].item(),:]
                merge_years[:,i] = year[:,indices[0][i].item()]
        setattr(self, 'examples', merge_examples.to(self.device))
        setattr(self, 'labels', merge_labels.to(self.device))
        setattr(self, 'year', merge_years.to(self.device))

    def replace_by_score(self, args, examples, labels, nodeID, year, idx, score):
        """
        Replace the buffer data according to the influence score of each time stamp.
        """
        examplesT = torch.cat((self.examples, examples))
        labelsT = torch.cat((self.labels, labels))
        nodeIDT = torch.cat((self.nodeID, nodeID))
        yearT = torch.cat((self.year, year))
        idxT = torch.cat((self.idx, idx))
        scoreT = torch.cat((self.score, score))

        sorted_id = torch.argsort(scoreT,0,descending = True)
        sorted_score = scoreT[sorted_id]
        sorted_idx = idxT[sorted_id]
        sorted_year = yearT[sorted_id]
        sorted_nodeID = nodeIDT[sorted_id]
        sorted_labels = labelsT[sorted_id]
        sorted_examples = examplesT[sorted_id]

        self.examples = sorted_examples[:self.buffer_size,:,:]
        self.labels = sorted_labels[:self.buffer_size,:,:]
        self.nodeID = sorted_nodeID[:self.buffer_size,:]
        self.year = sorted_year[:self.buffer_size,:]
        self.idx = sorted_idx[:self.buffer_size]
        self.score = sorted_score[:self.buffer_size]


        
        
        
        
        



