import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Batch
from torch.utils.data import Sampler

import torch
from typing import Sequence, Union
from torch_geometric.utils import degree
import copy

import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

class Dataset_Graph(Dataset):

    def __init__(
        self,
        graph: Union[HeteroData, None],
        low_input: Sequence[Union[torch.tensor, None]],
        high_input: Sequence[Union[torch.tensor, None]],
        target: Sequence[Union[torch.tensor, None]],
        history_length: int,
        **kwargs: Sequence[torch.tensor]
    ):
        self.graph = graph
        self.low_input = low_input
        self.high_input = high_input
        self.target = target
        self.history_length = history_length
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        # self._check_temporal_consistency()
        self._set_graph_static_high_x(high_input)
        #self._add_node_degree()

    def __len__(self):
        if self.target is not None:
            return len(self.target)
        
    # def _check_temporal_consistency(self):
    #     if self.target is not None:
    #         assert (self.low_input.shape[1]-self.history_length) == self.target.shape[1], "Temporal dimension inconsistency."

    # def _set_snapshot_count(self):
    #     self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, idx: int):
        x_low = self.low_input[:,idx-self.history_length:idx+1,:]
        # x_low = x_low.flatten(start_dim=2, end_dim=-1)            
        return x_low

    def _get_target(self, idx: int):
        return self.target[:,idx] # num nodes, time

    def _get_train_mask(self, target: torch.tensor):
        return ~torch.isnan(target)
    
    def _get_additional_feature(self, idx: int, feature_key: str):
        feature = getattr(self, feature_key)[:,idx]
        return feature
    
    def _get_additional_features(self, idx: int):
        additional_features = {
            key: self._get_additional_feature(idx, key)
            for key in self.additional_feature_keys
        }
        return additional_features
    
    def _set_graph_static_high_x(self, x):
        self.graph["high"].x = x

    def set_additional_features(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
    
    def __getitem__(self, idx: int):

        snapshot = copy.deepcopy(self.graph)

        # Get the dynamic initial node features (low input and target)
        # The static initial node features (elevation, land use) were 
        # already in the self.graph HeteroData object
        snapshot['low'].x = self._get_features(idx)
        y = self._get_target(idx) if self.target is not None else None
        train_mask = self._get_train_mask(y) if y is not None else None
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        
        snapshot.num_nodes = self.graph.num_nodes
        snapshot.idxs = idx

        additional_features = self._get_additional_features(idx)
        for key, value in additional_features.items():
            if value.shape[0] == self.graph['high'].x.shape[0]:
                snapshot['high'][key] = value
            elif value.shape[0] == self.graph['low'].x.shape[0]:
                snapshot['high'][key] = value

        #snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     
        #snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

        return snapshot
    

# class Sampler_Graph(Sampler):
#     def __init__(
#             self, shuffle, dataset, idxs_vector=None, skip=0):
        
#         self.shuffle = shuffle
#         self.idxs_vector = idxs_vector
#         self.dataset_len = len(dataset)
#         self.skip = skip
#         self.idx_max = self.dataset_len + skip

#         # build full index list
#         if self.idxs_vector is not None:
#             indices = self.idxs_vector.clone()
#         else:
#             indices = torch.arange(self.skip, self.idx_max)

#         if self.shuffle:
#             perm = torch.randperm(len(indices))
#             indices = indices[perm]

#         self.indices = indices

#     def __iter__(self):
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)

def custom_collate_fn_graph(batch_list):
    return Batch.from_data_list(batch_list)


