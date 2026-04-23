import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import numpy as np
import torch

from typing import Optional
from torch import Tensor

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from .registry import register_model

### Modified the Processor by:
# - using 'add' aggregation instead of 'mean'
# - using LayerNorm instead of BatchNorm
# - using 3 layers instead of 5

class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads,
                              dropout=dropout, aggr='add',
                              add_self_loops=True, bias=True)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.act = nn.ReLU()
        # we need same shape to apply skip connection as x + h
        self.rescale = None
        if in_dim != out_dim * heads:
            self.rescale = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        if self.rescale is not None:
            x = self.rescale(x)
        x = x + h          # residual
        x = self.norm(x)
        x = self.act(x)
        return x

class Processor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.block1 = GATBlock(hidden, 32, heads=2, dropout=0.2)
        self.block2 = GATBlock(64, 32, heads=2, dropout=0.2)
        self.block3 = GATBlock(64, 32, heads=2, dropout=0.2)
        # self.block4 = GATBlock(64, 32, heads=2, dropout=0.2)
        # self.block5 = GATBlock(64, 32, heads=2, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        # x = self.block4(x, edge_index)
        # x = self.block5(x, edge_index)
        return x


@register_model("GNN4CD_model")
class GNN4CD_model(nn.Module):

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--x_low_encoding_dim", type=int, default=128)
        parser.add_argument("--history_length", type=int, default=4)
        parser.add_argument("--rnn_input_dim", type=float, default=0.1)
        parser.add_argument("--rnn_n_layers", type=float, default=0.1)
        parser.add_argument("--x_high_dim", type=float, default=0.1)
        parser.add_argument("--x_low2high_dim", type=float, default=0.1)
        return parser
    
    def __init__(
        self,
        x_low_encoding_dim=128,
        history_length=24,
        rnn_input_dim=3*5,
        rnn_hidden_dim=3*5,
        rnn_n_layers=2,
        x_high_dim=6+1,
        x_low2high_dim=64,
        output_dim=1):

        super(GNN4CD_model, self).__init__()

        self.seq_length = history_length + 1

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(rnn_input_dim, rnn_hidden_dim, rnn_n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(rnn_hidden_dim*seq_length, dense_encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((x_low_encoding_dim, x_high_dim), out_channels=x_low2high_dim, aggr='mean'), 'x, edge_index -> x')
            ])
        
        self.processor = Processor(64)
    
        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
            )

    def forward(self, data):
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        # print(f"encod_rnn: {encod_rnn.shape}, data.x_dict['high']: {data.x_dict['high'].shape}, data['low', 'to', 'high'].edge_index: {data['low', 'to', 'high'].edge_index.shape}")
        encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data['low', 'to', 'high'].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        x_high = self.predictor(encod_high)
        return x_high