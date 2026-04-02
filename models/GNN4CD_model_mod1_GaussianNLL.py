import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import torch

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

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        return x


class GNN4CD_model_mod1_GaussianNLL(nn.Module):
    
    def __init__(self, encoding_dim=128, seq_length=25, h_in=5*5, h_hid=5*5, n_layers=2, high_in=6+1, low2high_out=64, high_out=64):
        super(GNN4CD_model_mod1_GaussianNLL, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_length, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 'x, edge_index -> x')
            ])
        
        self.processor = Processor(low2high_out)
    
        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, data):
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data['low', 'to', 'high'].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        out = self.predictor(encod_high)
        mu, log_sigma = out[:, 0], out[:, 1]
        # ensure positivity and avoid zero
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-4
        return mu, sigma
    