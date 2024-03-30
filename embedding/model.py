import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k):
        super(GraphSAGE, self).__init__()
        # neighborhood sampling depth
        self.k = k

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # first layer
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, aggr='mean')
        )

        # remaining layers
        for _ in self.k:
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            )

        # linear layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # apply each layer
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # linear layer (applied to each node)
        x = self.lin(x)

        return x