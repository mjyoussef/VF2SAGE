import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            hidden_channels: int, 
            out_channels: int, 
            k: int,
            dropout: float = 0.1
        ) -> None:

        super(GraphSAGE, self).__init__()
        # neighborhood sampling depth
        self.k = k

        # GraphSAGE layers
        self.convs = nn.ModuleList()

        # dropout layers
        self.dropouts = nn.ModuleList()

        for _ in k:
            self.dropouts.append(
                Dropout(p=dropout)
            )

        # first layer
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, aggr='mean')
        )

        # remaining layers
        for _ in self.k-1:
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            )

        # linear layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            # GraphSAGE layer
            x = F.relu(conv(x, edge_index))

            # dropout
            x = self.dropouts[i](x)

        # linear layer (applied to each node)
        x = self.lin(x)

        return x