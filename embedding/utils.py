import torch
from torch_geometric.utils import degree, k_hop_subgraph, to_dense_adj
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

def perturb_features(
        graph: Data,
        p_damp: float,
        p_trunc: float,
    ) -> torch.Tensor:
    '''Perturbs the feature vectors of a graph for positive training samples.

    Arguments:
    graph: PyTorch Geo graph
    p_damp: controls degree of perturbation
    p_trunc: max probability for masking a dimension
    '''

    # compute node centralities (n x 1 tensor)
    num_nodes = graph.x.size(0)
    node_centralities = degree(graph.edge_index[0], num_nodes=num_nodes) + degree(graph.edge_index[1], num_nodes=num_nodes)
    centralities = node_centralities.clone().unsqueeze(1)

    # n x m -> 1 x m tensor
    # this measures the relative importance of each dimension
    weights = torch.abs((centralities * graph.x).sum(dim=0))
    weights = torch.log(weights)

    # used for normalization
    max_weight = weights.max()

    # compute salt-pepper probability for each dimension
    # (features w/ greater weight should have lower probabilities)
    probs = ((max_weight - weights) / max_weight) * p_damp

    # truncate probabilities to avoid heavy perturbation
    probs = torch.clamp(probs, max=p_trunc)

    # apply salt and pepper noise
    mask = torch.rand(graph.x.size()) >= probs
    perturbed_features = graph.x * mask

    return perturbed_features

def perturb_topology(
        graph: Data, 
        dropout: float = 0.2,
    ) -> torch.Tensor:

    # compute node degrees
    num_nodes = graph.x.size(0)
    node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes) + degree(graph.edge_index[1], num_nodes=num_nodes)

    # compute centrality scores for each edge (average of centrality scores for endpoints)
    ec_scores = (node_degrees[graph.edge_index[0]] + node_degrees[graph.edge_index[1]]) / 2

    # sort the centrality scores in non-increasing order
    _, sorted_indices = torch.sort(ec_scores, descending=True)

    # sort the edge index as well
    sorted_edge_index = graph.edge_index[:, sorted_indices]

    # keep only the (1-dropout)-portion of edges
    portion_to_keep = int((1-dropout)*sorted_edge_index.size(1))

    return sorted_edge_index[:, :portion_to_keep]

class GraphPairsDataset(Dataset):
    def __init__(self, graphs, k, p_damp, p_trunc, dropout):
        self.k = k
        self.p_damp = p_damp
        self.p_trunc = p_trunc
        self.dropout = dropout
        self.graphs = graphs

        # preprocess to get positive and negative samples

        # positive pair is identified by tuple
        # consisting of True, graph idx, node idx
        positive = set()
        for g_idx, g in enumerate(graphs):
            for node_idx in range(g.num_nodes):
                positive.add((True, g_idx, node_idx))

        # negative pair is identified by tuple
        # consisting of False, graph idx, (node1 idx, node2 idx)
        negative = set()
        for g_idx, g in enumerate(graphs):
            g_adj_mat = to_dense_adj(g.edge_index)[0]
            g_adj_mat.fill_diagonal_(1)

            # compute g_adj_mat^k;
            # any node that is in the k-hop neighborhood of another node will be
            # in its neighborhood in the k_hop_adj_mat
            k_hop_adj_mat = g_adj_mat.clone()
            for _ in range(1, k):
                k_hop_adj_mat = torch.matmul(k_hop_adj_mat, g_adj_mat)

            # add non-neighboring nodes (w.r.t k_hop_adj_mat) as negative pairs
            for node_idx in range(g.num_nodes):
                non_neighbors = torch.nonzero(k_hop_adj_mat[node_idx] == 0, as_tuple=False).squeeze()
                picked = non_neighbors[torch.randperm(len(non_neighbors))[:1]]
                for non_neighbor_idx in picked:
                    negative.add((False, g_idx, tuple(sorted([node_idx, non_neighbor_idx]))))
        
        # merge the positive and negative pairs
        self.samples = list(positive + negative)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # positive pair
        if (sample[0]):
            g_idx = sample[1]
            g = self.graphs[g_idx]
            n_idx = sample[2]

            # get subgraph (k-hop neighborhood around node `n`)
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                node_idx=n_idx, 
                num_hops=self.k, 
                edge_index=g.edge_index, 
                relabel_nodes=True, 
                num_nodes=g.num_nodes
            )

            # create a data object for the subgraph
            subgraph = Data(
                x=g.x[subgraph_nodes],
                edge_index=subgraph_edge_index,
            )

            # get the perturbed graph
            perturbed_subg_topology = perturb_topology(subgraph, dropout=self.dropout)
            perturbed_subg_features = perturb_features(subgraph, self.p_damp, self.p_trunc)
            perturbed_subgraph = Data(x=perturbed_subg_features, edge_index=perturbed_subg_topology)

            return (subgraph, perturbed_subgraph), 1
        else:
            g_idx = sample[1]
            g = self.graphs[g_idx]
            n1_idx = sample[2][0]
            n2_idx = sample[2][1]

            # subgraph 1
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                node_idx=n1_idx, 
                num_hops=self.k, 
                edge_index=g.edge_index, 
                relabel_nodes=True, 
                num_nodes=g.num_nodes
            )
            subgraph1 = Data(
                x=g.x[subgraph_nodes],
                edge_index=subgraph_edge_index,
            )

            # subgraph 2
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                node_idx=n2_idx, 
                num_hops=self.k, 
                edge_index=g.edge_index, 
                relabel_nodes=True, 
                num_nodes=g.num_nodes
            )
            subgraph2 = Data(
                x=g.x[subgraph_nodes],
                edge_index=subgraph_edge_index,
            )

            return (subgraph1, subgraph2), 0

def create_loaders(dataset, train_split, val_split, batch_size):
    # training, validation, and test dataset indices
    train_split_idx = int(train_split * len(dataset))
    val_split_idx = train_split_idx + int((val_split * len(dataset)))

    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_split_idx]
    val_indices = indices[train_split_idx:val_split_idx]
    test_indices = indices[val_split_idx:]

    # create the dataset subsets (note: this doesn't actually load the 
    # graphs)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader, test_loader

def contrastive_loss(out1, out2, labels, margin=1.0):
    distances = F.pairwise_distance(out1, out2)

    positive_loss = labels * distances.pow(2)
    negative_loss = (1 - labels) * F.relu(margin - distances).pow(2)

    loss = 0.5 * torch.mean(positive_loss + negative_loss)
    return loss