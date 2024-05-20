import torch
import random
from torch_geometric.utils import degree, k_hop_subgraph, to_dense_adj
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Tuple

def _perturb_features(
        graph: Data,
        p_damp: float,
        p_trunc: float,
    ) -> torch.Tensor:
    '''
    Perturbs the feature vectors of a graph for positive training samples.

    ARGUMENTS:
    graph: PyTorch Geometric graph
    p_damp: value in (0, 1) that controls the degree of feature perturbation
    p_trunc: value in (0, 1) that represents the max probability for masking a dimension

    OUTPUT: the perturbed feature matrix 
    '''

    # compute node centralities (n x 1 tensor)
    num_nodes = graph.x.size(0)
    node_centralities = degree(graph.edge_index[0], num_nodes=num_nodes) + degree(graph.edge_index[1], num_nodes=num_nodes)
    centralities = node_centralities.unsqueeze(1)

    # n x m -> 1 x m tensor
    # (this measures the relative importance of each dimension)
    weights = torch.abs((centralities * graph.x).sum(dim=0))
    weights = torch.log(weights)

    # used for normalization
    max_weight = weights.max()

    # compute probabilities of masking each dimension
    probs = ((max_weight - weights) / max_weight) * p_damp

    # truncate probabilities to avoid heavy perturbation
    probs = torch.clamp(probs, max=p_trunc)

    # apply salt and pepper noise
    perturbed_features = graph.x * (torch.rand(graph.x.size()) >= probs)

    return perturbed_features

def _perturb_topology(
        graph: Data, 
        dropout: float = 0.2,
    ) -> torch.Tensor:
    '''
    Perturbs the topology of a graph for positive training samples.

    ARGUMENTS:
    graph: PyTorch Geometric graph
    dropout: value in (0, 1) that is the fraction of edges to drop

    OUTPUT: the perturbed edge index matrix of the graph
    '''

    # compute node centralities
    num_nodes = graph.x.size(0)
    node_degrees = degree(graph.edge_index[0], num_nodes=num_nodes) + degree(graph.edge_index[1], num_nodes=num_nodes)

    # compute centrality scores for each edge (average of centrality scores for endpoints)
    ec_scores = (node_degrees[graph.edge_index[0]] + node_degrees[graph.edge_index[1]]) / 2

    # sort the centrality scores in descending order
    _, sorted_indices = torch.sort(ec_scores, descending=True)
    sorted_edge_index = graph.edge_index[:, sorted_indices]

    # keep only the (1-dropout)-fraction of edges
    portion_to_keep = int((1-dropout)*sorted_edge_index.size(1))

    return sorted_edge_index[:, :portion_to_keep]
class GraphPairsDataset(Dataset):
    def __init__(self, 
                 graphs: Dataset, 
                 k: int, 
                 p_damp: float, 
                 p_trunc: float, 
                 dropout: float
        ) -> None:
        '''
        Dataset that dynamically generates positive and negative training samples.

        ARGUMENTS:
        graphs: dataset of PyTorch Geometric graphs
        k: k-hop neighborhood sizes
        p_damp, p_trunc, dropout: hyperparameters for feature/topology perturbation for 
        positive training samples

        OUTPUT: None
        '''
        self.k = k
        self.p_damp = p_damp
        self.p_trunc = p_trunc
        self.dropout = dropout
        self.graphs = graphs

        # positive pair: (True, graph, node idx)
        # negative pair: (False, graph, (node idx 1, node idx 2))
        positive = set()
        negative = set()
        for graph in graphs:
            for node_idx in range(graph.num_nodes):
                positive.add((True, graph, node_idx))
            
            # any nodes that are not in the same k-hop neighborhood are
            # candidates for negative samples
            g_adj_mat = to_dense_adj(graph.edge_index)[0]
            g_adj_mat.fill_diagonal_(1)

            # compute g_adj_mat^k: any node that is in the k-hop neighborhood of another node 
            # will be in its neighborhood in the k_hop_adj_mat
            k_hop_adj_mat = g_adj_mat
            for _ in range(1, k):
                k_hop_adj_mat = torch.matmul(k_hop_adj_mat, g_adj_mat)

            # add non-neighboring nodes from g_ad_mat^k as negative pairs
            for node_idx in range(graph.num_nodes):
                non_neighbors = torch.nonzero(k_hop_adj_mat[node_idx] == 0, as_tuple=False).squeeze()
                random_non_neighbor_idx = non_neighbors[torch.randperm(len(non_neighbors))[:1]][0]
                negative.add((False, graph, tuple(sorted([node_idx, random_non_neighbor_idx]))))

        # merge the positive and negative pairs and shuffle
        self.samples = list(positive + negative)
        random.shuffle(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[bool, Data, Data]:
        '''
        Gets the (idx)th contrastive pair for training.

        ARGUMENTS:
        idx: the index of the pair

        OUTPUT:
        A tuple indicating whether or not the pair is a positive pair along
        with the pair
        '''
        sample = self.samples[idx]

        if (sample[0]): # positive pair
            graph = sample[1]
            n_idx = sample[2]

            # get k-hop neighborhood around n_idx
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                node_idx=n_idx, 
                num_hops=self.k, 
                edge_index=graph.edge_index, 
                relabel_nodes=True, 
                num_nodes=graph.num_nodes,
            )

            # create a data object for the subgraph
            subgraph = Data(
                x=graph.x[subgraph_nodes],
                edge_index=subgraph_edge_index,
            )

            # get the perturbed graph
            perturbed_subg_topology = _perturb_topology(subgraph, dropout=self.dropout)
            perturbed_subg_features = _perturb_features(subgraph, self.p_damp, self.p_trunc)
            perturbed_subgraph = Data(x=perturbed_subg_features, edge_index=perturbed_subg_topology)

            return True, subgraph, perturbed_subgraph
        else: # negative pair
            graph = sample[1]
            n1_idx = sample[2][0]
            n2_idx = sample[2][1]

            # subgraph 1
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                node_idx=n1_idx, 
                num_hops=self.k, 
                edge_index=graph.edge_index, 
                relabel_nodes=True, 
                num_nodes=graph.num_nodes
            )
            subgraph1 = Data(
                x=graph.x[subgraph_nodes],
                edge_index=subgraph_edge_index,
            )

            # subgraph 2
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                node_idx=n2_idx, 
                num_hops=self.k, 
                edge_index=graph.edge_index, 
                relabel_nodes=True, 
                num_nodes=graph.num_nodes,
            )
            subgraph2 = Data(
                x=graph.x[subgraph_nodes],
                edge_index=subgraph_edge_index,
            )

            return False, subgraph1, subgraph2

def create_loaders(
        dataset: Dataset, 
        train_split: float, 
        val_split: float, 
        batch_size: float
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Generates dataloaders for training, validation, and testing.

    ARGUMENTS:
    dataset: a PyTorch Geometric Dataset
    train_split: fraction of samples to use for training
    val_split: fraction of samples to use for validation
    batch_size: the batch size

    OUTPUT: 
    dataloaders for training, validation, and testing
    '''
    # training, validation, and test dataset indices
    train_split_idx = int(train_split * len(dataset))
    val_split_idx = train_split_idx + int(val_split * len(dataset))

    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_split_idx]
    val_indices = indices[train_split_idx: val_split_idx]
    test_indices = indices[val_split_idx:]

    # create the dataset subsets (note: this doesn't actually load the graphs)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader, test_loader

def contrastive_loss(
        out1: torch.Tensor, 
        out2: torch.Tensor, 
        labels: torch.Tensor, 
        margin=1.0
    ) -> torch.Tensor:
    '''
    Computes the contrastive loss over two vector embeddings.

    ARGUMENTS:
    out1, out2: the vector embeddings
    labels: indicates whether each pair is a positive or negative contrastive pair
    margin: minimum expected distance between negative pairs

    OUTPUT: 
    the contrastive loss
    '''
    distances = F.pairwise_distance(out1, out2)

    positive_loss = labels * distances.pow(2)
    negative_loss = (1 - labels) * F.relu(margin - distances).pow(2)

    loss = 0.5 * torch.mean(positive_loss + negative_loss)
    return loss


if __name__ == '__main__':
    pass