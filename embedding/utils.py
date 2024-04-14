import torch
from typing import Tuple, List, Optional, Set
from torch_geometric.utils import degree
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
import random
from queue import Queue

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
    probs = torch.clamp(probs, p_trunc)

    # apply salt and pepper noise
    mask = torch.randn(graph.x.size()) >= probs
    perturbed_features = graph.x * mask

    return perturbed_features

def perturb_topology(
        graph: Data, 
        dropout=0.1,
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

def bfs(
        start: int, 
        k: int, 
        adj_mat: List[Set[int]], 
        features: torch.Tensor, 
        p: float = 0.2,
        min_edges: int = 10,
    ) -> Data:
    '''Returns the subgraph, with a maximum depth of k, centered at 
    some starting node.

    Arguments:
    start: the starting node
    k: maximum depth
    graph: a PyTorch Geo graph
    '''

    q = Queue()
    mapping = dict()
    counter = 0
    q.put((start, 0))

    # find all nodes in the k-hop neighborhood of `start`
    while q.qsize() > 0:
        n, ply = q.get()

        if n in mapping or ply > k:
            continue

        mapping[n] = counter
        counter += 1

        neighbors = list(adj_mat[n])
        truncated_neighbors = neighbors
        if len(neighbors) > min_edges:
            truncated_neighbors = neighbors[:int(p*len(neighbors))]
        
        for neighbor in truncated_neighbors:
            q.put((neighbor, ply+1))
    
    edge_index = [[], []]
    for n in mapping:
        for neighbor in adj_mat[n]:
            if neighbor not in mapping:
                continue
            edge_index[0].append(mapping[n])
            edge_index[1].append(mapping[neighbor])
    
    # edge_index
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # features matrix (x)
    idxs_float = torch.Tensor(sorted(list(mapping.keys())))
    idxs_long = idxs_float.to(dtype=torch.long)

    subgraph = Data(x=features[idxs_long], edge_index=edge_index)
    return subgraph

def generate_samples(
        graphs: Dataset,
        k: int,
        p_damp: float,
        p_trunc: float,
        dropout: float = 0.1,
    ) -> List[Tuple[Data, Data]]:
    '''Generates positive or negative training samples; returns a list of 
    corresponding graphs.
    
    Arguments:
    graphs: original dataset of graphs
    k: maximum depth of subgraph training samples
    max_subgraphs: max number of subgraphs to sample from each graph (for 
    positive and negative training samples)
    p_f, p_e, p_t: hyperparameters for perturbations
    '''

    data = []
    counter = 0

    for g in graphs:
        adj_mat = [set() for _ in range(g.x.size(0))]
        for i in range(g.edge_index.shape[1]):
            tail = g.edge_index[0, i]
            head = g.edge_index[1, i]
            adj_mat[tail].add(head)

        
        num_nodes = len(adj_mat)
        nodes = list(range(num_nodes))
        random.shuffle(nodes)

        for n in nodes:
            
            # get subgraph (k-hop neighborhood around node `n`)
            subg = bfs(n, k, adj_mat, g.x)

            # perturb topology and features
            perturbed_subg_topology = perturb_topology(subg, dropout=dropout)
            perturbed_subg_features = perturb_features(subg, p_damp, p_trunc)
            perturbed_subg = Data(x=perturbed_subg_features, edge_index=perturbed_subg_topology)

            # add pair
            data.append((perturbed_subg, subg))

            print(f"Positive pair: {counter}")
            counter += 1
        
        for idx in range(len(nodes)): # negative training samples
            subg1 = bfs(nodes[idx], k, adj_mat, g.x)

            idx2 = len(nodes) - 1 - idx
            subg2 = bfs(nodes[idx2], k, adj_mat, g.x)

            data.append((subg1, subg2))
            
            print(f"Negative pair: {counter}")
            counter += 1

    return data

def save_data(
        data: List[Tuple[Data, Data]], 
        folder: str
    ) -> None:
    '''Saves data to the folder.
    
    Arguments:
    data: data
    folder: folder path
    '''
    
    counter = 0
    for d1, d2 in data:
        dir_path = f"{folder}/p-{counter}"
        counter += 1

        # check if the directory exists
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # save
        full_path_d1 = os.path.join(dir_path, 'd1.pt')
        full_path_d2 = os.path.join(dir_path, 'd2.pt')
        torch.save(d1, full_path_d1)
        torch.save(d2, full_path_d2)

def read_data(folder: str) -> List[Tuple[Data, Data]]:
    '''Reads data from the folder.
    
    Arguments:
    folder: folder path
    '''

    data = []
    for root, _, files in os.walk(folder):
        if (root == folder):
            continue
        
        if len(files) != 2:
            continue

        d1_path = os.path.join(root, files[0])
        d2_path = os.path.join(root, files[1])

        pair = [torch.load(d1_path), torch.load(d2_path)]
        data.append(pair)
    
    return data

def load(
        data: List[Tuple[Data, Data]],
        train_split: float,
        val_split: float,
        bs: int,
        shuffle: Optional[bool] = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Returns dataloaders for training, validation, and test datasets.
    
    Arguments:
    data: a list of corresponding subgraphs
    train_split: portion of data to allocate for training
    val_split: portion of data to allocate for validation
    bs: batch size
    shuffles: whether to shuffle the data
    '''

    if shuffle:
        random.shuffle(data)

    size = len(data)
    train_idx = int(train_split*size)
    val_idx = train_idx + int(val_split*size)

    train_loader = DataLoader(data[0:train_idx], batch_size=bs, shuffle=False)
    val_loader = DataLoader(data[train_idx:val_idx], batch_size=bs, shuffle=False)
    test_loader = DataLoader(data[val_idx:], batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader