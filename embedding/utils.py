import torch
from typing import Tuple, Dict, List, Optional
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
import random
from queue import Queue

class UnionFind:
    '''UnionFind data structure for Kruskal's algorithm.'''
    def __init__(self, num_nodes: int) -> None:
        self.parent = [i for i in range(num_nodes)]
        self.rank = [0] * num_nodes
    
    def find(self, x: int) -> int:
        '''Returns the parent (aka representative) node of the subgraph
        that x is part of.
        
        Arguments:
        x: a node
        '''
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        '''Merges the subgraphs that x and y are part of.
        
        Arguments:
        x: a node
        y: a node
        '''
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += 1
        
        return True

def kruskals(
        edges: List[Tuple[float, int, int]],
        num_nodes: int,
    ) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]]]:
    '''Kruskal's algorithm for computing a minimum spanning tree (MST); returns a tuple
    consisting of the edges in the MST and NOT in the MST.
    
    Arguments:
    edges: a list of edges represented as tuples (weight, tail, head)
    num_nodes: the number of vertices in the graph
    '''

    mst_edges = []
    non_mst_edges = []
    uf = UnionFind(num_nodes)
    sorted_edges = sorted(edges, key=lambda x: x[0]) # sort by weight

    for e in sorted_edges:
        if uf.union(e[1], e[2]): # true if they are NOT connected
            mst_edges.append(e)
        else:
            non_mst_edges.append(e)
    
    return mst_edges, non_mst_edges
        
def compute_node_centralities(adj_mat: List[List[int]]) -> List[int]:
    '''Computes a centrality score for each node in the graph. This is based on
    the degree of a node.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    '''

    return [sum(n) for n in adj_mat]

def compute_edge_centralities(adj_mat: List[List[int]]) -> Dict[Tuple[int, int], float]:
    '''Computes a centrality score for each edge in the graph. This is based on the
    average centrality of the endpoints in an undirected graph or the centrality of the head
    of the edge in a directed graph.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    '''

    edge_centralities = dict()

    for n, neighbors in enumerate(adj_mat):
        for n2 in neighbors:
            e = tuple(sorted([n, n2]))
            edge_centralities[e] = edge_centralities.get(e, 0) + (len(adj_mat[n2]) / 2)
    
    return edge_centralities

def perturb_features(
        adj_mat: List[List[int]],
        features: torch.Tensor,
        p_f: float,
        p_t: float,
    ) -> torch.Tensor:
    '''Perturbs the feature vectors of a graph for positive training samples.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    features: the node features of the graph
    p_f: a hyperparameter for controlling the degree of perturbation
    p_t: maximum probability for masking an entry in a feature vector
    '''

    # record centralities as a n x 1 tensor
    centralities = compute_node_centralities(adj_mat)
    centralities = torch.tensor(centralities).unsqueeze(1)

    # n x m -> 1 x m tensor
    # this measures the relative importance of each dimension
    weights = torch.abs((centralities * features).sum(dim=0))

    # used for normalization
    max_weight = weights.max().item()

    # compute salt-pepper probability for each dimension
    # (features w/ greater weight should have lower probabilities)
    probs = ((max_weight - weights) / max_weight) * p_f

    # truncate probabilities to avoid heavy perturbation
    probs = torch.min(probs, p_t)

    # apply salt and pepper noise
    mask = torch.randn(features.size()) >= probs
    perturbed_features = features * mask

    return perturbed_features

def perturb_topology(
        adj_mat: List[List[int]],
        p_e: float,
        p_t: float,
    ) -> List[List[int]]:
    '''
    Perturbs the topology of the graph, based on degree centrality,
    for positive training samples.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    p_e: a hyperparameter for controlling the degree of perturbation
    p_t: maximum probability for deleting an edge
    '''

    edge_centralities = compute_edge_centralities(adj_mat)
    max_weight = max([edge_centralities[e] for e in edge_centralities])
    edges = []
    for e in edge_centralities:
        w = edge_centralities[e]
        # normalize (truncates to p_t to prevent the perturbation from being too heavy)
        normalized_w = min(((max_weight - w) * p_e) / max_weight, p_t)

        edges.append((normalized_w, e[0], e[1]))
    
    # MST using Kruskal's algorithm
    mst_edges, non_mst_edges = kruskals(edges)

    # add back non_mst_edges with probabilities equal to their 1 - weights
    # (ie. lower probability -> should be added with greater liklihood)
    final_edges = mst_edges
    for e in non_mst_edges:
        if random.random() >= e[0]:
            final_edges.append(e)
    
    # transform back into an adjacency matrix
    perturbed_graph = [[] for _ in range(len(adj_mat))]

    for e in final_edges:
        _, n, n2 = e
        perturbed_graph[n].append(n2)

        # check if the edge is bidirectional
        if n in adj_mat[n2]:
            perturbed_graph[n2].append(n)
            
    return perturbed_graph

def graph_to_adj_mat(graph: Data) -> List[List[int]]:
    '''
    Converts a PyTorch Geo graph into an adjacency matrix.

    Arguments:
    graph: a PyTorch Geo graph
    '''

    adj_mat = [[] for _ in range(graph.x.size(0))]

    for i in range(graph.edge_index.size(1)): # iterate over each edge
        tail, head = graph.edge_index[:, i]
        tail, head = int(tail.item()), int(head.item())
        adj_mat[tail].append(head)

    return adj_mat

def adj_mat_to_graph(x: torch.Tensor, adj_mat: List[List[int]]) -> Data:
    '''Converts an adjacency matrix into a PyTorch Geo graph.

    Arguments:
    x: node features tensor
    adj_mat: an adjacency matrix for the graph
    '''

    num_edges = sum(len(neighbors) for neighbors in adj_mat)
    edge_index = torch.zeros(2, num_edges)

    idx = 0
    for tail, neighbors in enumerate(adj_mat):
        for head in neighbors:
            edge_index[0][idx] = tail
            edge_index[1][idx] = head
            idx += 1

    return Data(x=x, edge_index=edge_index)

def bfs(start: int, k: int, graph: Data) -> Tuple[List[List[int]], torch.Tensor]:
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

        for neighbor in graph.edge_index[1][graph.edge_index[0] == n]:
            q.put((int(neighbor.item()), ply+1))
    
    subgraph = [[] for _ in range(len(mapping))]
    for n in mapping:
        # add neighbors that are ONLY in the `mapping`
        for neighbor in graph.edge_index[1][graph.edge_index[0] == n]:
            if neighbor in mapping:
                subgraph[mapping[n]] = mapping[neighbor]
    
    idxs = torch.Tensor(sorted(list(mapping.keys())))
    return subgraph, graph.x[idxs]

def generate_samples(
        graphs: Dataset,
        k: int,
        max_subgraphs: int,
        p_f: float,
        p_e: float,
        p_t: float,
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

    for g in graphs:

        size = g.size(0)
        nodes = list(range(n))
        rand_nodes_pos = random.sample(nodes, min(size, max_subgraphs))

        for n in rand_nodes_pos: # positive training samples
            sub_adj_mat, features = bfs(n, k, g)

            perturbed_sub_adj_mat = perturb_topology(sub_adj_mat, p_e, p_t)
            perturbed_features = perturb_features(sub_adj_mat, features, p_f, p_t)

            data.append(
                (
                    adj_mat_to_graph(perturbed_features, perturbed_sub_adj_mat),
                    adj_mat_to_graph(features, sub_adj_mat)
                )
            )
        
        rand_nodes_neg = random.sample(nodes, min(size, max_subgraphs*2))
        for idx in range(len(rand_nodes_neg)//2): # negative training samples
            sub_adj_mat, features = bfs(rand_nodes_neg[idx], k, g)

            idx2 = len(rand_nodes_neg) - 1 - idx
            sub_adj_mat2, features2 = bfs(rand_nodes_neg[idx2], k, g)

            data.append(
                (
                    adj_mat_to_graph(features, sub_adj_mat), 
                    adj_mat_to_graph(features2, sub_adj_mat2)
                )
            )

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