import torch
from typing import Tuple, Dict, List, Optional
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import math
import random
from queue import Queue

class UnionFind:
    '''UnionFind data structure for Kruskal's algorithm.'''
    def __init__(self, num_vertices: int) -> None:
        self.parent = [i for i in range(num_vertices)]
        self.rank = [0] * num_vertices
    
    def find(self, x: int) -> int:
        '''Returns the parent (aka representative) vertex of the subgraph
        that node x is part of.
        
        Arguments:
        x: a node
        '''
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        '''Merges the subgraphs that nodes x and y are part of.
        
        Arguments:
        x: a node
        y: a node
        '''
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += 1

def kruskals(
        edges: List[Tuple[float, int, int]],
        num_vertices,
    ) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]]]:
    '''Kruskal's algorithm for computing a minimum spanning tree (MST); returns the list
    of edges in the MST and NOT in the MST
    
    Arguments:
    edges: a list of edges represented as tuples consisting of weight, tail, head
    num_vertices: the number of vertices in the graph
    '''

    mst_edges = []
    non_mst_edges = []
    uf = UnionFind(num_vertices)
    sorted_edges = sorted(edges, key=lambda x: x[0]) # sort by weight

    for e in sorted_edges:
        if uf.find(e[1]) != uf.find(e[2]):
            uf.union(e[1], e[2])
            mst_edges.append(e)
        else:
            non_mst_edges.append(e)
    
    return mst_edges, non_mst_edges

class DatasetWrapper(Dataset):
    def __init__(self, data_lst: List[Data]) -> None:
        super(Dataset, self).__init__()
        self.data_lst = data_lst
    
    def __len__(self) -> int:
        return len(self.data_lst)
    
    def __getitem__(self, idx: int) -> Data:
        return self.data_lst[idx]
        
def compute_node_centralities(adj_mat: List[List[int]]) -> List[int]:
    '''Computes a centrality score for each node in the graph. This is based on the 
    degree of nodes.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    '''

    return [sum(n) for n in adj_mat]

def compute_edge_centralities(adj_mat: List[List[int]]) -> Dict[Tuple[int, int], float]:
    '''Computes a centrality score for each edge in the graph. This is based on the
    average centrality of the endpoints in an undirected graph or the centrality of the head
    of the edge in an directed graph.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    '''

    edge_centralities = dict()

    for n, neighbors in enumerate(adj_mat):
        for n2 in neighbors:
            e = tuple(sorted([n, n2]))
            edge_centralities[e] = edge_centralities.get(e, 0) + 0.5
    
    return edge_centralities

def perturb_features(
        adj_mat: List[List[int]],
        features: torch.Tensor,
        p_f: float,
        p_t: float,
    ) -> torch.Tensor:
    '''Perturbs the feature vectors of a graph for positive training samples; 
    the degree of perturbation of each dimension is determined using degree centrality.

    Arguments:
    adj_mat: an adjacency matrix for a graph
    feature: the node features of the graph
    p_f: a hyperparameter for controlling the degree of perturbation
    p_t: maximum probability for masking an entry in a feature vector
    '''

    # record centralities as a n x 1 tensor
    centralities = compute_node_centralities(adj_mat)
    centralities = torch.tensor(centralities).unsqueeze(1)

    # n x m -> 1 x m tensor
    weights = torch.abs((centralities * features).sum(dim=0))

    max_weight = weights.max().item()

    # use log the avoid heavily perturbing nodes w/ dense connections
    weights = torch.log(weights)

    # normalize the weights
    probs = ((max_weight - weights) / max_weight) * p_f
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
    Generates a perturbed view of the graph, based on degree centrality,
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
        # using log alleviates the impact of nodes w/ heavily dense connections
        w = math.log(edge_centralities[e])

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
    
    # transform back into a dictionary graph
    perturbed_graph = [[] for _ in range(len(adj_mat))]

    for e in final_edges:
        _, n, n2 = e
        perturbed_graph[n].append(n2)

        # check if the edge is bidirectional (ie. if the graph is undirected)
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
        tail, head = graph.edge_index[:, 1]
        tail, head = int(head.item()), int(tail.item())
        adj_mat[tail].append(head)

    return adj_mat

def adj_mat_to_graph(x: torch.Tensor, adj_mat: List[List[int]]) -> Data:
    '''Converts an adjacency matrix into a PyTorch Geo graph.

    Arguments:
    x: node features tensor
    adj_mat: an adjacency matrix
    '''

    num_edges = sum(len(neighbors) for neighbors in adj_mat)
    edge_index = torch.tensor(2, num_edges)

    idx = 0
    for tail, neighbors in enumerate(adj_mat):
        for head in neighbors:
            edge_index[0][idx] = tail
            edge_index[1][idx] = head
            idx += 1

    return Data(x=x, edge_index=edge_index)

def bfs(start: int, k: int, graph: Data) -> Tuple[List[List[int]], torch.Tensor]:
    '''Returns the subgraph, with a maximum depth of self.k, centered at the `start` node.

    Arguments:
    start: the starting node
    k: maximum depth
    graph: a PyTorch Geo graph
    adj_mat: adjacency matrix of the graph
    '''
    # TODO
    # optimize this using only the graph since generating an adj_mat is costly!

    q = Queue()
    mapping = dict()
    counter = 0
    q.put((start, 0))

    while len(q) > 0:
        n, ply = q.get()

        if n in mapping or ply > k:
            continue

        mapping[n] = counter
        counter += 1

        for neighbor in graph.edge_index[1][graph.edge_index[0] == n]:
            q.put((neighbor.item(), ply+1))
    
    subgraph = [[] for _ in range(len(mapping))]
    for n in mapping:
        for neighbor in graph.edge_index[1][graph.edge_index[0] == n]:
            if neighbor in mapping:
                subgraph[mapping[n]] = mapping[neighbor]
    
    idxs = torch.Tensor(sorted(list(mapping.keys())))
    return subgraph, graph.x[idxs]

def generate_samples(
        graphs: Dataset,
        k: int,
        positive_samples_per_graph: int,
        negative_samples_per_graph: int,
        p_f: float,
        p_e: float,
        p_t: float,
    ) -> Tuple[Dataset, Dataset, List[int]]:
    '''Generates positive or negative training samples; returns a list subgraphs and 
    corresponding superset graphs.
    
    Arguments:
    graphs: original dataset of graphs
    k: maximum depth of subgraph training samples
    max_positive_subgraphs: max number of positive samples for each graph
    max_negative_subgraphs: max number of negative samples for each graph
    p_f, p_e, p_t: hyperparameters for perturbation step
    '''

    data_subgraphs = []
    data_supersets = []

    for g in graphs:

        size = g.size(0)
        nodes = list(range(n))
        rand_nodes_pos = random.sample(nodes, min(size, positive_samples_per_graph))

        for n in rand_nodes_pos: # positive training samples
            sub_adj_mat, features = bfs(n, k, g)

            perturbed_sub_adj_mat = perturb_topology(sub_adj_mat, p_e, p_t)
            perturbed_features = perturb_features(sub_adj_mat, features, p_f, p_t)

            # add to the datasets
            data_subgraphs.append(
                adj_mat_to_graph(perturbed_features, perturbed_sub_adj_mat)
            )
            data_supersets.append(
                adj_mat_to_graph(features, sub_adj_mat)
            )
        
        rand_nodes_neg = random.sample(nodes, min(size, negative_samples_per_graph*2))
        for idx, n in range(len(rand_nodes_neg)//2): # negative training samples
            sub_adj_mat, features = bfs(n, k, g)

            idx2 = len(rand_nodes_neg) - idx - 1
            sub_adj_mat2, features2 = bfs(rand_nodes_neg[idx2], k, g)
            data_subgraphs.append(
                adj_mat_to_graph(features, sub_adj_mat)
            )
            data_supersets.append(
                adj_mat_to_graph(features2, sub_adj_mat2)
            )

    return data_subgraphs, data_supersets

def generate_all_samples(
        graphs: Dataset, 
        k: int, 
        max_subgraphs: int,
        p_f: float,
        p_e: float,
        p_t: float,
    ) -> List[Tuple[Data, Data]]:
    '''
    Generates all training samples; returns a dataset of subgraphs and corresponding
    superset graphs.
    '''
    positive_subgraphs, positive_supersets = generate_samples(
        graphs,
        k,
        max_subgraphs,
        True,
        p_f,
        p_e,
        p_t
    )

    negative_subgraphs, negative_supersets = generate_samples(
        graphs,
        k,
        max_subgraphs,
        False,
        p_f,
        p_e,
        p_t
    )

    subgraphs = positive_subgraphs + negative_subgraphs
    supersets = positive_supersets + negative_supersets

    out = []
    for i, subgraph in enumerate(subgraphs):
        out.append((subgraph, supersets[i]))

    return out

def load(
        data: List[Tuple[Data, Data]],
        train_split: float,
        val_split: float,
        bs: int,
        shuffle: Optional[bool] = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if shuffle:
        random.shuffle(data)

    size = len(data)
    train_idx = int(train_split*size)
    val_idx = train_idx + int(val_split*size)

    train_loader = DataLoader(data[0:train_idx], batch_size=bs, shuffle=False)
    val_loader = DataLoader(data[train_idx:val_idx], batch_size=bs, shuffle=False)
    test_loader = DataLoader(data[val_idx:], batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader