import torch
from typing import Tuple, Dict, List
from torch_geometric.data import Dataset, Data
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
        
class SubgraphLoader:
    '''
    Utilities for generating subgraphs from a dataset of graphs (used for training GraphSAGE).
    '''
    def __init__(self, graphs: Dataset, k: int) -> None:
        self.graphs = graphs
        self.k = k
        self.subgraph_dataset = None
        self.superset_dataset = None
        self.mapping = None

    def _perturb_features(
            graph: Dict[int, List[int]],
            features: torch.Tensor,
            positive: bool,
            p_f: float,
            p_t: float,
        ) -> torch.Tensor:
        '''Perturbs the feature vectors of a graph; the degree of perturbation of each
        dimension is determined using degree centrality.

        Arguments:
        graph: a graph
        feature: the node features of the graph
        positive: whether to create a positive or negative training sample perturbation
        p_f: a hyperparameter for controlling the degree of perturbation
        p_t: maximum probability for masking an entry in a feature vector
        '''

        # degree centralities
        centralities = [len(graph[n]) for n in graph]

        # `weights` is a 1 x m tensor
        col_wise_sum = torch.abs(features.sum(dim=0))
        weights = col_wise_sum * centralities
        max_weight = weights.max().item()

        # use log the avoid heavily perturbing nodes w/ dense connections
        weights = torch.log(weights)

        # normalize the weights
        probs = ((max_weight - weights) / max_weight) * p_f
        probs = torch.min(probs, p_t)

        # negate if positive = False
        probs = probs if positive else 1 - probs

        # apply salt and pepper noise
        rand_matrix = torch.randn(features.size())
        mask = rand_matrix >= probs
        features *= mask

        return features

    def _perturb_topology(
            graph: Dict[int, List[int]], 
            positive: bool,
            p_e: float,
            p_t: float,
        ) -> Dict[int, List[int]]:
        '''
        Generates a perturbed view of the graph, based on degree centrality,
        for positive or negative training samples.

        Arguments:
        graph: a graph
        positive: whether to create a positive or negative training sample perturbation
        p_e: a hyperparameter for controlling the degree of perturbation
        p_t: maximum probability for deleting an edge
        '''

        # key = ordered tuple (by nodes)
        # value = weight (average degree centrality of endpoints)
        max_weight = None
        edge_weights = dict()
        for n1 in graph:
            for n2 in graph.get(n1):
                e = sorted([n1, n2])
                w = edge_weights.get(e, 0)
                w += len(graph.get(n1)) # add the degree
                edge_weights[e] = w

                # we need to store the maximum weight for normalization (later on)
                if max_weight == None:
                    max_weight = w
                else:
                    max_weight = max(max_weight, w)

        for e in edge_weights:
            # using log alleviates the impact of nodes w/ heavily dense connections
            w = math.log(edge_weights[e] / 2)

            # normalize (truncates to p_t to prevent the perturbation from being too heavy)
            normalized_w = min(((max_weight - w) * p_e) / max_weight, p_t)
            normalized_w = normalized_w if positive else 1 - normalized_w

            edge_weights[w] = normalized_w
        
        # MST using Kruskal's algorithm
        # flatten the edges into a list where each entry is formatted as (weight, tail, head)
        edges = []
        for e in edge_weights:
            edges.append([edge_weights[e], e[0], e[1]])
        
        mst_edges, non_mst_edges = kruskals(edges)
        
        final_edges = mst_edges
        # add back non_mst_edges with probabilities equal to their weights
        for e in non_mst_edges:
            p = random.random()
            if p >= e[0]: # lower probability -> should be added with greater liklihood
                final_edges.append(e)
        
        # transform back into a dictionary graph
        perturbed_graph = dict()
        for e in final_edges:
            n1 = perturbed_graph.get(e[1], [])
            n2 = perturbed_graph.get(e[2], [])
            n1.append(e[2])
            n2.append(e[1])
            perturbed_graph[e[1]] = n1
            perturbed_graph[e[2]] = n2
                
        return perturbed_graph

    def _torch_graph_to_adj_dict(
            self, 
            graph: Data,
        ) -> Dict[int, List[int]]:
        '''
        Converts a PyTorch Geo graph into an adjacency dictionary.

        Arguments:
        graph: a PyTorch Geo graph
        '''
        # TODO

        adj_dict = dict()
        for i in range(graph.edge_index.size(1)):
            head, tail = graph.edge_index[:, 1]
            head, tail = int(head.item()), int(tail.item())

            neighbors = adj_dict.get(head, [])
            neighbors.append(tail)
            adj_dict[head] = neighbors

        return graph.x, adj_dict

    def _adj_dict_to_tensor(
            self, 
            adj_dict: Dict[int, List[int]],
        ) -> Data:
        '''Converts an adjacency dictionary into a torch tensor (stored in the `edge_index`
        field of a PyTorch geo data object).

        Arguments:
        adj_dict: an adjacency dictionary
        '''

        # TODO
        num_edges = sum(len(edges) for edges in adj_dict.values())
        edge_index = torch.tensor(2, num_edges)

        idx = 0
        for head in adj_dict:
            for tail in adj_dict[head]:
                edge_index[0][idx] = head
                edge_index[1][idx] = tail
                idx += 1

        return edge_index
    
    def _bfs(self, 
            start: int,
            graph: Dict[int, List[int]],
        ) -> Dict[int, List[int]]:
        '''Returns the subgraph, with a maximum depth of self.k, centered at the `start` node.

        Arguments:
        start: the starting node
        graph: a graph
        '''
        q = Queue()
        visited = set()
        q.put((start, 0))

        while len(q) > 0:
            n, ply = q.get()

            if n in visited or ply > self.k:
                continue

            visited.add(n)
            for neighbor in graph.get(n):
                q.put((neighbor, ply+1))
        
        subgraph = dict()
        for n in visited:
            subgraph_neighbors = []
            for neighbor in graph.get(n):
                if neighbor in visited:
                    subgraph_neighbors.append(n)
            
            subgraph[n] = subgraph_neighbors
        
        return subgraph

    def generate_samples(
            self, 
            max_positive_subgraphs: int,
            max_negative_subgraphs: int,
            p_f: float,
            p_e: float,
            p_t: float,
        ) -> Tuple[Dataset, Dataset, List[Tuple[int, int]]]:
        '''Generates positive and negative training samples; returns a loader for subgraphs, a loader
        for superset graphs, and a mapping from subgraph to superset nodes (by index).
        
        Arguments:
        max_positive_subgraphs: max umber of positive samples for each graph
        max_negative_subgraphs: max number of negative samples for each graph
        p_f, p_e, p_t: hyperparameters for perturbation step
        '''

        for g in self.graphs:
            # TODO:
            # Generate a positive sample:
            # 1. select an arbitrary node in g
            # 2. run a random BFS traversal from that node
            # 3. perturb the subgraph (w/ positive=True) and add it to the dataset
            # make sure to perturb the topology and node features!

            # Generate a negative sample:
            # 1. select an arbitrary node in g
            # 2. run a random BFS traversal from that node
            # 3. perturb the subgraph (w/ positive=False) and add it to the dataset
            # make sure to perturb the topology and node features!

            # don't forget to update self.subgraph_dataset, self.superset_dataset, and self.mapping
            pass