import torch
from typing import Tuple, Dict, Any, List
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import PPI, WordNet18RR
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
        that node x is part of.'''
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        '''Merges the subgraphs that nodes x and y are part of.'''
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
    
    def _perturb_graph(
            graph: Dict[int, List[int]], 
            positive: bool,
            p_e: float,
            p_t: float,
        ) -> Dict[int, List[int]]:
        '''
        Generates a perturbed view of the graph, based on degree centrality,
        for positive or negative training samples.
        '''

        # key = ordered tuple (by nodes)
        # value = weight (average degree centrality of endpoints)
        max_weight = None
        edge_weights = dict()
        for n1 in graph:
            for n2 in graph.get(n1):
                e = sorted([n1, n2])
                w = edge_weights.get(e, 0)
                w += len(len(graph.get(n1)))
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

    def _store_torch_graph_as_dict(
            self, 
            graph: Data,
        ) -> Dict[int, int]:
        pass

    def _store_dict_as_torch_graph(
            self, 
            graph: Dict[int, int],
        ) -> Data:
        pass
    
    def _bfs(self, 
            start: int,
            graph: Dict[int, List[int]], 
            depth: int
        ) -> Dict[int, List[int]]:
        '''Returns the subgraph, with a maximum depth, centered at the `start` node.

        Arguments:
        start: the starting node
        graph: the graph
        depth: maximum depth for BFS traversal
        '''
        q = Queue()
        visited = set()
        q.put((start, 0))

        while len(q) > 0:
            n, ply = q.get()

            if n in visited or ply > depth:
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
            positive_subgraphs: int,
            negative_subgraphs: int,
        ) -> Tuple[Dataset, Dataset, List[Tuple[int, int]]]:
        '''Generates positive training samples; returns a loader for subgraphs, a loader
        for superset graphs, and a mapping from subgraph to superset nodes (by index).
        
        Arguments:
        positive_subgraphs: number of positive samples for each graph
        negative_subgraphs: number of negative samples for each graph
        '''

        '''
        Generating positive samples:
        1. G' <- run a random BFS from a node in g
        2. G'' <- perturbation on G' w/ positive = True
        3. Add (G', G'')

        Generating negative samples:
        1. G' <- run a random BFS from a node in g
        2. G'' <- perturbation on G' w/ positive = False
        3. Add (G', G'')
        '''

        dataset = []
        for g in self.graphs:
            g_dict = self._store_torch_graph_as_dict(g)
            subgraph_dict = None
            pass

    def generate_negative_samples(
            self, 
            num_subgraphs: int,
        ) -> Tuple[Dataset, Dataset, List[Tuple[int, int]]]:
        '''Generates negative training samples; returns a loader for subgraphs, a loader
        for superset graphs, and a mapping from subgraph to superset nodes (by index).
        
        Arguments:
        num_subgraphs: maximum number of subgraphs to generate from each graph
        '''
        pass