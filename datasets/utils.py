import torch
from typing import Tuple
from torch_geometric.data import Dataset
from torch_geometric.datasets import PPI, WordNet18RR
from torch_geometric.loader import DataLoader

class SubgraphLoader():
    '''
    Utilities for generating subgraphs from a dataset of graphs (used for training GraphSAGE).
    '''
    def __init__(self, graphs: Dataset, max_nodes: int, max_edges: int) -> None:
        self.graphs = graphs
        self.max_nodes = max_nodes
        self.max_edges = max_edges
    
    def generate_positive_samples(
            self, 
            num_subgraphs: int
        ) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
        '''Generates positive training samples; returns a loader for subgraphs, a loader
        for superset graphs, and a mapping from subgraph to superset nodes (by index).
        
        Arguments:
        num_subgraphs: maximum number of subgraphs to generate from each graph
        '''
        pass

    def generate_negative_samples(
            self, 
            num_subgraphs: int
        ) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
        '''Generates negative training samples; returns a loader for subgraphs, a loader
        for superset graphs, and a mapping from subgraph to superset nodes (by index).
        
        Arguments:
        num_subgraphs: maximum number of subgraphs to generate from each graph
        '''
        pass