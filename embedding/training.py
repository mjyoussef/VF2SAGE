import argparse
from typing import Dict, Any
from torch_geometric.datasets import WikiCS
from utils import *

def load_data_samples(dataset):
    pass

def test_data_loader():
    graphs = WikiCS('../datasets/wikics')
    generate_samples(
        './data/wikics',
        graphs,
        2,
        0.1,
        0.7,
        dropout=0.1,
    )

def train(args: argparse.ArgumentParser) -> None:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''
    Recommend parameters: https://arxiv.org/pdf/2010.14945.pdf
    WikiCS:
    k: 2
    p_e: 0.3
    p_f: 0.1
    p_t: 0.7
    lr: 0.01
    activation: PReLu
    hd: 256

    AmazonPhoto:
    k: 2
    p_e: 0.4
    p_f: 0.1
    p_t: 0.7
    lr: 0.1
    activation: ReLu
    hd: 256

    CoauthorCS:
    k: 2
    p_e: 0.25
    p_f: 0.35
    p_t: 0.7
    lr: 0.0005
    activation: RReLu
    hd: 256
    
    '''

    test_data_loader()

    # parser.add_argument('--dataset', type=str, required=True, choices=['WikiCS', 'AmazonPhoto', 'CoauthorCS'])
    # parser.add_argument('--k', type=int, required=True)
    # parser.add_argument('--p_f', type=float, required=True)
    # parser.add_argument('--p_e', type=float, required=True)
    # parser.add_argument('--p_t', type=float, required=True)
    # parser.add_argument('--max_subgraphs', type=int, required=True)
    # parser.add_argument('--hidden_dimensions', type=float, required=True)
    # parser.add_argument('--activation', type=str, required=True, choices=['PReLu', 'RReLu', 'ReLu'])
    # parser.add_argument('--lr', type=float, required=True)
    # parser.add_argument('--epochs', type=int, required=True)
    # parser.add_argument('--bs', type=int, required=True)
    # parser.add_argument('--logging', type=bool, required=True)

    # args = parser.parse_args()

    # train(args.dataset)