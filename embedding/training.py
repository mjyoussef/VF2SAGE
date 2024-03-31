import argparse
from typing import Dict, Any

def train(args: argparse.ArgumentParser) -> None:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True, choices=['WordNet', 'PPI', 'CoraML'])
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--p_f', type=float, required=True)
    parser.add_argument('--p_e', type=float, required=True)
    parser.add_argument('--p_t', type=float, required=True)
    parser.add_argument('--max_subgraphs', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--logging', type=bool, required=True)

    args = parser.parse_args()

    train(args.dataset)