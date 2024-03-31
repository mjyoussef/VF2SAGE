import argparse

def train(dataset: str, k: int) -> None:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['WordNet', 'PPI', 'CoraML'])
    parser.add_argument('--k', type=int, default=2)
    args = parser.parse_args()
    
    train(args.dataset, args.k)