import torch
import argparse

def wikics():
    pass

def amazon_photo():
    pass

def coauthor_cs():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['WikiCS', 'AmazonPhoto', 'CoauthorCS'])
    parser.add_argument('--k', type=int, default=2, required=False)
    parser.add_argument('--p_damp', type=float, required=False)
    parser.add_argument('--p_trunc', type=float, required=False)
    parser.add_argument('--dropout', type=float, required=False)
    parser.add_argument('--buffer_size', type=float, required=False)


    # parser.add_argument('--p_e', type=float, required=True)
    # parser.add_argument('--p_t', type=float, required=True)
    # parser.add_argument('--max_subgraphs', type=int, required=True)
    # parser.add_argument('--hidden_dimensions', type=float, required=True)
    # parser.add_argument('--activation', type=str, required=True, choices=['PReLu', 'RReLu', 'ReLu'])
    # parser.add_argument('--lr', type=float, required=True)
    # parser.add_argument('--epochs', type=int, required=True)
    # parser.add_argument('--bs', type=int, required=True)
    # parser.add_argument('--logging', type=bool, required=True)