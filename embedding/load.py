import argparse
from utils import *
from torch_geometric.datasets import WikiCS, Amazon, Coauthor

def wikics(args):
    k = args.k
    p_damp = args.p_damp
    p_trunc = args.p_trunc
    dropout = args.dropout

    # set optional parameters
    if (p_damp is not None):
        p_damp = 0.1
    
    if (p_trunc is not None):
        p_trunc = 0.7
    
    graphs = WikiCS('../datasets/wikics')

    generate_samples(
        "./data/wikics",
        graphs,
        k,
        p_damp,
        p_trunc,
        dropout
    )

def amazon_photo(args):
    k = args.k
    p_damp = args.p_damp
    p_trunc = args.p_trunc
    dropout = args.dropout

    # set optional parameters
    if (p_damp is not None):
        p_damp = 0.1
    
    if (p_trunc is not None):
        p_trunc = 0.7
    
    graphs = Amazon('../datasets/amazon_photo', 'photo')

    generate_samples(
        "./data/amazon_photo",
        graphs,
        k,
        p_damp,
        p_trunc,
        dropout
    )

def coauthor_cs(args):
    k = args.k
    p_damp = args.p_damp
    p_trunc = args.p_trunc
    dropout = args.dropout

    # set optional parameters
    if (p_damp is not None):
        p_damp = 0.35
    
    if (p_trunc is not None):
        p_trunc = 0.7
    
    graphs = Coauthor('../datasets/coauthor_cs', 'CS')

    generate_samples(
        "./data/coauthor_cs",
        graphs,
        k,
        p_damp,
        p_trunc,
        dropout
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['WikiCS', 'AmazonPhoto', 'CoauthorCS'])
    parser.add_argument('--k', type=int, default=2, required=False)
    parser.add_argument('--p_damp', type=float, default=None, required=False)
    parser.add_argument('--p_trunc', type=float, default=None, required=False)
    parser.add_argument('--dropout', type=float, default=0.1, required=False)

    args = parser.parse_args()

    if (args.dataset == 'WikiCS'):
        wikics(args)
    elif (args.dataset == 'AmazonPhoto'):
        amazon_photo(args)
    else:
        coauthor_cs(args)