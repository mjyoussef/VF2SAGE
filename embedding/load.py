import argparse
import torch.optim as optim
from utils import GraphPairsDataset, contrastive_loss, create_loaders
from torch_geometric.datasets import WikiCS, Amazon, Coauthor
from model import GraphSAGE

def train_routine(args, model, train_loader, val_loader):
    for epoch in range(args.epochs):
        # training
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        batch_idx = 0
        for graph_pair, labels in train_loader:
            optimizer.zero_grad()
            out1 = model(graph_pair[0])
            out2 = model(graph_pair[1])
            loss = contrastive_loss(out1, out2, labels, args.margin)
            loss.backward()
            optimizer.step()
            print(f"Training: epoch {epoch}, batch {batch_idx}")
            batch_idx += 1
        
        # validation
        model.eval()
        avg_val_loss = 0
        val_loader_size = len(val_loader)
        for graph_pair, labels in val_loader:
            out1 = model(graph_pair[0])
            out2 = model(graph_pair[1])
            loss = contrastive_loss(out1, out2, labels, args.margin)
            avg_val_loss += loss.item()
        
        avg_val_loss /= val_loader_size
        print(f"Average validation loss: {avg_val_loss}")

def test_routine(args, model, test_loader):
    model.eval()
    avg_test_loss = 0
    test_loader_size = len(test_loader)
    for graph_pair, labels in test_loader:
        out1 = model(graph_pair[0])
        out2 = model(graph_pair[1])
        loss = contrastive_loss(out1, out2, labels, args.margin)
        avg_test_loss += loss.item()
        
    avg_test_loss /= test_loader_size
    print(f"Average test loss: {avg_test_loss}")

def train_and_test(args, train_loader, val_loader, test_loader):
    if (args.dataset == 'WikiCS'):
        args, train_loader, val_loader, test_loader = wikics(args)
    elif (args.dataset == 'AmazonPhoto'):
        args, train_loader, val_loader, test_loader = amazon_photo(args)
    else:
        args, train_loader, val_loader, test_loader = coauthor_cs(args)
    
    # initialize the model
    model = GraphSAGE(
        args.in_channels, 
        args.hidden_channels, 
        args.out_channels,
        args.k
    )

    train_routine(args, model, train_loader, val_loader)

    test_routine(args, model, test_loader)

def wikics(args):

    # store the dataset
    graphs = WikiCS('../datasets/wikics')

    # handle optional parameters w/o defaults
    if (args.p_damp is None):
        args.p_damp = 0.1
    
    if (args.p_trunc is None):
        args.p_trunc = 0.7
    
    if (args.in_dimension is None):
        args.in_dimension = graphs[0].x.shape[1]
    
    if (args.hidden_dimension is None):
        args.hidden_dimension = 256
        args.out_dimension = args.hidden_dimension
    
    if (args.lr is None):
        args.lr = 0.001
    
    if (args.batch_size is None):
        args.batch_size = 64
    
    if (args.epochs is None):
        args.epochs = 50

    dataset = GraphPairsDataset(graphs, args.k, args.p_damp, args.p_trunc, args.dropout)

    return args, create_loaders(dataset, args.train_split, args.val_split, args.batch_size)

def amazon_photo(args):
    pass

def coauthor_cs(args):
    pass

# def amazon_photo(args):
#     buffer_size = args.buffer_size
#     k = args.k
#     p_damp = args.p_damp
#     p_trunc = args.p_trunc
#     dropout = args.dropout

#     # set optional parameters
#     if (p_damp is None):
#         p_damp = 0.1
    
#     if (p_trunc is None):
#         p_trunc = 0.7
    
#     graphs = Amazon('../datasets/amazon_photo', 'photo')

#     generate_samples(
#         f"./data/amazon_photo/{args.version}",
#         buffer_size,
#         graphs,
#         k,
#         p_damp,
#         p_trunc,
#         dropout=dropout
#     )

# def coauthor_cs(args):
#     buffer_size = args.buffer_size
#     k = args.k
#     p_damp = args.p_damp
#     p_trunc = args.p_trunc
#     dropout = args.dropout

#     # set optional parameters
#     if (p_damp is None):
#         p_damp = 0.35
    
#     if (p_trunc is None):
#         p_trunc = 0.7
    
#     graphs = Coauthor('../datasets/coauthor_cs', 'CS')

#     generate_samples(
#         f"./data/coauthor_cs/{args.version}",
#         buffer_size,
#         graphs,
#         k,
#         p_damp,
#         p_trunc,
#         dropout=dropout
#     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['WikiCS', 'AmazonPhoto', 'CoauthorCS'])
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--k', type=int, default=2, required=False)
    parser.add_argument('--p_damp', type=float, default=None, required=False)
    parser.add_argument('--p_trunc', type=float, default=None, required=False)
    parser.add_argument('--dropout', type=float, default=0.2, required=False)
    parser.add_argument('--train_split', type=float, default=0.7, required=False)
    parser.add_argument('--val_split', type=float, default=0.15, required=False)
    parser.add_argument('--margin', type=float, default=1.0, required=False)

    args = parser.parse_args()

    train_and_test(args)