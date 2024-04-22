import argparse
import torch.optim as optim
from utils import GraphPairsDataset, contrastive_loss, create_loaders
from torch_geometric.datasets import WikiCS, Amazon, Coauthor
from model import GraphSAGE
import json

def train_routine(params, model, train_loader, val_loader):
    for epoch in range(params['epochs']):
        # training
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        batch_idx = 0
        for graph_pair, labels in train_loader:
            optimizer.zero_grad()
            out1 = model(graph_pair[0])
            out2 = model(graph_pair[1])
            loss = contrastive_loss(out1, out2, labels, params['margin'])
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
            loss = contrastive_loss(out1, out2, labels, params['margin'])
            avg_val_loss += loss.item()
        
        avg_val_loss /= val_loader_size
        print(f"Average validation loss: {avg_val_loss}")

def test_routine(params, model, test_loader):
    model.eval()
    avg_test_loss = 0
    test_loader_size = len(test_loader)
    for graph_pair, labels in test_loader:
        out1 = model(graph_pair[0])
        out2 = model(graph_pair[1])
        loss = contrastive_loss(out1, out2, labels, params['margin'])
        avg_test_loss += loss.item()
        
    avg_test_loss /= test_loader_size
    print(f"Average test loss: {avg_test_loss}")

def train_and_test(dataset_name, params):
    graphs = None
    if (dataset_name == 'WikiCS'):
        graphs = WikiCS('../datasets/WikiCS')
    elif (dataset_name == 'AmazonPhoto'):
        graphs = Amazon('../datasets/AmazonPhoto', 'Photo')
    else:
        graphs = Coauthor('../datasets/CoauthorCS', 'CS')
    
    # update the input channels based on the graph's feature matrix
    params['in_channels'] = graphs[0].x.shape[1]

    # create the dataset
    dataset = GraphPairsDataset(graphs, params['k'], params['p_damp'], params['p_trunc'], params['dropout'])

    # create the loaders
    train_loader, val_loader, test_loader = create_loaders(
        dataset, 
        params['train_split'], 
        params['val_split'], 
        params['batch_size']
    )

    # initialize the model
    model = GraphSAGE(
        params['in_channels'],
        params['hidden_dimension'],
        params['out_dimension'],
        params['k'],
    )

    train_routine(params, model, train_loader, val_loader)

    test_routine(params, model, test_loader)

if __name__ == '__main__':

    # get the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['WikiCS', 'AmazonPhoto', 'CoauthorCS'])
    args = parser.parse_args()

    # load the hyperparameters
    params = json.load(f'../configs/{args.dataset}.json')

    train_and_test(args.dataset, params)