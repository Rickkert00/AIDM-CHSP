import argparse
import json
import os
import time

import dgl
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from imitation_learning.neural_network import RemovalTimePredictor

INF = 9999

def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--work_dir', default='work_dir', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args(_args)
    return args


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(train_loader, model, optimizer, criterion, device):
    """
    Trains network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
    model.train()  # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization

    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for i, data in enumerate(train_loader):
        # Get the inputs; data is a list of [inputs, labels]
        (graph_inputs, node_inputs, edge_inputs), labels = data
        # Move data to target device
        node_inputs, edge_inputs, labels = node_inputs.to(device), edge_inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        period, _ = model(graph_inputs, node_inputs, edge_inputs)

        # TODO find out why all values of period tensor are equivalent
        loss = criterion(period[-1].squeeze(0), labels)  # calculate on the last tensor as that is the final tank

        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
        total += labels.size(0)
        # correct += torch.isclose(points_in_image.data, labels_in_image, atol=2).sum().item() // 8

    return avg_loss.item() / len(train_loader)  # , 100 * correct / total


def load_graph_data(device, training_size=0.8):
    solutions_and_input = np.load('../chsp-generators-main/instances/linear_solutions.npy', allow_pickle=True)
    input_data = solutions_and_input[:, 0]
    solutions = solutions_and_input[:, 1]
    num_nodes = [torch.from_numpy(np.array(input['Ninner'])).float() for input in input_data]

    # fully connected adj matrix with weights, including self connections
    adjs = [torch.ones((int(num_tanks.item()+2), int(num_tanks.item()+2))) for num_tanks in
            num_nodes]  # obtain adjacency matrix, add 2 extra as we need start and end node
    # structure format: (tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    # so structure[0][0] and structure[1][0] are connected etc.
    structure = list(zip([torch.nonzero(adj, as_tuple=True) for adj in
                          adjs]))  # edges list, 2 lists that indicate what nodes are connected
    graphs = [dgl.graph((u_v[0][0], u_v[0][1]), device=device) for u_v in structure]

    # offset input variable 'f' time as by 1 as f[0] is from initial stage to first tank, need to add that later
    node_feats = [[torch.from_numpy(np.array([0, INF, input['f'][idx]])) if idx == 0
                   else torch.from_numpy(np.array([input['tmin'][idx-1], input['tmax'][idx-1], input['f'][idx]]))
                   for idx in range(int(num_nodes[i].item())+1)]
                   for i, input in enumerate(input_data)]  # need to have 3 features per node, and num_nodes nodes per problem. We add a start node already in this loop
    # now need to add final node
    for node_feat in node_feats:
        node_feat.append(torch.from_numpy(np.array([0, INF, 0]))) # Add ending node

    # convert node_feats to correct input shape and format
    corr_node_feats = []
    for tensors in node_feats:
        corr_node_feats.append(torch.stack(tensors, dim=0))
        corr_node_feats[-1] = corr_node_feats[-1].type(torch.float32)
    edge_feats = []
    # 'e' input param format: [[6  0  6 12], [12  6  0  6], [18 12  6  0], [0  6 12 18]] for 1 example instance
    # corr format would be: [0,6,12,18,0,6,0,6,12,0,12,6,0,6,0,0,0,0,0,0]
    for idx, input in enumerate(input_data):
        edge_feat = []
        edge_array = input['e']
        # add start stage
        edge_feat.extend(np.append(edge_array[-1], 0))
        for edge_num in range(int(num_nodes[idx].item())): # also include starting stage
            # add in between stages
            edge_feat.extend(np.append(edge_array[edge_num], 0))
        # add final stage
        edge_feat.extend(np.array([0 for _ in range(int(num_nodes[idx].item()) + 2)]))

        # add edges to final stage
        float_edges = torch.Tensor(edge_feat).reshape(-1, 1)
        edge_feats.append(float_edges)

    input_data = [(graphs[i], corr_node_feats[i], edge_feats[i]) for i in range(len(input_data))] # convert input into 1 tuple
    solved = [torch.from_numpy(np.array(d['objective'])).float().unsqueeze(dim=0) for d in
              solutions]  # TODO change this to 'r' if we want to train using removal times
    length = len(input_data)
    # TODO is this a good split? maybe use random split instead of this
    training_length = int(length * training_size)
    training = [(input_data[i], solved[i]) for i in range(training_length)]
    test = [(input_data[i], solved[i]) for i in range(training_length, length)]
    return training, test


def collate_fn(batches):
    """
    Custom function for creating batches of training samples, right now this will construct batches of
    size 1, as we cannot have batches > size 1 because of varying input size.
    :param batches: Tuple of input and output pairs that should be merged into a batch
    :return: returns the batched input samples
    """
    graphs = []
    node_f = None
    edge_f = None
    labels = None
    for batch in batches:  # this will be batch size 1 for graph network
        node_features = batch[0][1]
        edge_features = batch[0][2]
        label = batch[1]
        graphs = batch[0][0]
        if node_f is None:
            node_f = node_features
        if labels is None:
            labels = label
        if edge_f is None:
            edge_f = edge_features
            continue
        else:
            torch.stack((node_f, node_features), dim=0)
            torch.stack((edge_f, edge_features), dim=0)
            torch.stack((labels, label), dim=0)

    return (graphs, node_f, edge_f), labels


def main(_args=None):
    print("args", _args)
    args = parse_args(_args)
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    exp_id = str(int(np.random.random() * 100000))
    set_seed_everywhere(args.seed)
    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H", ts)
    exp_name = 'exp'
    exp_name += '-s' + str(args.seed)

    exp_name += '-id' + exp_id
    args.work_dir = args.work_dir + '/' + exp_name
    make_dir(args.work_dir)
    print("Working in directory:", args.work_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    folder = '../chsp-generators-main/instances/linear_solutions/'
    train_set, test_set = load_graph_data(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    model = RemovalTimePredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)  # , weight_decay=args.weight_decay)

    for i in range(args.epochs):
        loss = train(train_loader, model, optimizer, criterion, device)
        print(loss)
    # if args.model_dir is not None:
    #   pass
    # agent.load(args.model_dir, args.model_step)
    # L = Logger(args.work_dir, use_tb=args.save_tb)


if __name__ == '__main__':
    main()
