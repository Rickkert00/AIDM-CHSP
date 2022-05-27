import argparse
import json
import os
import time

import dgl
import numpy as np
import torch
from numpy import mean
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from imitation_learning.neural_network import RemovalTimePredictor

INF = 9999


def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--learning_rate', default=8e-4, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--work_dir', default='work_dir', type=str)
    parser.add_argument('--epochs', default=1000, type=int)
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

def _run(data_loader, model: RemovalTimePredictor, criterion, device, optimizer=None):
    avg_loss = 0
    correct = 0
    total = 0
    # Iterate through batches
    for _, data in enumerate(data_loader):
        # Get the inputs; data is a list of [inputs, labels]
        graph_inputs, labels = data
        # Move data to target device
        graph_inputs, labels = graph_inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        period, _ = model(graph_inputs, graph_inputs.ndata['x'], graph_inputs.edata['w'])

        # TODO find out why all values of period tensor are equivalent
        period_index = 0
        output = torch.zeros(graph_inputs.batch_size)
        for i in range(graph_inputs.batch_size):
            g_nodes = graph_inputs._batch_num_nodes['_N'][i]
            period_index += g_nodes
            output[i] = period[period_index - 1]
        loss = criterion(output, labels)  # calculate on the last tensor as that is the final tank

        if optimizer:
            loss.backward()
            optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
        total += labels.size(0)
        correct += torch.isclose(output, labels, atol=5).sum().item()

    return avg_loss.item() / len(data_loader), 100 * correct / total


def load_graph_data(training_size=0.8):
    solutions_and_input = np.load('../chsp-generators-main/instances/linear_solutions.npy', allow_pickle=True)
    input_data = solutions_and_input[:, 0]
    solutions = solutions_and_input[:, 1]
    num_nodes = [torch.from_numpy(np.array(input['Ninner'])).float() for input in input_data]

    # fully connected adj matrix with weights, including self connections
    adjs = [torch.ones((int(num_tanks.item() + 2), int(num_tanks.item() + 2))) for num_tanks in
            num_nodes]  # obtain adjacency matrix, add 2 extra as we need start and end node
    # structure format: (tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    # so structure[0][0] and structure[1][0] are connected etc.
    structure = list(zip([torch.nonzero(adj, as_tuple=True) for adj in
                          adjs]))  # edges list, 2 lists that indicate what nodes are connected
    graphs = [dgl.graph((u_v[0][0], u_v[0][1])) for u_v in structure]
    # g1.ndata['x'] = th.zeros(g1.num_nodes(), 3)
    # g1.edata['w'] = th.ones(g1.num_edges(), 2)
    # offset input variable 'f' time as by 1 as f[0] is from initial stage to first tank, need to add that later
    node_feats = [[torch.from_numpy(np.array([0, INF, input['f'][idx]])) if idx == 0
                   else torch.from_numpy(np.array([input['tmin'][idx - 1], input['tmax'][idx - 1], input['f'][idx]]))
                   for idx in range(int(num_nodes[i].item()) + 1)]
                  for i, input in
                  enumerate(
                      input_data)]  # need to have 3 features per node, and num_nodes nodes per problem. We add a start node already in this loop
    # now need to add final node
    for node_feat in node_feats:
        node_feat.append(torch.from_numpy(np.array([0, INF, 0])))  # Add ending node

    # convert node_feats to correct input shape and format
    corr_node_feats = []
    for tensors in node_feats:
        corr_node_feats.append(torch.stack(tensors, dim=0).type(torch.float32))
    edge_feats = []
    # 'e' input param format: [[6  0  6 12], [12  6  0  6], [18 12  6  0], [0  6 12 18]] for 1 example instance
    # corr format would be: [0,6,12,18,0,6,0,6,12,0,12,6,0,6,0,0,0,0,0,0]
    for idx, input in enumerate(input_data):
        edge_feat = []
        edge_array = input['e']
        # add start stage
        edge_feat.extend(np.append(edge_array[-1], 0))
        for edge_num in range(int(num_nodes[idx].item())):  # also include starting stage
            # add in between stages
            edge_feat.extend(np.append(edge_array[edge_num], 0))
        # add final stage
        edge_feat.extend(np.array([0 for _ in range(int(num_nodes[idx].item()) + 2)]))

        # add edges to final stage
        float_edges = torch.Tensor(edge_feat).reshape(-1, 1)
        edge_feats.append(float_edges)
    for i in range(len(input_data)):
        graphs[i].ndata['x'] = corr_node_feats[i]
        graphs[i].edata['w'] = edge_feats[i]
    input_data = [graphs[i]for i in range(len(input_data))]  # convert input into 1 tuple
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
    graphs, labels = map(list, zip(*batches))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.Tensor(labels)


def main(_args=None):
    debug = False
    print("args", _args)
    args = parse_args(_args)
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    exp_id = 'test'
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
    device = torch.device('cpu')  # Use cpu to debug faster
    print("Using device:", device)
    train_set, test_set = load_graph_data()
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    train_loader = dgl.dataloading.GraphDataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    print('len training', len(test_set))

    test_set = test_set[len(test_set)//args.batch_size*args.batch_size]
    print('test_set', len(test_set))
    # test_loader = dgl.dataloading.GraphDataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=True)

    if not debug:
        writer = SummaryWriter()

    model = RemovalTimePredictor()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)  # , weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.992)
    train_time_avg = []
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()  # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization
        train_loss, train_accuracy = _run(train_loader, model, criterion, device, optimizer)
        train_time = time.time() - start_time
        train_time_avg.append(train_time)

        # print('train step seconds:', mean(train_time_avg).round(3))
        test_loss, test_accuracy = None, None
        # if epoch % 4 == 1:
        #     model.eval()
        #     with torch.no_grad():
        #         test_loss, test_accuracy = _run(test_loader, model, criterion, device)

        print("epoch", epoch, "train_loss", round(train_loss,1), "train_accuracy", round(train_accuracy,2))

        if not debug:
            loss_d = {
                'Train_GNN': train_loss,
            }
            acc_d = {
                'Train_GNN': train_accuracy,
            }
            if test_loss:
                loss_d['Test_GNN'] = test_loss
                acc_d['Test_GNN'] = test_accuracy
            writer.add_scalars('Loss', loss_d, epoch)
            writer.add_scalars('Accurary', acc_d, epoch)
        scheduler.step()
        print("lr:", round(scheduler.get_last_lr()[0],8))
    if not debug:
        writer.flush()
        writer.close()
    # if args.model_dir is not None:
    #   pass
    # agent.load(args.model_dir, args.model_step)
    # L = Logger(args.work_dir, use_tb=args.save_tb)


if __name__ == '__main__':
    main()