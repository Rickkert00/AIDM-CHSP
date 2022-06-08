import argparse
import json
import os
import time
from numpy import mean

import dgl
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch.utils.data import DataLoader, random_split, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from imitation_learning.neural_network import RemovalTimePredictor

INF = 9999


class GraphDataset(IterableDataset):
    def __init__(self, dataset):
        super(GraphDataset).__init__()
        self.list = dataset

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]


def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--decay', default=1, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--work_dir', default='work_dir', type=str)
    parser.add_argument('--epochs', default=2500, type=int)
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


def train(train_loader, model, optimizer, criterion, device, test_loader):
    """
    Trains network for one epoch in batches.
    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
    model.train()  # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization

    total_train_loss = 0
    absolute_values = [1, 2, 4, 8, 16, 32, 64]
    scaling_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    total_train_accuracy_absolute = {k: 0 for k in absolute_values}
    total_train_accuracy_scaling = {k: 0 for k in scaling_values}
    avg_train_accuracy_absolute = {k: 0 for k in absolute_values}
    avg_train_accuracy_scaling = {k: 0 for k in scaling_values}
    total = 0

    # Iterate through batches
    for i, data in enumerate(train_loader):
        # Get the inputs; data is a list of [inputs, labels]
        (graph_inputs, node_inputs, edge_inputs), (removal_times_gt, period_gt) = data
        # Move data to target device
        node_inputs, edge_inputs, removal_times = node_inputs.to(device), edge_inputs.to(device), removal_times_gt.to(
            device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        removal_times, _ = model(graph_inputs, node_inputs, edge_inputs)
        removal_times = removal_times.squeeze(-1)
        if removal_times[1:-1].shape != removal_times_gt.shape:
            print("WRONG SHAPE")
        # print(removal_times)
        # TODO find out why all values of period tensor are equivalent
        # loss = criterion(period[-1].squeeze(0), labels)  # calculate on the last tensor as that is the final tank
        loss = criterion(removal_times[1:-1], removal_times_gt)
        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        total_train_loss += loss
        for k in total_train_accuracy_absolute:
            total_train_accuracy_absolute[k] += (torch.isclose(removal_times[1:-1], removal_times_gt, atol=k,
                                                               rtol=0).sum().item() / removal_times_gt.size(0))
        for k in total_train_accuracy_scaling:
            total_train_accuracy_absolute[k] += (
                        torch.isclose(removal_times[1:-1], removal_times_gt, atol=k * period_gt,
                                      rtol=0).sum().item() / removal_times_gt.size(0))

    for k in total_train_accuracy_absolute:
        avg_train_accuracy_absolute[k] = total_train_accuracy_absolute[k] / len(train_loader)
    for k in total_train_accuracy_scaling:
        avg_train_accuracy_scaling[k] = total_train_accuracy_scaling[k] / len(train_loader)
    avg_train_loss = total_train_loss.item() / len(train_loader)
    # calculate test loss
    model.eval()
    with torch.no_grad():
        accuracy = 0
        total_test_loss = 0
        total_test_accuracy = 0
        avg_loss = 0
        for i, data in enumerate(test_loader):
            # get inputs
            (graph_inputs, node_inputs, edge_inputs), labels = data
            # Move data to target device
            node_inputs, edge_inputs, labels = node_inputs.to(device), edge_inputs.to(device), labels.to(device)
            # Forward + backward
            removal_times, _ = model(graph_inputs, node_inputs, edge_inputs)
            removal_times = removal_times.squeeze(-1)
            loss = criterion(removal_times[1:-1], labels)
            total_test_loss += loss
            total += labels.size(0)
    avg_test_loss = total_test_loss.item() / len(test_loader)

    return avg_loss.item() / len(train_loader)  # , 100 * correct / total


def load_graph_data_without_batching(device, training_size=0.8, scaling=0.01):
    solutions_and_input = np.load('../chsp-generators-main/instances/linear_solutions.npy', allow_pickle=True)
    solutions_and_input = np.vstack([s for s in solutions_and_input if s is not None and s[1] is not None])
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
    graphs = [dgl.graph((u_v[0][0], u_v[0][1]), device=device) for u_v in structure]

    # offset input variable 'f' time as by 1 as f[0] is from initial stage to first tank, need to add that later
    node_feats = [[torch.from_numpy(np.array([0, INF, input['f'][idx]*scaling])) if idx == 0
                   else torch.from_numpy(np.array([input['tmin'][idx - 1]*scaling, input['tmax'][idx - 1]*scaling, input['f'][idx]*scaling]))
                   for idx in range(int(num_nodes[i].item()) + 1)]
                  for i, input in enumerate(
            input_data)]  # need to have 3 features per node, and num_nodes nodes per problem. We add a start node already in this loop
    # now need to add final node
    for node_feat in node_feats:
        node_feat.append(torch.from_numpy(np.array([0, INF, 0])))  # Add ending node

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
        edge_feat.extend(np.append(edge_array[-1]*scaling, 0))
        for edge_num in range(int(num_nodes[idx].item())):  # also include starting stage
            # add in between stages
            edge_feat.extend(np.append(edge_array[edge_num]*scaling, 0))
        # add final stage
        edge_feat.extend(np.array([0 for _ in range(int(num_nodes[idx].item()) + 2)]))

        # add edges to final stage
        float_edges = torch.Tensor(edge_feat).reshape(-1, 1)
        edge_feats.append(float_edges)

    input_data = [(graphs[i], corr_node_feats[i], edge_feats[i]) for i in
                  range(len(input_data))]  # convert input into 1 tuple
    # TODO change this to 'r' if we want to train using removal times
    solved = [(torch.from_numpy(np.array(d['r'][1:])*scaling).float().unsqueeze(dim=0).reshape(-1, 1),
               torch.from_numpy(np.array(d['objective']*scaling)).float().unsqueeze(dim=0)) for d in
              solutions]  # we do not include the first element of the tensor as that is always 0 in all solutions ( we move from the initial stage as we start)

    length = len(input_data)
    # TODO is this a good split? maybe use random split instead of this
    training_length = int(length * training_size)
    test_length = round((1 - training_size) * length)
    dataset = GraphDataset([(input_data[i], solved[i]) for i in range(length)])
    training, test = random_split(dataset, [training_length, test_length])
    # training = [(input_data[i], solved[i]) for i in range(training_length)]
    # test = [(input_data[i], solved[i]) for i in range(training_length, length)]
    return training, test


def load_graph_data_with_batching(device, path='../chsp-generators-main/instances/linear_solutions.npy',
                                  training_size=0.8, predict_period=False, scaling=0.01):
    solutions_and_input = np.load(path, allow_pickle=True)
    input_data = solutions_and_input[:, 0]
    solutions = solutions_and_input[:, 1]
    num_nodes = [torch.tensor(input['Ninner']).float() for input in input_data]

    # fully connected adj matrix with weights, including self connections
    adjs = [torch.ones((int(num_tanks.item() + 2), int(num_tanks.item() + 2))) for num_tanks in
            num_nodes]  # obtain adjacency matrix, add 2 extra as we need start and end node
    # structure format: (tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    # so structure[0][0] and structure[1][0] are connected etc.
    max = 99999
    # top_layers = [torch.nonzero(torch.ones(num_tanks), as_tuple=True)+num_tanks for num_tanks in
    #         num_nodes]
    structure = list(zip([torch.nonzero(adj, as_tuple=True) for adj in
                          adjs]))  # edges list, 2 lists that indicate what nodes are connected
    graphs = [dgl.graph((u_v[0][0], u_v[0][1]), device=device) for u_v in structure]
    # offset input variable 'f' time as by 1 as f[0] is from initial stage to first tank, need to add that later
    node_feats = [[torch.tensor([0, max, input['f'][idx]*scaling]) if idx == 0
                   else torch.tensor([input['tmin'][idx - 1]*scaling, input['tmax'][idx - 1]*scaling, input['f'][idx]*scaling])
                   for idx in range(int(num_nodes[i].item()) + 1)]
                  for i, input in enumerate(
            input_data)]  # need to have 3 features per node, and num_nodes nodes per problem. We add a start node already in this loop
    # now need to add final node
    for node_feat in node_feats:
        node_feat.append(torch.tensor([0, max, 0]))  # Add ending node

    # convert node_feats to correct input shape and format
    corr_node_feats = []
    for tensors in node_feats:
        corr_node_feats.append(torch.stack(tensors, dim=0).float())
    edge_feats = []
    # 'e' input param format: [[6  0  6 12], [12  6  0  6], [18 12  6  0], [0  6 12 18]] for 1 example instance
    # corr format would be: [0,6,12,18,0,6,0,6,12,0,12,6,0,6,0,0,0,0,0,0]
    for idx, input in enumerate(input_data):
        edge_feat = []
        edge_array = input['e']
        # add start stage
        edge_feat.extend(np.append(edge_array[-1]*scaling, 0))
        for edge_num in range(int(num_nodes[idx].item())):  # also include starting stage
            # add in between stages
            edge_feat.extend(np.append(edge_array[edge_num]*scaling, 0))
        # add final stage
        edge_feat.extend(np.array([0 for _ in range(int(num_nodes[idx].item()) + 2)]))

        # add edges to final stage
        float_edges = torch.Tensor(edge_feat).reshape(-1, 1)
        edge_feats.append(float_edges)
    for i in range(len(input_data)):
        graphs[i].ndata['x'] = corr_node_feats[i]
        graphs[i].edata['w'] = edge_feats[i]
    input_data = graphs
    label = 'r'
    if predict_period:
        label = 'objective'
    # solved = [torch.tensor(d[label][1:]+[d['objective']]).float().unsqueeze(dim=0) for d in solutions]
    solved = [(torch.tensor(d[label][1:])*scaling).float().unsqueeze(dim=0) for d in solutions]
    length = len(input_data)
    # training, test = train_test_split(input_data, solved, train_size=training_size)
    training_length = int(length * training_size)
    test_length = round((1 - training_size) * length)
    dataset = GraphDataset([(input_data[i], solved[i]) for i in range(length)])
    training, test = random_split(dataset, [training_length, test_length])
    # test = [(input_data[i], solved[i]) for i in range(training_length, length)]
    return training, test


def collate_fn_old(batches):
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


def collate_fn(batches):
    """
    Custom function for creating batches of training samples, right now this will construct batches of
    size 1, as we cannot have batches > size 1 because of varying input size.
    :param batches: Tuple of input and output pairs that should be merged into a batch
    :return: returns the batched input samples
    """
    graphs, labels = map(list, zip(*batches))
    batched_graph = dgl.batch(graphs)
    max_len = len(max(labels, key=lambda l: len(l[0]))[0])
    labels_padded = torch.zeros((len(labels), max_len))
    for i, l in enumerate(labels):
        labels_padded[i, :len(l[0])] = l
    return batched_graph, labels_padded


def _run_old(data_loader,test_loader, model: RemovalTimePredictor, criterion, device, optimizer=None, epoch=0, tolerances=None, rel_tolerance=None):
    model.train()  # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization

    total_train_loss = 0
    absolute_values = [1, 2, 4, 8, 16, 32, 64]
    absolute_values = [k*0.01 for k in absolute_values]
    scaling_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    total_train_accuracy_absolute = {k: 0 for k in absolute_values}
    total_train_accuracy_scaling = {k: 0 for k in scaling_values}
    avg_train_accuracy_absolute = {k: 0 for k in absolute_values}
    avg_train_accuracy_scaling = {k: 0 for k in scaling_values}
    total = 0
    for i, data in enumerate(data_loader):
        # Get the inputs; data is a list of [inputs, labels]
        (graph_inputs, node_inputs, edge_inputs), (removal_times_gt, period_gt) = data
        # Move data to target device
        node_inputs, edge_inputs, removal_times_gt = node_inputs.to(device), edge_inputs.to(device), removal_times_gt.to(
            device)

        if optimizer: optimizer.zero_grad()

        removal_times, _ = model(graph_inputs, node_inputs, edge_inputs)
        removal_times = removal_times.squeeze(-1)
        if i % 1000 == 0:
            print(removal_times)
        if removal_times[1:-1].shape != removal_times_gt.shape:
            print("WRONG SHAPE")
        loss = criterion(removal_times[1:-1], removal_times_gt)
        if optimizer:
            loss.backward()
            optimizer.step()

        # Keep track of loss and accuracy
        # Keep track of loss and accuracy
        total_train_loss += loss
        for k in total_train_accuracy_absolute:
            total_train_accuracy_absolute[k] += (torch.isclose(removal_times[1:-1], removal_times_gt, atol=k,
                                                               rtol=0).sum().item() / removal_times_gt.size(0))
        for k in total_train_accuracy_scaling:
            total_train_accuracy_scaling[k] += (
                    torch.isclose(removal_times[1:-1], removal_times_gt, atol=(k * period_gt).item(),
                                  rtol=0).sum().item() / removal_times_gt.size(0))

    for k in total_train_accuracy_absolute:
        avg_train_accuracy_absolute[k] = total_train_accuracy_absolute[k] / len(data_loader)
    for k in total_train_accuracy_scaling:
        avg_train_accuracy_scaling[k] = total_train_accuracy_scaling[k] / len(data_loader)
    avg_train_loss = total_train_loss.item() / len(data_loader)
    # calculate test loss
    model.eval()
    with torch.no_grad():
        accuracy = 0
        total_test_loss = 0
        total_test_accuracy = 0
        avg_loss = 0
        for i, data in enumerate(test_loader):
            # get inputs
            (graph_inputs, node_inputs, edge_inputs), (removal_times_gt, period_gt) = data
            # Move data to target device
            node_inputs, edge_inputs, removal_times_gt = node_inputs.to(device), edge_inputs.to(device), removal_times_gt.to(device)
            # Forward + backward
            removal_times, _ = model(graph_inputs, node_inputs, edge_inputs)
            removal_times = removal_times.squeeze(-1)
            loss = criterion(removal_times[1:-1], removal_times_gt)
            total_test_loss += loss
    avg_test_loss = total_test_loss.item() / len(test_loader)
    return avg_train_loss, avg_train_accuracy_absolute, avg_train_accuracy_scaling, avg_test_loss

def _run(data_loader, model: RemovalTimePredictor, criterion, device, optimizer=None, epoch=0, tolerances=None, rel_tolerance=None):
    avg_loss = 0
    correct = 0
    total = 0
    total_accuracy_absolute = {k: 0 for k in tolerances}
    total_accuracy_scaling = {k: 0 for k in rel_tolerance}
    avg_accuracy_absolute = {}
    avg_accuracy_scaling = {}
    # Iterate through batches
    output = None
    for _, data in enumerate(data_loader):
        # Get the inputs; data is a list of [inputs, labels]
        graph_inputs, labels = data
        # Move data to target device
        graph_inputs, labels= graph_inputs.to(device), labels

        if optimizer: optimizer.zero_grad()

        output, edges = model(graph_inputs, graph_inputs.ndata['x'], graph_inputs.edata['w'])

        # TODO find out why all values of period tensor are equivalent
        output_index = 0
        only_period = False
        if not only_period:
            padded_output = torch.zeros((graph_inputs.batch_size, len(labels[0])))

            for i in range(graph_inputs.batch_size):
                nodes = graph_inputs._batch_num_nodes['_N'][i].item()
                padded_output[i, :nodes - 2] = output[output_index+1:output_index + nodes - 1, 0, 0]
                output_index += nodes
            total_labels = output_index
            padded_label_count = padded_output.shape[0]*padded_output.shape[1] - total_labels
            loss = criterion(padded_output, labels)  # calculate on the last tensor as that is the final tank
            correct += (torch.isclose(padded_output, labels, atol=5).sum().item() - padded_label_count)/total_labels
            for k in tolerances:
                total_accuracy_absolute[k] += (torch.isclose(padded_output, labels, atol=k, rtol=0).sum().item() - padded_label_count)/total_labels
            for k in rel_tolerance:
                total_accuracy_scaling[k] += (torch.isclose(padded_output, labels, atol=0, rtol=k).sum().item() - padded_label_count)/total_labels
            total += graph_inputs.batch_size

        else:
            indexes = torch.cumsum(graph_inputs._batch_num_nodes['_N'],dim=0)-1
            period_label = torch.zeros(graph_inputs.batch_size)
            for i in range(graph_inputs.batch_size):
                period_label[i] = labels[i, graph_inputs._batch_num_nodes['_N'][i]-2]
            pred = output[indexes,0,0].cpu()
            loss = criterion(pred.cpu(), period_label)  # calculate on the last tensor as that is the final tank
            correct += torch.isclose(pred, period_label, atol=5).sum().item()
            total += period_label.size(0)

        if optimizer:
            loss.backward()
            optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
    if epoch % 2 == 1:
        print('Pred', output[:graph_inputs._batch_num_nodes['_N'][0]][:,0,0])
        print('Pred edges rounded', edges[:10,0,0].round())

    for k in total_accuracy_absolute:
        avg_accuracy_absolute[k] = total_accuracy_absolute[k] / total
    for k in total_accuracy_scaling:
        avg_accuracy_scaling[k] = total_accuracy_scaling[k] / total
    return avg_loss.item() / len(data_loader), 100 * correct / total, total_accuracy_absolute, total_accuracy_scaling

def main(_args=None):
    debug = False
    scaling = 0.01
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    folder = '../chsp-generators-main/instances/linear_solutions/'
    train_set, test_set = load_graph_data_without_batching(device)
    # train_loader = dgl.dataloading.GraphDataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn,
    #                                                shuffle=True, drop_last=True)
    # print('len training', len(train_set))
    #
    # # test_set = test_set[len(test_set)//args.batch_size*args.batch_size]
    # print('test_set', len(test_set))
    # test_loader = dgl.dataloading.GraphDataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn,
    #                                               shuffle=True, drop_last=True)
    train_loader = DataLoader(train_set, collate_fn=collate_fn_old, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, collate_fn=collate_fn_old, shuffle=True, drop_last=True)
    if not debug:
        writer = SummaryWriter()

    absolute_values = [1, 2, 4, 8, 16, 32, 64]
    if scaling:
        absolute_values = list(map(lambda x: x*scaling, absolute_values))
    scaling_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    model = RemovalTimePredictor()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)  # , weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.decay) # todo do not change learning rate to prevent stagnation
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=args.epochs)
    train_time_avg = []
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()  # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization
        train_loss, train_acc_absolute, train_acc_scaling, test_loss = _run_old(train_loader, test_loader, model, criterion, device, optimizer)
        train_time = time.time() - start_time
        train_time_avg.append(train_time)

        print('train step seconds:', mean(train_time_avg).round(3))
        # test_loss, test_accuracy = None, None
        # if epoch % 4 == 1:
        #     model.eval()
        #     with torch.no_grad():
        #         test_loss, test_accuracy, test_acc_absolute, test_acc_scaling = _run_old(test_loader, model, criterion, device, epoch=epoch,tolerances=absolute_values, rel_tolerance=scaling_values)

        print("epoch", epoch, "train_loss", round(train_loss, 1), "train_accuracy", train_acc_absolute, "test loss", test_loss)

        if not debug:
            loss_d = {
                'Train_GNN': train_loss,
                'Test_GNN': test_loss
            }

            acc_d = {
                'Train_GNN': train_acc_absolute[0.08], # this is 8 units off
            }

            acc_d = {**acc_d, **{f'Train_GNN_absolute_{k}': train_acc_absolute[k] for k in train_acc_absolute}}
            acc_d = {**acc_d, **{f'Train_GNN_scaling_{k}': train_acc_scaling[k] for k in train_acc_scaling}}
            # if test_loss:
            #     loss_d['Test_GNN'] = test_loss
            #     acc_d['Test_GNN'] = test_accuracy
            #     acc_d = {**acc_d, **{f'Test_GNN_absolute_{k}': test_acc_absolute[k] for k in test_acc_absolute}}
            #     acc_d = {**acc_d, **{f'Test_GNN_scaling_{k}': test_acc_scaling[k] for k in test_acc_scaling}}
            writer.add_scalars('Loss', loss_d, epoch)
            writer.add_scalars('Accuracy', acc_d, epoch)
        scheduler.step()
        print("lr:", round(scheduler.get_last_lr()[0], 8))

    # we save the model
    torch.save(model, os.path.join(args.work_dir, 'GNN_Model.pt'))
    if not debug:
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
