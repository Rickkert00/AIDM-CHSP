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
from torch.utils.tensorboard import SummaryWriter

from imitation_learning.neural_network import RemovalTimePredictor

def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--learning_rate', default=2e-3, type=float)
    parser.add_argument('--decay', default=0.999, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--work_dir', default='work_dir', type=str)
    parser.add_argument('--epochs', default=2000, type=int)
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


def _run(data_loader, model: RemovalTimePredictor, criterion, device, optimizer=None, epoch=0):
    avg_loss = 0
    correct = 0
    total = 0
    # Iterate through batches
    output = None
    for j, data in enumerate(data_loader):
        # Get the inputs; data is a list of [inputs, labels]
        graph_inputs, labels = data
        # Move data to target device
        graph_inputs, labels = graph_inputs.to(device), labels

        if optimizer: optimizer.zero_grad()

        output, edges = model(graph_inputs, graph_inputs.ndata['x'], graph_inputs.edata['w'])

        # TODO find out why all values of period tensor are equivalent
        output_index = 0
        only_period = False
        divide = 1
        if not only_period:
            padded_output = torch.zeros((graph_inputs.batch_size, len(labels[0])))

            for i in range(graph_inputs.batch_size):
                nodes = graph_inputs._batch_num_nodes['_N'][i].item()
                edges_count = nodes**2
                # padded_output[i, :nodes - 1] = output[output_index+1:output_index + nodes, 0, 0] # nodes
                prediction = edges[output_index+1:output_index + nodes, 0, 0]
                if i == 0 and j == 0:
                    test_str = 'TEST' if optimizer is None else 'TRAIN'
                    print(f"{test_str} PRED ", prediction)
                    print(f"{test_str} Answer ", labels[i, :nodes-1])
                padded_output[i, :nodes - 1] = prediction # edges

                # padded_output[i, :nodes - 1] = edges[output_index::nodes, 0, 0][:nodes-1] # edges other idea

                # padded_output[i, :nodes -2] = output[output_index+1:output_index + nodes - 1, 0]
                # output_index += nodes
                output_index += edges_count
                correct += torch.isclose(padded_output[i, :nodes - 1], labels[i, :nodes-1], atol=5).sum().item()/nodes
            loss = criterion(padded_output/divide, labels/divide)  # calculate on the last tensor as that is the final tank
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
        avg_loss += loss * divide
    # if epoch % 2 == 1:
    # print('Pred', output[:graph_inputs._batch_num_nodes['_N'][0]][:, 0, 0])
    # print('Pred edges rounded', edges[:10, 0, 0].round())
        # print('Pred', output[:graph_inputs._batch_num_nodes['_N'][0]][:,0])

        # print('Pred edges rounded', edges[:10,0,0].round())

    return avg_loss.item() / len(data_loader), 100 * correct / total


def load_graph_data(files, training_size=0.9, predict_period=False, divide_nodes=1e4, total_samples=10000):
    all_solutions_and_input = None
    for f in files:
        print("loading", f)
        solutions_and_input = np.load(f, allow_pickle=True)
        solutions_and_input = np.vstack([s for s in solutions_and_input if s is not None and s[1] is not None])[:total_samples]
        if all_solutions_and_input is None:
            all_solutions_and_input = solutions_and_input
        else:
            all_solutions_and_input = np.vstack([all_solutions_and_input, solutions_and_input]) 
        print("Shape input", all_solutions_and_input.shape)
    input_data = all_solutions_and_input[:, 0]
    solutions = all_solutions_and_input[:, 1]
    num_nodes = [torch.tensor(input['Ninner']).float() for input in input_data]

    # fully connected adj matrix with weights, including self connections
    adjs = [torch.ones((int(num_tanks.item() + 2), int(num_tanks.item() + 2))) for num_tanks in
            num_nodes]  # obtain adjacency matrix, add 2 extra as we need start and end node
    # structure format: (tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    # so structure[0][0] and structure[1][0] are connected etc.
    max = 0
    # top_layers = [torch.nonzero(torch.ones(num_tanks), as_tuple=True)+num_tanks for num_tanks in
    #         num_nodes]
    structure = list(zip([torch.nonzero(adj, as_tuple=True) for adj in
                          adjs]))  # edges list, 2 lists that indicate what nodes are connected
    graphs = [dgl.graph((u_v[0][0], u_v[0][1])) for u_v in structure]
    # offset input variable 'f' time as by 1 as f[0] is from initial stage to first tank, need to add that later
    node_feats = [[torch.tensor([0, max, input['f'][idx]]) if idx == 0
                   else torch.tensor([input['tmin'][idx - 1], input['tmax'][idx - 1], input['f'][idx]])
                   for idx in range(int(num_nodes[i].item()) + 1)]
                  for i, input in enumerate(input_data)] # need to have 3 features per node, and num_nodes nodes per problem. We add a start node already in this loop
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
        graphs[i].ndata['x'] = corr_node_feats[i]/divide_nodes
        graphs[i].edata['w'] = edge_feats[i]
    label = 'r'
    if predict_period:
        label = 'objective'
    solved = [torch.tensor(d[label][1:]+[d['objective']]).float().unsqueeze(dim=0) for d in solutions]
    length = len(graphs)
    # TODO is this a good split? maybe use random split instead of this
    training_length = int(length * training_size)
    dataset = [[graphs[i], solved[i]] for i in range(length)]
    training, test = torch.utils.data.random_split(dataset, [training_length, length-training_length])
    return training, test, input_data[training.indices]

def save(model, model_dir, epoch):
    torch.save(
        model.state_dict(), '%s/model_%s.pt' % (model_dir, epoch)
    )
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

# best so far May27_23-04-50
def main(_args=None):
    debug = True
    print("args", _args)
    args = parse_args(_args)
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    exp_id = 'random_split_big3'
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

    device = torch.device('cuda' if torch.cuda.is_available() and not debug or True else 'cpu')
    # device = torch.device('cpu')  # Use cpu to debug faster
    print("Using device:", device)
    base_path = 'chsp-generators-main/instances/'
    files = [base_path+f"linear_solutions_{i}.npy" for i in range(1,9)]
    train_set, test_set, _ = load_graph_data(files)
    for i, t in enumerate(train_set):
        train_set[i][0] = t[0].to(device)
        train_set[i][1] = t[1].to(device)
    for i, t in enumerate(test_set):
        test_set[i][0] = t[0].to(device)
        test_set[i][1] = t[1].to(device)
    print(train_set[0][0].device)
    train_loader = dgl.dataloading.GraphDataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn,
                                                   shuffle=True, drop_last=True)
    print('len training', len(train_set))

    # test_set = test_set[len(test_set)//args.batch_size*args.batch_size]
    print('test_set', len(test_set))
    test_loader = dgl.dataloading.GraphDataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn,
                                                   shuffle=True, drop_last=True)
    run_test = True
    if not debug:
        writer = SummaryWriter()

    model = RemovalTimePredictor()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=args.decay)
    train_time_avg = []
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()  # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization
        train_loss, train_accuracy = _run(train_loader, model, criterion, device, optimizer, epoch=epoch)
        train_time = time.time() - start_time
        train_time_avg.append(train_time)

        print('train step seconds:', mean(train_time_avg).round(3))
        test_loss, test_accuracy = None, None
        if run_test and epoch % 4 == 1:
            model.eval()
            with torch.no_grad():
                test_loss, test_accuracy = _run(test_loader, model, criterion, device)

        print("epoch", epoch, "train_loss", round(train_loss, 1), "train_accuracy", round(train_accuracy, 2))

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
            if epoch >= 2 and epoch % 5 == 0:
                save(model, model_dir=args.work_dir, epoch=epoch)
        scheduler.step()
        print("lr:", round(scheduler.get_last_lr()[0], 8))
    if not debug:
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()