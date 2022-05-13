import argparse
import json
import os
import time
from random import random

import numpy as np
import torch
from torch.utils.data import DataLoader

def parse_args(_args=None):
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--work_dir', default='work_dir/', type=str)
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
    model.train() # model.train needs to be activated since the test method sets it to model.test. This is needed for batch normalization

    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for i, data in enumerate(train_loader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # Move data to target device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        points, has_square = model(inputs)
        square_in_image = torch.any(labels != 0, dim=1)

        points_in_image = points[square_in_image]
        labels_in_image = labels[square_in_image]

        loss = criterion(points_in_image, labels_in_image)

        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
        total += labels.size(0)
        # correct += torch.isclose(points_in_image.data, labels_in_image, atol=2).sum().item() // 8

    return avg_loss / len(train_loader)#, 100 * correct / total

def load_data(folder="../data/dzn/", training_size=0.8):
    input_data = np.load('../data/dzn/dzn.npy', allow_pickle=True)
    solved = np.load('../data/solved/dzn_output.npy', allow_pickle=True)
    input_data = np.array(input_data)
    solved = np.array(solved)
    length = len(input_data)
    training_length = int(length * training_size)
    training = (input_data[:training_length], solved[:training_length])
    test = (input_data[training_length:], solved[training_length:])
    return training, test

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

    train_set, test_set, image_width = load_data()
    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=True)

    if args.model_dir is not None:
      pass
        # agent.load(args.model_dir, args.model_step)
    # L = Logger(args.work_dir, use_tb=args.save_tb)



if __name__ == '__main__':
    main()