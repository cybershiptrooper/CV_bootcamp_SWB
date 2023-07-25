import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

def seed_everything(seed=0):
    '''
    Seeds basic parameters for reproductibility of results
    
    :param seed: (int) 
    '''
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed) # Commented as it can slow down training
    #torch.backends.cudnn.deterministic = True # Commented as it can slow down training
    #torch.backends.cudnn.benchmark = True

def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    '''
    Evaluate the model on the test set
    :param model: the model to evaluate
    :param test_loader: the test data loader
    :return: the accuracy
    '''
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # move to device
            output = model(data) # forward pass
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # get the number of correct predictions
    return correct / len(test_loader.dataset) # return the accuracy


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> float:
    '''
    Train the model for one epoch
    :param model: the model to train
    :param train_loader: the training data loader
    :param optimizer: the optimizer to use
    :param loss_fn: the loss function to use
    :return: the average loss
    '''
    model.train()
    loss_sum = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device) # move to device
        optimizer.zero_grad() # zero the parameter gradients
        output = model(data) # forward pass
        loss = loss_fn(output, target) # calculate loss
        loss_sum += loss.item()
        loss.backward() # backprop
        optimizer.step() # update weights
    return loss_sum / len(train_loader.dataset)