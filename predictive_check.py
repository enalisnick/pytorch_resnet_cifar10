import argparse
import os
import shutil
import time
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')



def softmax(x):
    x = np.exp(x)
    return x * 1/x.sum(axis=1)[np.newaxis].T


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    expected_probs = train(train_loader, model)
    print(expected_probs)
    print(np.sum(expected_probs))
    pkl.dump(expected_probs, open("predCheck_artifacts/predCheck_HeScaling_3_wBN.pkl", "wb"))
    
    

def train(train_loader, model):

    prob_tracker = np.zeros((10,))
    batch_counter = 0

    # need train mode to activate BatchNorm
    model.train() 

    for i, (input, target) in enumerate(train_loader):

        target = target
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        output = output.data.numpy()

        prob_tracker += np.sum(softmax(output), axis=0)
        batch_counter += 1

    return prob_tracker / (batch_counter * args.batch_size)



if __name__ == '__main__':
    main()
