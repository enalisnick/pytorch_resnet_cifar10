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
import bayesResNet



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')



def main():
    global args, best_prec1
    args = parser.parse_args()

    # define nested models
    base_model = torch.nn.DataParallel(resnet.__dict__[args.arch]('base'))
    extended_model = torch.nn.DataParallel(resnet.__dict__[args.arch]('extended'))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    tau_vals, pi_tau_vals  = get_prior(train_loader, base_model, extended_model)
    print(expected_probs)
    print(np.sum(expected_probs))
    pkl.dump(expected_probs, open("predCheck_HeScaling_temp.pkl", "wb"))
    
    

def train(train_loader, base_model, extended_model, n_samples=10):

    # define KLD
    kld = nn.KLDivLoss(reduction='none')
    kld_accum = Variable(torch.zeros(1), requires_grad=True)

    # need train mode to activate BatchNorm                                                                                                                                                           
    base_model.train()
    extended_model.train()

    for i, (input, target) in enumerate(train_loader):

        # compute base's output
        base_output = base_model(input)
        base_log_probs = nn.LogSoftmax(base_output)

        # define local KLD
        batch_KLD_accum = Variable(torch.zeros(args.batch_size), requires_grad=True)

        # draw samples to marginalize 1st level
        for s_idx in range(n_samples):

            # compute output
            extended_output = extended_model(output)
            extended_probs = nn.Softmax(extended_output)

            # compute KLD[p+ || p0]
            batch_KLD_accum += torch.sum(kld(input=base_log_probs, target=extended_probs), dim=1)

        # compute MC approx of batch KLD
        batch_KLD_accum = batch_KLD_accum / n_samples

        # add batch KLD to running total
        kld_accum += torch.sum(batch_KLD_accum)

    # divide by total data points
    kld_accum = kld_accum / ((i+1) * args.batch_size)

    # compute volume element
    kld_accum.backward()
    dKLD_dTau = extended_model.tau.grad

    # compute prior
    prior_val = kl_prior() * np.abs(dKLD_dTau)
        
    extended_output = output.data.numpy()

    return prob_tracker / (batch_counter * args.batch_size)



if __name__ == '__main__':
    main()
OB
