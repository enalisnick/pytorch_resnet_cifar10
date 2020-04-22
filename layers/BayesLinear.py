import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



class BBBLinear(nn.Module):
    
    def __init__(self, in_features, out_features, tau_shape=(1, 1), scale_init=1., bias=True, name='BBBLinear'):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_shape = tau_shape
        
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.tau = scale_init
        
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.name = name


    def reset_parameters(self):
        #stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.zero_()  #uniform_(-stdv, stdv)
        #self.log_tau.data.fill_(log_tau_init)
        if self.bias is not None:
            self.bias.data.zero_()

            
    def forward(self, x):

        # sample weight
        #if self.training:
        epsilon = self.W.data.new(self.W.size()).normal_()
        #else:
        #    epsilon = 0.0

        w_hat = self.W + self.tau * epsilon

        out = F.linear(x, w_hat)
        if self.bias is not None:
            out = out + self.bias

        return out
