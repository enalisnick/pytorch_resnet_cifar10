import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class BBBConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, scale_init=1., stride=1, padding=0, dilation=1, bias=True, name='BBBConv2d'):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        #self.tau_shape = tau_shape
        self.groups = 1

        # mean parameter
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        # log-scale parameter
        self.tau = scale_init

        # bias
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
            
        # output functions
        self.out_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride, self.padding, self.dilation, self.groups)

        
        self.reset_parameters()
        self.name = name
        

    def reset_parameters(self):
        #n = self.in_channels
        #for k in self.kernel_size:
        #    n *= k
        #stdv = 1. / math.sqrt(n)
        self.weight.data.zero_()  #uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()  #uniform_(-stdv, stdv)
        #self.log_tau.data.fill_(-5.0)

        
    def forward(self, x):

        ### sample weights
        
        #if self.training:
        epsilon = self.weight.data.new(self.weight.size()).normal_()
        #else:
        #    epsilon = 0.0
            
        w_hat = self.weight + self.tau * epsilon

        out = self.out_bias(x, w_hat)

        ## TO DO: Implement local reparameterization trick
         
        return out
