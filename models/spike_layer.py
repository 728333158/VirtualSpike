import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
class WrapperFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params, forward, backward):
        ctx.backward = backward
        pack, output = forward(input)
        ctx.save_for_backward(*pack)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward = ctx.backward
        pack = ctx.saved_tensors
        grad_input = backward(grad_output, *pack)
        return grad_input, None, None, None

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x
    
class LIFNode(SpikeModule):
    '''Generates spikes based on LIF module.
    '''
    def __init__(self, step):
        super(LIFNode, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.tau_m = 5/3
        self.forward = lambda input: WrapperFunction.apply(
            input, None, self.manual_forward, self.manual_backward)

    def manual_forward(self, x):
        V = torch.zeros_like(x[0])
        V_seq = torch.zeros_like(x)
        output = torch.zeros_like(x)
        dv = torch.zeros_like(x)
        tau_m = self.tau_m
        threshold = self.V_th
        T, B, C, H, W = x.shape
        for t in range(0, T, 1):
            dv[t] = - 1 / tau_m * V + x[t]
            V = V + dv[t]
            V_seq[t] = V
            
            output[t] = (V >= threshold).float() - (V <= -threshold).float()
            V = (1 - torch.abs(output[t])) * V
        return (output, V_seq, dv), output

    def manual_backward(self, grad_output, output, V_seq, dv):
        
        lambda_V = torch.zeros_like(output[0])
        output_term = torch.zeros_like(output[0])
        jump_term = torch.zeros_like(output[0])
        grad_input = torch.zeros_like(output)
        grad = torch.zeros_like(output[0])
        
        k = self.tau_m
        T = output.shape[0]
        Vth = self.V_th
        alpha = 8.0
        expV = torch.exp(-alpha * (torch.abs(V_seq[T-1]) - Vth) )
        grad_H = alpha * expV / (1 + expV) / (1 + expV)
        
        grad_H[V_seq[T-1]< -Vth - 1] = 0.0
        grad_H[V_seq[T-1]> Vth + 1] = 0.0
        
        lambda_V = lambda_V - grad_output[T-1] * grad_H
        grad_input[T-1] = -lambda_V

        for t in range(T-2, -1, -1):
            lambda_V = (1 - 1 / k) * lambda_V
            grad = output[t] / (dv[t])
            grad[grad!=grad] = 0.0
            grad = torch.clamp(grad, -2.0, 2.0)
            if t == 0:
                jump_term = grad * grad_output[t] / k
                output_term = grad * lambda_V * (dv[t+1])
            else:
                jump_term = grad * (torch.abs(output[t]) - torch.abs(output[t-1])) * grad_output[t] / k
                output_term = grad * (torch.abs(output[t]) - torch.abs(output[t-1])) * lambda_V * (dv[t+1])

            lambda_V = (1 - torch.abs(output[t])) * lambda_V + output_term - jump_term
            grad_input[t] = -lambda_V * k
        
        return grad_input

class SpikeConv(SpikeModule):


    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.conv(x)
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out

class SpikePool(SpikeModule):

    def __init__(self, pool, step=2):
        super().__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.pool(x)
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = self.pool(out)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out

class myBatchNorm3d(SpikeModule):
    def __init__(self, BN: nn.BatchNorm2d, step=2):
        super().__init__()
        self.bn = nn.BatchNorm3d(BN.num_features)
        self.step = step
    def forward(self, x):
        if self._spiking is not True:
            return BN(x)
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out

class tdBatchNorm2d(nn.BatchNorm2d, SpikeModule):
    """Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, bn: nn.BatchNorm2d, alpha: float):
        super(tdBatchNorm2d, self).__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.alpha = alpha
        self.V_th = 0.5
        # self.weight.data = bn.weight.data
        # self.bias.data = bn.bias.data
        # self.running_mean.data = bn.running_mean.data
        # self.running_var.data = bn.running_var.data

    def forward(self, input):
        if self._spiking is not True:
            # compulsory eval mode for normal bn
            self.training = False
            return super().forward(input)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        channel_dim = input.shape[2]
        input = self.alpha * self.V_th * (input - mean.reshape(1, 1, channel_dim, 1, 1)) / \
                (torch.sqrt(var.reshape(1, 1, channel_dim, 1, 1) + self.eps))
        if self.affine:
            input = input * self.weight.reshape(1, 1, channel_dim, 1, 1) + self.bias.reshape(1, 1, channel_dim, 1, 1)

        return input

class Vth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = (input >= 0).float()
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_H = torch.zeros_like(grad_output)
        grad_input = torch.zeros_like(grad_output)
        V, = ctx.saved_tensors
        expV = torch.exp(-4.0 * V)
        grad_H = 4.0 * expV / (1 + expV) / (1 + expV)
        grad_H[V > 7.5 / 4.0] = 0.
        grad_H[V < -7.5 / 4.0] = 0.
        
        if torch.isnan(grad_H).any() or torch.isinf(grad_H).any():
            pdb.set_trace()
            
        grad_input = grad_output * grad_H
        return grad_input

    def __init__(self, step):
        super(LIFNode, self).__init__()
        
class LIFNode_SG(SpikeModule):
    def __init__(self, step):
        super().__init__()
        self.threshold = 1.0
        self.step = step
        self.tau_m = 5/3

    def forward(self, x):
        V = torch.zeros_like(x[0])
        output = torch.zeros_like(x)
        threshold = self.threshold
        tau_m = self.tau_m
        T = x.shape[0]
        for t in range(0, T, 1):
            V = (1 - 1 / tau_m) * V + x[t]
            output[t] = Vth.apply(V - threshold) - Vth.apply(-threshold-V)
            V = (1 - F.relu(output[t]) - F.relu(-output[t])) * V
        return output

class LIFNode_Binary(SpikeModule):
    def __init__(self, step):
        super().__init__()
        self.threshold = 1.0
        self.step = step
        self.tau_m = 2.0

    def forward(self, x):
        V = torch.zeros_like(x[0])
        output = torch.zeros_like(x)
        threshold = self.threshold
        tau_m = self.tau_m
        T = x.shape[0]
        for t in range(0, T, 1):
            V = (1 - 1 / tau_m) * V + x[t]
            output[t] = Vth.apply(V - threshold)
            V = (1 - output[t]) * V
        return output