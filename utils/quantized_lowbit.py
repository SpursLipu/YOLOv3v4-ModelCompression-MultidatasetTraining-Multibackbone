# Author:LiPu
# Author:LiPu
import math
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


# 定义前向传播，反向传播三值化函数
class Ternarize(Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    # 使用静态方法定义三值激活类
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = input.new(input.size())
        output[input > 0.5] = 1

        # 由于暂时不知道pytorch如何进行与运算，用此段代码实现
        # output[input>=-0.5 and input<=0.5]
        temp = torch.add((input >= -0.5), (input <= 0.5))
        temp[temp == 2] = 1
        temp[temp == 1] = 0
        output[temp] = 0

        output[input < -0.5] = -1
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


# 定义前向传播，反向传播二值化函数
class Binarize(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = 0
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


binarize = Binarize.apply

ternarize = Ternarize.apply


# 重载LeakyRelu
class BinaryLeakyReLU(nn.LeakyReLU):
    def __init__(self):
        super(BinaryLeakyReLU, self).__init__()

    def forward(self, input):
        output = EQ(input)
        return output


# 对线性层权重做量化，必须有reset_parameters函数
class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = ternarize(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


# BWN量化
class BWNConv2d(nn.Conv2d):

    def forward(self, input):
        bw = binarize(self.weight)
        alpha = torch.div(self.weight.norm(1), torch.numel(self.weight))
        output = alpha * (F.conv2d(input, bw, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups))
        return output

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


# BNN量化
class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        # bw = (self.weight - torch.mean(self.weight)) / torch.sqrt(torch.std(self.weight))
        bw = binarize(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


