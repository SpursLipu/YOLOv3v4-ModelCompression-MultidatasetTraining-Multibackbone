# Author:LiPu
# Author:LiPu
import math
import time
import numpy as np
import pandas as pd
import scipy.io as io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output


class Quantizer(nn.Module):
    def __init__(self, bits, out_channels):
        super().__init__()
        self.bits = bits
        if out_channels == -1:
            self.register_buffer('scale', torch.zeros(1))  # 量化比例因子
            self.register_buffer('float_range', torch.zeros(1))
        else:
            self.register_buffer('scale', torch.zeros(out_channels, 1, 1, 1))  # 量化比例因子
            self.register_buffer('float_range', torch.zeros(out_channels, 1, 1, 1))
        self.scale_list = [0 for i in range(bits)]

    def update_params(self, step):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val))  # 量化后范围
        temp = self.float_range
        self.float_range.add_(-temp).add_(2 ** step)
        self.scale = self.float_range / quantized_range  # 量化比例因子

    # 量化
    def quantize(self, input):
        output = input / self.scale
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # 截断
    def clamp(self, input):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        output = torch.clamp(input, min_val, max_val)
        return output

    # 反量化
    def dequantize(self, input):
        output = (input) * self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if self.training == True:
                max_metrics = -1
                max_step = 0
                for i in range(self.bits):
                    self.update_params(i)
                    output = self.quantize(input)  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output)  # 反量化
                    cosine_similarity = torch.cosine_similarity(input.view(-1), output.view(-1), dim=0)
                    if cosine_similarity > max_metrics:
                        max_metrics = cosine_similarity
                        max_step = i
                self.scale_list[max_step] += 1
                Global_max_step = self.scale_list.index(max(self.scale_list))
                self.update_params(Global_max_step)

            output = self.quantize(input)  # 量化
            output = self.round(output)
            output = self.clamp(output)  # 截断
            output = self.dequantize(output)  # 反量化
            return output

    def get_quantize_value(self, input):

        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            output = self.quantize(input)  # 量化
            output = self.round(output)
            output = self.clamp(output)  # 截断
        return output

    '''def get_bias_scale(self):
        
        #############移位修正
        move_scale = math.log2(self.scale)
        a = np.array(move_scale).reshape(1, -1)
        np.savetxt(('./b_scale_out/scale %f.txt' % time.time()), a, delimiter='\n')'''

    ################获得量化因子所对应的移位数
    def get_scale(self):
        #############移位修正
        move_scale = math.log2(self.scale)
        move_scale = np.array(move_scale).reshape(1, -1)
        return move_scale


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


# ********************* bn融合_量化卷积（bn融合后，同时量化A/W，并做卷积） *********************


class BNFold_COSPTQuantizedConv2d_For_FPGA(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            eps=1e-5,
            momentum=0.01,  # 考虑量化带来的抖动影响,对momentum进行调整(0.1 ——> 0.01),削弱batch统计参数占比，一定程度抑制抖动。经实验量化训练效果更好,acc提升1%左右
            a_bits=8,
            w_bits=8,
            bn=0,
            activate='leaky',
            quantizer_output=False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = bn
        self.activate = activate
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.zeros(out_channels))
        self.quantizer_output = quantizer_output

        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = Quantizer(bits=a_bits, out_channels=-1)
        self.weight_quantizer = Quantizer(bits=w_bits, out_channels=-1)
        self.bias_quantizer = Quantizer(bits=w_bits, out_channels=-1)

    def forward(self, input):
        if self.bn:
            # BN融合
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                        self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(
                    self.beta - self.running_mean * self.gamma / torch.sqrt(
                        self.running_var + self.eps))  # b融running
            weight = self.weight * reshape_to_weight(
                self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
        else:
            bias = self.bias
            weight = self.weight

        # 量化A和bn融合后的W
        q_weight = self.weight_quantizer(weight)
        q_bias = self.bias_quantizer(bias)

        if self.quantizer_output == True:  # 输出量化参数txt文档

            # 创建的quantizer_output输出文件夹
            if not os.path.isdir('./quantier_output'):
                os.makedirs('./quantier_output')

            if not os.path.isdir('./quantier_output/q_weight_out'):
                os.makedirs('./quantier_output/q_weight_out')
            if not os.path.isdir('./quantier_output/w_scale_out'):
                os.makedirs('./quantier_output/w_scale_out')
            if not os.path.isdir('./quantier_output/q_weight_max'):
                os.makedirs('./quantier_output/q_weight_max')
            if not os.path.isdir('./quantier_output/max_weight_count'):
                os.makedirs('./quantier_output/max_weight_count')
            #######################输出当前层的权重量化因子
            weight_scale = self.weight_quantizer.get_scale()
            np.savetxt(('./quantier_output/w_scale_out/scale %f.txt' % time.time()), weight_scale, delimiter='\n')
            #######################输出当前层的量化权重
            q_weight_txt = self.weight_quantizer.get_quantize_value(weight)
            q_weight_txt = np.array(q_weight_txt.cpu()).reshape(1, -1)
            q_weight_max = [np.max(q_weight_txt)]
            # q_weight_max = np.argmax(q_weight_txt)
            max_weight_count = [np.sum(abs(q_weight_txt) >= 127)]  # 统计该层溢出的数目
            np.savetxt(('./quantier_output/max_weight_count/max_weight_count %f.txt' % time.time()), max_weight_count)
            np.savetxt(('./quantier_output/q_weight_max/max_weight %f.txt' % time.time()), q_weight_max)
            np.savetxt(('./quantier_output/q_weight_out/weight %f.txt' % time.time()), q_weight_txt, delimiter='\n')
            # io.savemat('save.mat',{'q_weight_txt':q_weight_txt})

            #######################创建输出偏置txt的文件夹
            if not os.path.isdir('./quantier_output/q_bias_out'):
                os.makedirs('./quantier_output/q_bias_out')
            if not os.path.isdir('./quantier_output/b_scale_out'):
                os.makedirs('./quantier_output/b_scale_out')
            #######################输出当前层偏置的量化因子
            bias_scale = self.bias_quantizer.get_scale()
            np.savetxt(('./quantier_output/b_scale_out/scale %f.txt' % time.time()), bias_scale, delimiter='\n')
            #######################输出当前层的量化偏置
            q_bias_txt = self.bias_quantizer.get_quantize_value(bias)
            q_bias_txt = np.array(q_bias_txt.cpu()).reshape(1, -1)
            np.savetxt(('./quantier_output/q_bias_out/bias %f.txt' % time.time()), q_bias_txt, delimiter='\n')

        # 量化卷积
        output = F.conv2d(
            input=input,
            weight=q_weight,
            bias=q_bias,  # 注意，这里加bias，做完整的conv+bn
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        if self.activate == 'leaky':
            output = F.leaky_relu(output, 0.125, inplace=True)
        elif self.activate == 'relu6':
            output = F.relu6(output, inplace=True)
        elif self.activate == 'h_swish':
            output = output * (F.relu6(output + 3.0, inplace=True) / 6.0)
        elif self.activate == 'relu':
            output = F.relu(output, inplace=True)
        elif self.activate == 'mish':
            output = output * F.softplus(output).tanh()
        elif self.activate == 'linear':
            # return output
            pass
        else:
            print(self.activate + "%s is not supported !")

        if self.quantizer_output == True:

            if not os.path.isdir('./quantier_output/q_activation_out'):
                os.makedirs('./quantier_output/q_activation_out')
            if not os.path.isdir('./quantier_output/a_scale_out'):
                os.makedirs('./quantier_output/a_scale_out')
            if not os.path.isdir('./quantier_output/q_activation_max'):
                os.makedirs('./quantier_output/q_activation_max')
            if not os.path.isdir('./quantier_output/max_activation_count'):
                os.makedirs('./quantier_output/max_activation_count')
            ##################输出当前激活的量化因子
            activation_scale = self.activation_quantizer.get_scale()
            np.savetxt(('./quantier_output/a_scale_out/scale %f.txt' % time.time()), activation_scale, delimiter='\n')
            ##################输出当前层的量化激活
            q_activation_txt = self.activation_quantizer.get_quantize_value(output)
            q_activation_txt = np.array(q_activation_txt.cpu()).reshape(1, -1)
            q_activation_max = [np.max(q_activation_txt)]  # 统计该层的最大值(即查看是否有溢出)
            max_activation_count = [np.sum(abs(q_activation_txt) >= 127)]  # 统计该层溢出的数目
            # q_weight_max = np.argmax(q_weight_txt)
            np.savetxt(('./quantier_output/max_activation_count/max_activation_count %f.txt' % time.time()),
                       max_activation_count)
            np.savetxt(('./quantier_output/q_activation_max/max_activation %f.txt' % time.time()), q_activation_max)
            np.savetxt(('./quantier_output/q_activation_out/activation %f.txt' % time.time()), q_activation_txt,
                       delimiter='\n')

        output = self.activation_quantizer(output)
        return output

    def BN_fuse(self):
        if self.bn:
            # BN融合
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                        self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(
                    self.beta - self.running_mean * self.gamma / torch.sqrt(
                        self.running_var + self.eps))  # b融running
            weight = self.weight * reshape_to_weight(
                self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
        else:
            bias = self.bias
            weight = self.weight
        return weight, bias
