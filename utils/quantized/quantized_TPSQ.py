# Author:LiPu
import numpy as np
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Search_Pow2(Function):

    @staticmethod
    def forward(self, input):
        ceil_float_range = 2 ** input.log2().ceil()
        floor_float_range = 2 ** input.log2().floor()
        if abs(ceil_float_range - input) < abs(floor_float_range - input):
            output = ceil_float_range
        else:
            output = floor_float_range
        return output

    @staticmethod
    def backward(self, grad_output):
        # 线性
        grad_input = 0.8985 * (grad_output.clone())
        # 多项式
        # temp = grad_output.clone()
        # grad_input = -0.668 * temp + 1.335

        return grad_input


class Quantizer(nn.Module):
    def __init__(self, bits, out_channels, FPGA):
        super().__init__()
        self.bits = bits
        self.FPGA = FPGA
        self.first = True
        self.momentum = 0.1

        self.out_channels = out_channels
        if self.out_channels == -1:
            self.register_buffer('min_val', torch.zeros(1))
            self.register_buffer('max_val', torch.zeros(1))
            self.scale = Parameter(torch.Tensor(1))  # 量化比例因子
        else:
            self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
            self.scale = Parameter(torch.Tensor(self.out_channels, 1, 1, 1))  # 量化比例因子
        init.ones_(self.scale)

    # 截断
    def clamp(self, input):
        if self.FPGA:
            output = 0.5 * (
                    torch.abs(input + Search_Pow2.apply(self.scale)) - torch.abs(input - Search_Pow2.apply(self.scale)))
        else:
            output = 0.5 * (
                    torch.abs(input + self.scale) - torch.abs(input - self.scale))
        return output

    # 量化
    def quantize(self, input):
        quantized_range = torch.tensor((1 << (self.bits - 1)) - 1)
        if self.FPGA:
            output = (input * quantized_range) / Search_Pow2.apply(self.scale)
        else:
            output = (input * quantized_range) / self.scale
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # 反量化
    def dequantize(self, input):
        quantized_range = torch.tensor((1 << (self.bits - 1)))
        if self.FPGA:
            output = (input * Search_Pow2.apply(self.scale)) / quantized_range
        else:
            output = (input * self.scale) / quantized_range
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            output = self.clamp(input)  # 截断
            output = self.quantize(output)  # 量化
            output = self.round(output)
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


class RangeTracker(nn.Module):
    def __init__(self):
        super().__init__()

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        min_val = torch.min(input)
        max_val = torch.max(input)
        self.update_range(min_val, max_val)


class GlobalRangeTracker(RangeTracker):  # W,min_max_shape=(N, 1, 1, 1),channel级,取本次和之前相比的min_max —— (N, C, W, H)
    def __init__(self):
        super().__init__()
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))


class Bias_Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', torch.zeros(1))  # 量化比例因子

    def update_params(self):
        raise NotImplementedError

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
                self.range_tracker(input)
                self.update_params()
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

    ################获得量化因子所对应的移位数
    def get_scale(self):
        #############移位修正
        move_scale = math.log2(self.scale)
        move_scale = np.array(move_scale).reshape(1, -1)
        return move_scale


# 对称量化
class SymmetricQuantizer(Bias_Quantizer):

    def update_params(self):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)

        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val))  # 量化后范围

        float_max = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))  # 量化前范围
        floor_float_range = 2 ** float_max.log2().floor()
        ceil_float_range = 2 ** float_max.log2().ceil()
        if abs(ceil_float_range - float_max) < abs(floor_float_range - float_max):
            float_range = ceil_float_range
        else:
            float_range = floor_float_range
        self.scale = float_range / quantized_range  # 量化比例因子


# ********************* 量化卷积（同时量化A/W，并做卷积） *********************
class TPSQ_QuantizedConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            a_bits=8,
            w_bits=8):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = Quantizer(bits=a_bits, out_channels=-1, FPGA=False)
        self.weight_quantizer = Quantizer(bits=w_bits, out_channels=-1, FPGA=False)

    def forward(self, input):
        # 量化A和W
        if input.shape[1] != 3:
            input = self.activation_quantizer(input)
        q_weight = self.weight_quantizer(self.weight)
        # 量化卷积
        output = F.conv2d(
            input=input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


# ********************* bn融合_量化卷积（bn融合后，同时量化A/W，并做卷积） *********************
class TPSQ_BNFold_QuantizedConv2d_For_FPGA(TPSQ_QuantizedConv2d):
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
            steps=0,
            quantizer_output=False,
            maxabsscaler=False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.bn = bn
        self.activate = activate
        self.eps = eps
        self.momentum = momentum
        self.freeze_step = int(steps * 0.9)
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.zeros(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        self.register_buffer('step', torch.zeros(1))
        self.quantizer_output = quantizer_output
        self.maxabsscaler = maxabsscaler
        init.normal_(self.gamma, 1, 0.5)
        init.zeros_(self.beta)

        self.activation_quantizer = Quantizer(bits=a_bits, out_channels=-1, FPGA=True)
        self.weight_quantizer = Quantizer(bits=w_bits, out_channels=-1, FPGA=True)
        self.bias_quantizer = SymmetricQuantizer(bits=w_bits,
                                                 range_tracker=GlobalRangeTracker())

    def forward(self, input):
        # 训练态
        if self.training:
            self.step += 1
            if self.bn:
                # 先做普通卷积得到A，以取得BN参数
                output = F.conv2d(
                    input=input,
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                # 更新BN统计参数（batch和running）
                dims = [dim for dim in range(4) if dim != 1]
                self.batch_mean = torch.mean(output, dim=dims)
                self.batch_var = torch.var(output, dim=dims)
                with torch.no_grad():
                    if self.first_bn == 0 and torch.equal(self.running_mean, torch.zeros_like(
                            self.running_mean)) and torch.equal(self.running_var, torch.zeros_like(self.running_var)):
                        self.first_bn.add_(1)
                        self.running_mean.add_(self.batch_mean)
                        self.running_var.add_(self.batch_var)
                    else:
                        self.running_mean.mul_(1 - self.momentum).add_(self.batch_mean * self.momentum)
                        self.running_var.mul_(1 - self.momentum).add_(self.batch_var * self.momentum)
                # BN融合
                if self.step < self.freeze_step:
                    if self.bias is not None:
                        bias = reshape_to_bias(
                            self.beta + (self.bias - self.batch_mean) * (
                                    self.gamma / torch.sqrt(self.batch_var + self.eps)))
                    else:
                        bias = reshape_to_bias(
                            self.beta - self.batch_mean * (
                                    self.gamma / torch.sqrt(self.batch_var + self.eps)))  # b融batch
                    weight = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(self.batch_var + self.eps))  # w融running
                else:
                    if self.bias is not None:
                        bias = reshape_to_bias(
                            self.beta + (self.bias - self.running_mean) * (
                                    self.gamma / torch.sqrt(self.running_var + self.eps)))
                    else:
                        bias = reshape_to_bias(
                            self.beta - self.running_mean * (
                                    self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融batch
                    weight = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
            else:
                bias = self.bias
                weight = self.weight
        # 测试态
        else:
            # print(self.running_mean, self.running_var)
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
            if not os.path.isdir('./quantizer_output'):
                os.makedirs('./quantizer_output')

            if not os.path.isdir('./quantizer_output/q_weight_out'):
                os.makedirs('./quantizer_output/q_weight_out')
            if not os.path.isdir('./quantizer_output/w_scale_out'):
                os.makedirs('./quantizer_output/w_scale_out')
            if not os.path.isdir('./quantizer_output/q_weight_max'):
                os.makedirs('./quantizer_output/q_weight_max')
            if not os.path.isdir('./quantizer_output/max_weight_count'):
                os.makedirs('./quantizer_output/max_weight_count')
            #######################输出当前层的权重量化因子
            weight_scale = self.weight_quantizer.get_scale()
            np.savetxt(('./quantizer_output/w_scale_out/%f.txt' % time.time()), weight_scale, delimiter='\n')
            #######################输出当前层的量化权重
            q_weight_txt = self.weight_quantizer.get_quantize_value(weight)
            q_weight_txt = np.array(q_weight_txt.cpu()).reshape(1, -1)
            q_weight_max = [np.max(q_weight_txt)]
            # q_weight_max = np.argmax(q_weight_txt)
            max_weight_count = [np.sum(abs(q_weight_txt) >= 127)]  # 统计该层溢出的数目
            np.savetxt(('./quantizer_output/max_weight_count/%f.txt' % time.time()), max_weight_count)
            np.savetxt(('./quantizer_output/q_weight_max/%f.txt' % time.time()), q_weight_max)
            np.savetxt(('./quantizer_output/q_weight_out/%f.txt' % time.time()), q_weight_txt, delimiter='\n')
            # io.savemat('save.mat',{'q_weight_txt':q_weight_txt})

            #######################创建输出偏置txt的文件夹
            if not os.path.isdir('./quantizer_output/q_bias_out'):
                os.makedirs('./quantizer_output/q_bias_out')
            if not os.path.isdir('./quantizer_output/b_scale_out'):
                os.makedirs('./quantizer_output/b_scale_out')
            #######################输出当前层偏置的量化因子
            bias_scale = self.bias_quantizer.get_scale()
            np.savetxt(('./quantizer_output/b_scale_out/%f.txt' % time.time()), bias_scale, delimiter='\n')
            #######################输出当前层的量化偏置
            q_bias_txt = self.bias_quantizer.get_quantize_value(bias)
            q_bias_txt = np.array(q_bias_txt.cpu()).reshape(1, -1)
            np.savetxt(('./quantizer_output/q_bias_out/%f.txt' % time.time()), q_bias_txt, delimiter='\n')

        # 量化卷积
        if self.training:  # 训练态
            output = F.conv2d(
                input=input,
                weight=q_weight,
                bias=q_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )

        else:  # 测试态
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
            output = F.leaky_relu(output, 0.125 if not self.maxabsscaler else 0.25, inplace=True)
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

            if not os.path.isdir('./quantizer_output/q_activation_out'):
                os.makedirs('./quantizer_output/q_activation_out')
            if not os.path.isdir('./quantizer_output/a_scale_out'):
                os.makedirs('./quantizer_output/a_scale_out')
            if not os.path.isdir('./quantizer_output/q_activation_max'):
                os.makedirs('./quantizer_output/q_activation_max')
            if not os.path.isdir('./quantizer_output/max_activation_count'):
                os.makedirs('./quantizer_output/max_activation_count')
            ##################输出当前激活的量化因子
            activation_scale = self.activation_quantizer.get_scale()
            np.savetxt(('./quantizer_output/a_scale_out/%f.txt' % time.time()), activation_scale, delimiter='\n')
            ##################输出当前层的量化激活
            q_activation_txt = self.activation_quantizer.get_quantize_value(output)
            q_activation_txt = np.array(q_activation_txt.cpu()).reshape(1, -1)
            q_activation_max = [np.max(q_activation_txt)]  # 统计该层的最大值(即查看是否有溢出)
            max_activation_count = [np.sum(abs(q_activation_txt) >= 127)]  # 统计该层溢出的数目
            # q_weight_max = np.argmax(q_weight_txt)
            np.savetxt(('./quantizer_output/max_activation_count/%f.txt' % time.time()),
                       max_activation_count)
            np.savetxt(('./quantizer_output/q_activation_max/%f.txt' % time.time()), q_activation_max)
            np.savetxt(('./quantizer_output/q_activation_out/%f.txt' % time.time()), q_activation_txt,
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
