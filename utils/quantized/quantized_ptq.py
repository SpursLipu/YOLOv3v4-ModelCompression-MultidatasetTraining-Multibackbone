# Author:LiPu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* range_trackers(范围统计器，统计量化前范围) *********************
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':  # A,min_max_shape=(1, 1, 1, 1),layer级
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # W,min_max_shape=(N, 1, 1, 1),channel级
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
        self.update_range(min_val, max_val)


class GlobalRangeTracker(RangeTracker):  # W,min_max_shape=(N, 1, 1, 1),channel级,取本次和之前相比的min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros(1))
            self.register_buffer('max_val', torch.zeros(1))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
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


class AveragedRangeTracker(RangeTracker):  # A,min_max_shape=(1, 1, 1, 1),layer级,取running_min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros(1))
            self.register_buffer('max_val', torch.zeros(1))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)


# ********************* quantizers（量化器，量化） *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output


class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker, out_channels, FPGA, sign=True):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.FPGA = FPGA
        self.sign = sign
        if out_channels == -1:
            self.register_buffer('scale', torch.zeros(1))  # 量化比例因子
            self.register_buffer('zero_point', torch.zeros(1))  # 量化零点
        else:
            self.register_buffer('scale', torch.zeros(out_channels, 1, 1, 1))  # 量化比例因子
            self.register_buffer('zero_point', torch.zeros(out_channels, 1, 1, 1))  # 量化零点

    def update_params(self):
        raise NotImplementedError

    # 量化
    def quantize(self, input):
        output = input / self.scale + self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # 截断
    def clamp(self, input):
        if self.sign:
            min_val = torch.tensor(-(1 << (self.bits - 1)))
            max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        if not self.sign:
            min_val = torch.tensor(0)
            max_val = torch.tensor((1 << self.bits) - 1)
        output = torch.clamp(input, min_val, max_val)
        return output

    # 反量化
    def dequantize(self, input):
        output = (input - self.zero_point) * self.scale
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


# 对称量化
class SymmetricQuantizer(Quantizer):

    def update_params(self):
        if self.sign:
            min_val = torch.tensor(-(1 << (self.bits - 1)))
            max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        else:
            min_val = torch.tensor(0)
            max_val = torch.tensor((1 << self.bits) - 1)
        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val))  # 量化后范围
        if self.FPGA == False:
            float_range = torch.max(torch.abs(self.range_tracker.min_val),
                                    torch.abs(self.range_tracker.max_val))  # 量化前范围
        else:
            float_max = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))  # 量化前范围
            floor_float_range = 2 ** float_max.log2().floor()
            ceil_float_range = 2 ** float_max.log2().ceil()
            if abs(ceil_float_range - float_max) < abs(floor_float_range - float_max):
                float_range = ceil_float_range
            else:
                float_range = floor_float_range
        self.scale = float_range / quantized_range  # 量化比例因子
        self.zero_point = torch.zeros_like(self.scale)  # 量化零点


# 非对称量化
class AsymmetricQuantizer(Quantizer):

    def update_params(self):
        if self.sign:
            min_val = torch.tensor(-(1 << (self.bits - 1)))
            max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        else:
            min_val = torch.tensor(0)
            max_val = torch.tensor((1 << self.bits) - 1)
        quantized_range = max_val - min_val  # 量化后范围
        if self.FPGA == False:
            float_range = self.range_tracker.max_val - self.range_tracker.min_val  # 量化前范围
        else:
            float_range = self.range_tracker.max_val - self.range_tracker.min_val  # 量化前范围
            ceil_float_range = 2 ** float_range.log2().ceil()
            floor_float_range = 2 ** float_range.log2().floor()
            if abs(ceil_float_range - float_range) < abs(floor_float_range - float_range):
                float_range = ceil_float_range
            else:
                float_range = floor_float_range
        self.scale = float_range / quantized_range  # 量化比例因子
        self.zero_point = torch.round(max_val - self.range_tracker.max_val / self.scale)  # 量化零点


# ********************* 量化卷积（同时量化A/W，并做卷积） *********************
class PTQuantizedConv2d(nn.Conv2d):
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
            w_bits=8,
            q_type=0):
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
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L',
                                                                                                           out_channels=-1),
                                                           out_channels=-1, FPGA=False)
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C',
                                                                                                     out_channels=out_channels),
                                                       out_channels=out_channels, FPGA=False)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits,
                                                            range_tracker=AveragedRangeTracker(q_level='L',
                                                                                               out_channels=-1),
                                                            out_channels=-1, FPGA=False, sign=False)
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C',
                                                                                                      out_channels=out_channels),
                                                        out_channels=out_channels, FPGA=False, sign=False)

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


class BNFold_PTQuantizedConv2d_For_FPGA(PTQuantizedConv2d):
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
            q_type=0,
            bn=0,
            activate='leaky'
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
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.zeros(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))

        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L',
                                                                                                           out_channels=-1),
                                                           out_channels=-1, FPGA=True)
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits,
                                                       range_tracker=GlobalRangeTracker(q_level='L', out_channels=-1),
                                                       out_channels=-1, FPGA=True)
            self.bias_quantizer = SymmetricQuantizer(bits=w_bits,
                                                     range_tracker=GlobalRangeTracker(q_level='L', out_channels=-1),
                                                     out_channels=-1, FPGA=True)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits,
                                                            range_tracker=AveragedRangeTracker(q_level='L',
                                                                                               out_channels=-1),
                                                            out_channels=-1, FPGA=True, sign=False)
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits,
                                                        range_tracker=GlobalRangeTracker(q_level='L', out_channels=-1),
                                                        out_channels=-1, FPGA=True, sign=False)
            self.bias_quantizer = AsymmetricQuantizer(bits=w_bits,
                                                      range_tracker=GlobalRangeTracker(q_level='L', out_channels=-1),
                                                      out_channels=-1, FPGA=True, sign=False)

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