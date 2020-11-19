# Author:LiPu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# ********************* A(特征)量化 ***********************
class activation_quantize(nn.Module):
    def __init__(self, a_bits):
        super().__init__()
        self.a_bits = a_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)  # 特征A截断前先进行缩放（* 0.1），以减小截断误差
            scale = float(2 ** self.a_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
        return output


# ********************* W(模型参数)量化 ***********************
class weight_quantize(nn.Module):
    def __init__(self, w_bits):
        super().__init__()
        self.w_bits = w_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            scale = float(2 ** self.w_bits - 1)
            output = output * scale
            output = self.round(output)
            output = output / scale
            output = 2 * output - 1
        return output

    def get_weights(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            scale = float(2 ** self.w_bits - 1)
            output = output * scale
            output = self.round(output)
        return output


# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class DorefaConv2d(nn.Conv2d):
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
            first_layer=0
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
        # 实例化调用A和W量化器
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)
        self.first_layer = first_layer

    def forward(self, input):
        # 量化A和W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(self.weight)
        # 量化卷积
        output = F.conv2d(
            input=q_input,
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


class BNFold_DorefaConv2d(DorefaConv2d):

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
            bn=0
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
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.zeros(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))

        init.normal_(self.gamma, 1, 0.5)
        init.zeros_(self.beta)

        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)

    def forward(self, input):
        # 训练态
        if self.training:
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
                        self.running_mean.mul_(1 - self.momentum).add_(self.momentum * self.batch_mean)
                        self.running_var.mul_(1 - self.momentum).add_(self.momentum * self.batch_var)
                # BN融合
                if self.bias is not None:
                    bias = reshape_to_bias(
                        self.beta + (self.bias - self.batch_mean) * (
                                self.gamma / torch.sqrt(self.batch_var + self.eps)))
                else:
                    bias = reshape_to_bias(
                        self.beta - self.batch_mean * (self.gamma / torch.sqrt(self.batch_var + self.eps)))  # b融batch
                weight = self.weight * reshape_to_weight(
                    self.gamma / torch.sqrt(self.batch_var + self.eps))  # w融running
                # if self.bias is not None:
                #     bias = reshape_to_bias(
                #         self.beta + (self.bias - self.running_mean) * (
                #                     self.gamma / torch.sqrt(self.running_var + self.eps)))
                # else:
                #     bias = reshape_to_bias(
                #         self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融batch
                # weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running

            else:
                bias = self.bias
                weight = self.weight
        # 测试态
        else:
            # print(self.running_mean, self.running_var)
            # BN融合
            if self.bn:
                if self.bias is not None:
                    bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                            self.gamma / torch.sqrt(self.running_var + self.eps)))
                else:
                    bias = reshape_to_bias(
                        self.beta - self.running_mean * (
                                self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
                weight = self.weight * reshape_to_weight(
                    self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
            else:
                bias = self.bias
                weight = self.weight
        # 量化A和bn融合后的W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(weight)
        # 量化卷积
        if self.training:  # 训练态
            output = F.conv2d(
                input=q_input,
                weight=q_weight,
                # bias=self.bias,  # 注意，这里不加bias（self.bias为None）
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            # # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
            # output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
            # output += reshape_to_activation(
            #     self.gamma * (self.running_mean / (self.running_var + self.eps) - batch_mean / (batch_var + self.eps)))
            # output += reshape_to_activation(bias)
        else:  # 测试态
            output = F.conv2d(
                input=q_input,
                weight=q_weight,
                bias=bias,  # 注意，这里加bias，做完整的conv+bn
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        return output


class DorefaLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)

    def forward(self, input):
        # 量化A和W
        q_input = self.activation_quantizer(input)
        q_weight = self.weight_quantizer(self.weight)
        # 量化全连接
        output = F.linear(input=q_input, weight=q_weight, bias=self.bias)
        return output
