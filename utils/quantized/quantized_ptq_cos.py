# Author:LiPu
import math
import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
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
            momentum=0.1,
            a_bits=8,
            w_bits=8,
            bn=0,
            activate='leaky',
            quantizer_output=False,
            reorder=False, TM=32, TN=32,
            name='', layer_idx=-1, maxabsscaler=False
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
        if bias == False:
            self.bias = Parameter(torch.zeros(out_channels))
        self.activate = activate
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('q_bias', torch.zeros(out_channels))
        self.register_buffer('q_weight', torch.zeros(self.weight.shape))
        self.efficency = 0
        self.deviation = 0
        self.stop = False
        self.quantized = False
        self.quantizer_output = quantizer_output
        self.reorder = reorder
        self.TM = TM
        self.TN = TN
        self.name = name
        self.layer_idx = layer_idx
        self.maxabsscaler = maxabsscaler
        self.a_bits = a_bits
        self.w_bits = w_bits
        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = Quantizer(bits=a_bits, out_channels=-1)
        self.weight_quantizer = Quantizer(bits=w_bits, out_channels=-1)
        self.bias_quantizer = Quantizer(bits=w_bits, out_channels=-1)

    def forward(self, input):
        if not self.quantized:
            if self.bn:
                # BN融合
                if self.bias is not None:
                    self.bias.data = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                            self.gamma / torch.sqrt(self.running_var + self.eps)))
                else:
                    self.bias.data = reshape_to_bias(
                        self.beta - self.running_mean * self.gamma / torch.sqrt(
                            self.running_var + self.eps))  # b融running
                self.weight.data = self.weight * reshape_to_weight(
                    self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
            else:
                self.bias = self.bias
                self.weight = self.weight
            # 量化A和bn融合后的W
            self.q_weight = self.weight_quantizer(self.weight)
            self.q_bias = self.bias_quantizer(self.bias)
            self.quantized = True
        if self.training:
            if isinstance(input, list):
                quant_input = input[0]
                float_input = input[1]
            else:
                quant_input = input
                float_input = input

            # 浮点卷积
            float_output = F.conv2d(
                input=float_input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )

            # 计算bias_correct
            if not self.stop:
                # 量化卷积
                output = F.conv2d(
                    input=quant_input,
                    weight=self.q_weight,
                    bias=self.q_bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )

                # 补偿卷积
                correct_output = F.conv2d(
                    input=quant_input,
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                rate = 0.05
                error = torch.add(output, correct_output, alpha=-1).data
                noise = error.pow(2).mean()
                if noise > 0:
                    eff = 1.25 * correct_output.pow(2).mean().div(noise).log10().detach().cpu().numpy()
                    dev = math.fabs(eff - self.efficency)
                    if dev > 0:
                        self.efficency = (self.efficency * 4 + eff) * 0.2
                        self.deviation = (self.deviation * 4 + dev) * 0.2
                        if self.efficency > 4.0:
                            rate = rate * 0.5
                        if self.efficency > 4.3 or (self.deviation / self.efficency) < 0.05 or math.fabs(
                                dev - self.deviation / dev) < 0.05:
                            self.stop = True
                    else:
                        self.stop = True
                else:
                    self.stop = True
                if not self.stop:
                    error = error.mean(dim=[0, 2, 3])
                    self.bias.data = torch.sub(self.bias.data, error, alpha=rate)
                    self.q_bias = self.bias_quantizer(self.bias)
                torch.cuda.empty_cache()
            output = F.conv2d(
                input=quant_input,
                weight=self.q_weight,
                bias=self.q_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        else:
            output = F.conv2d(
                input=input,
                weight=self.q_weight,
                bias=self.q_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
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

            if not os.path.isdir('./quantizer_output/q_weight_reorder'):
                os.makedirs('./quantizer_output/q_weight_reorder')
            if not os.path.isdir('./quantizer_output/q_bias_reorder'):
                os.makedirs('./quantizer_output/q_bias_reorder')

            if self.layer_idx == -1:

                #######################输出当前层的权重量化因子
                weight_scale = - self.weight_quantizer.get_scale()
                np.savetxt(('./quantizer_output/w_scale_out/w_scale_%s.txt' % self.name), weight_scale, delimiter='\n')
                #######################输出当前层的量化权重
                q_weight_txt = self.weight_quantizer.get_quantize_value(weight)

                #############权重重排序

                w_para = q_weight_txt  # 重排序参数
                if self.reorder == True:
                    # print("use weights reorder!")
                    shape_output = w_para.shape[0]
                    shape_input = w_para.shape[1]
                    num_TN = int(shape_input / self.TN)
                    remainder_TN = shape_input % self.TN
                    num_TM = int(shape_output / self.TM)
                    remainder_TM = shape_output % self.TM
                    first = True
                    reorder_w_para = None
                    if self.activate == 'linear':
                        print('layer-linear reorder!')
                        for k in range(num_TN):
                            temp = w_para[0:remainder_TM, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                            temp = temp.permute(2, 0, 1).contiguous().view(-1)
                            if first:
                                reorder_w_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())
                    else:
                        for j in range(num_TM):
                            if shape_input == 3 or shape_input == 1:  # 第一层
                                print('The first layer~~~~~~~~~~~~')
                                temp = w_para[j * self.TM:(j + 1) * self.TM,
                                       num_TN * self.TN:num_TN * self.TN + remainder_TN, :,
                                       :]
                                temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                                fill = torch.zeros(self.TM, self.TN, temp.shape[2]).to(temp.device)
                                fill[:, 0:remainder_TN, :] = temp
                                temp = fill.permute(2, 0, 1).contiguous().view(-1)
                                if first:  # 创建数组存储
                                    reorder_w_para = temp.clone().cpu().data.numpy()
                                    first = False
                                else:
                                    reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())
                            else:
                                for k in range(num_TN):
                                    temp = w_para[j * self.TM:(j + 1) * self.TM, k * self.TN:(k + 1) * self.TN, :, :]
                                    # #合并成论文图10(a)的TM*TN*(K2)的张量格式
                                    temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                                    # 转换为图10(b)的重排序格式
                                    temp = temp.permute(2, 0, 1).contiguous().view(-1)
                                    if first:
                                        reorder_w_para = temp.clone().cpu().data.numpy()
                                        first = False
                                    else:
                                        reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())

                    w_para_flatten = reorder_w_para
                    # print(reorder_w_para.size)
                    #####验证重排序结果的正确性
                    '''if w_para_flatten.size == w_para.shape[0] * w_para.shape[1] * w_para.shape[2] * w_para.shape[3]:
                        print("weights convert correctly!")
                    else:
                        print("weights convert mismatchingly!")'''

                    q_weight_reorder = w_para_flatten
                    q_weight_reorder = np.array(q_weight_reorder).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_weight_reorder/w_reorder_%s.txt' % self.name), q_weight_reorder,
                               delimiter='\n')
                ################权重重排序结束

                q_weight_txt = np.array(q_weight_txt.cpu()).reshape(1, -1)
                q_weight_max = [np.max(q_weight_txt)]
                # q_weight_max = np.argmax(q_weight_txt)
                max_weight_count = [np.sum(abs(q_weight_txt) >= (1 << (self.w_bits - 1)) - 1)]  # 统计该层溢出的数目
                np.savetxt(('./quantizer_output/max_weight_count/max_w_count_%s.txt' % self.name), max_weight_count)
                np.savetxt(('./quantizer_output/q_weight_max/max_w_%s.txt' % self.name), q_weight_max)
                np.savetxt(('./quantizer_output/q_weight_out/q_weight_%s.txt' % self.name), q_weight_txt,
                           delimiter='\n')
                # io.savemat('save.mat',{'q_weight_txt':q_weight_txt})

                #######################创建输出偏置txt的文件夹
                if not os.path.isdir('./quantizer_output/q_bias_out'):
                    os.makedirs('./quantizer_output/q_bias_out')
                if not os.path.isdir('./quantizer_output/b_scale_out'):
                    os.makedirs('./quantizer_output/b_scale_out')
                #######################输出当前层偏置的量化因子
                bias_scale = - self.bias_quantizer.get_scale()
                np.savetxt(('./quantizer_output/b_scale_out/b_scale_%s.txt' % self.name), bias_scale, delimiter='\n')
                #######################输出当前层的量化偏置
                q_bias_txt = self.bias_quantizer.get_quantize_value(bias)
                q_bias_txt = np.array(q_bias_txt.cpu()).reshape(1, -1)
                np.savetxt(('./quantizer_output/q_bias_out/q_bias_%s.txt' % self.name), q_bias_txt, delimiter='\n')

                #############偏置重排序
                if self.reorder == True:
                    b_para = np.zeros(2048, dtype=int)
                    b_para[0:q_bias_txt.size] = q_bias_txt
                    # print(b_para.shape)
                    # b_para = np.array(b_para.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_bias_reorder/q_b_reorder_%s.txt' % self.name), b_para,
                               delimiter='\n')
                    ######权重和偏置的重排序数据的二进制文件保存
                    bias_weight_reorder = np.append(b_para, q_weight_reorder)
                    wb_flat = bias_weight_reorder.astype(np.int8)
                    writer = open('./quantizer_output/q_weight_reorder/%s_bias_weight_q_bin' % self.name, "wb")
                    writer.write(wb_flat)
                    writer.close()
                ################偏置重排序结束
            elif int(self.name[1:4]) == self.layer_idx:
                #######################输出当前层的权重量化因子
                weight_scale = - self.weight_quantizer.get_scale()
                np.savetxt(('./quantizer_output/w_scale_out/w_scale_%s.txt' % self.name), weight_scale, delimiter='\n')
                #######################输出当前层的量化权重
                q_weight_txt = self.weight_quantizer.get_quantize_value(weight)

                #############权重重排序

                w_para = q_weight_txt  # 重排序参数
                if self.reorder == True:
                    # print("use weights reorder!")
                    shape_output = w_para.shape[0]
                    shape_input = w_para.shape[1]
                    num_TN = int(shape_input / self.TN)
                    remainder_TN = shape_input % self.TN
                    num_TM = int(shape_output / self.TM)
                    remainder_TM = shape_output % self.TM
                    first = True
                    reorder_w_para = None
                    if self.activate == 'linear':
                        print('layer-linear reorder!')
                        for k in range(num_TN):
                            temp = w_para[0:remainder_TM, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                            temp = temp.permute(2, 0, 1).contiguous().view(-1)
                            if first:
                                reorder_w_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())
                    else:
                        for j in range(num_TM):
                            if shape_input == 3 or shape_input == 1:  # 第一层
                                print('The first layer~~~~~~~~~~~~')
                                temp = w_para[j * self.TM:(j + 1) * self.TM,
                                       num_TN * self.TN:num_TN * self.TN + remainder_TN, :,
                                       :]
                                temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                                fill = torch.zeros(self.TM, self.TN, temp.shape[2]).to(temp.device)
                                fill[:, 0:remainder_TN, :] = temp
                                temp = fill.permute(2, 0, 1).contiguous().view(-1)
                                if first:  # 创建数组存储
                                    reorder_w_para = temp.clone().cpu().data.numpy()
                                    first = False
                                else:
                                    reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())
                            else:
                                for k in range(num_TN):
                                    temp = w_para[j * self.TM:(j + 1) * self.TM, k * self.TN:(k + 1) * self.TN, :, :]
                                    # #合并成论文图10(a)的TM*TN*(K2)的张量格式
                                    temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                                    # 转换为图10(b)的重排序格式
                                    temp = temp.permute(2, 0, 1).contiguous().view(-1)
                                    if first:
                                        reorder_w_para = temp.clone().cpu().data.numpy()
                                        first = False
                                    else:
                                        reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())

                    w_para_flatten = reorder_w_para
                    # print(reorder_w_para.size)
                    #####验证重排序结果的正确性
                    '''if w_para_flatten.size == w_para.shape[0] * w_para.shape[1] * w_para.shape[2] * w_para.shape[3]:
                        print("weights convert correctly!")
                    else:
                        print("weights convert mismatchingly!")'''

                    q_weight_reorder = w_para_flatten
                    q_weight_reorder = np.array(q_weight_reorder).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_weight_reorder/w_reorder_%s.txt' % self.name), q_weight_reorder,
                               delimiter='\n')
                ################权重重排序结束

                q_weight_txt = np.array(q_weight_txt.cpu()).reshape(1, -1)
                q_weight_max = [np.max(q_weight_txt)]
                # q_weight_max = np.argmax(q_weight_txt)
                max_weight_count = [np.sum(abs(q_weight_txt) >= (1 << (self.w_bits - 1)) - 1)]  # 统计该层溢出的数目
                np.savetxt(('./quantizer_output/max_weight_count/max_w_count_%s.txt' % self.name), max_weight_count)
                np.savetxt(('./quantizer_output/q_weight_max/max_w_%s.txt' % self.name), q_weight_max)
                np.savetxt(('./quantizer_output/q_weight_out/q_weight_%s.txt' % self.name), q_weight_txt,
                           delimiter='\n')
                # io.savemat('save.mat',{'q_weight_txt':q_weight_txt})

                #######################创建输出偏置txt的文件夹
                if not os.path.isdir('./quantizer_output/q_bias_out'):
                    os.makedirs('./quantizer_output/q_bias_out')
                if not os.path.isdir('./quantizer_output/b_scale_out'):
                    os.makedirs('./quantizer_output/b_scale_out')
                #######################输出当前层偏置的量化因子
                bias_scale = - self.bias_quantizer.get_scale()
                np.savetxt(('./quantizer_output/b_scale_out/b_scale_%s.txt' % self.name), bias_scale, delimiter='\n')
                #######################输出当前层的量化偏置
                q_bias_txt = self.bias_quantizer.get_quantize_value(bias)
                q_bias_txt = np.array(q_bias_txt.cpu()).reshape(1, -1)
                np.savetxt(('./quantizer_output/q_bias_out/q_bias_%s.txt' % self.name), q_bias_txt, delimiter='\n')

                #############偏置重排序
                if self.reorder == True:
                    b_para = np.zeros(2048, dtype=int)
                    b_para[0:q_bias_txt.size] = q_bias_txt
                    # print(b_para.shape)
                    # b_para = np.array(b_para.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_bias_reorder/q_b_reorder_%s.txt' % self.name), b_para,
                               delimiter='\n')
                    ######权重和偏置的重排序数据的二进制文件保存
                    bias_weight_reorder = np.append(b_para, q_weight_reorder)
                    wb_flat = bias_weight_reorder.astype(np.int8)
                    writer = open('./quantizer_output/q_weight_reorder/%s_bias_weight_q_bin' % self.name, "wb")
                    writer.write(wb_flat)
                    writer.close()
                ################偏置重排序结束

        if self.activate == 'leaky':
            output = F.leaky_relu(output, 0.125 if not self.maxabsscaler else 0.25, inplace=True)
            if self.training:
                float_output = F.leaky_relu(float_output, 0.125 if not self.maxabsscaler else 0.25, inplace=True)
        elif self.activate == 'relu6':
            output = F.relu6(output, inplace=True)
            if self.training:
                float_output = F.relu6(float_output, inplace=True)
        elif self.activate == 'h_swish':
            output = output * (F.relu6(output + 3.0, inplace=True) / 6.0)
            if self.training:
                float_output = output * (F.relu6(float_output + 3.0, inplace=True) / 6.0)
        elif self.activate == 'relu':
            output = F.relu(output, inplace=True)
            if self.training:
                float_output = F.relu(float_output, inplace=True)
        elif self.activate == 'mish':
            output = output * F.softplus(output).tanh()
            if self.training:
                float_output = output * F.softplus(float_output).tanh()
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
            if not os.path.isdir('./quantizer_output/q_activation_reorder'):
                os.makedirs('./quantizer_output/q_activation_reorder')

            if self.layer_idx == -1:
                ##################输出当前激活的量化因子
                activation_scale = - self.activation_quantizer.get_scale()
                np.savetxt(('./quantizer_output/a_scale_out/a_scale_%s.txt' % self.name), activation_scale,
                           delimiter='\n')
                ##################输出当前层的量化激活
                q_activation_txt = self.activation_quantizer.get_quantize_value(output)

                a_para = q_activation_txt
                #############输入特征图重排序
                if self.reorder == True:
                    # 重排序参数
                    # print("use activation reorder!")
                    shape_input = a_para.shape[1]
                    num_TN = int(shape_input / self.TN)
                    remainder_TN = shape_input % self.TN
                    first = True
                    reorder_a_para = None
                    if self.activate == 'linear':
                        print('layer-linear reorder!')
                        temp = a_para[:, 0:remainder_TN, :, :]
                        temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                        temp = temp.permute(1, 2, 0).contiguous().view(-1)
                        if first:
                            reorder_a_para = temp.clone().cpu().data.numpy()
                            first = False
                        else:
                            reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())
                    else:
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                    a_para_flatten = reorder_a_para
                    #####验证重排序结果的正确性
                    '''if a_para_flatten.size == a_para.shape[0] * a_para.shape[1] * a_para.shape[2] * a_para.shape[3]:
                        print("activation convert correctly!")
                    else:
                        print("activation convert mismatchingly!")'''

                    q_activation_reorder = a_para_flatten
                    q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_reorder/a_reorder_%s.txt' % self.name),
                               q_activation_reorder, delimiter='\n')
                    ###保存重排序的二进制文件
                    activation_flat = q_activation_reorder.astype(np.int8)
                    writer = open('./quantizer_output/q_activation_reorder/%s_activation_q_bin' % self.name, "wb")
                    writer.write(activation_flat)
                    writer.close()
                ##########特征图重排序结束

                q_activation_txt = np.array(q_activation_txt.cpu()).reshape(1, -1)
                q_activation_max = [np.max(q_activation_txt)]  # 统计该层的最大值(即查看是否有溢出)
                max_activation_count = [np.sum(abs(q_activation_txt) >= (1 << (self.w_bits - 1)) - 1)]  # 统计该层溢出的数目
                # q_weight_max = np.argmax(q_weight_txt)
                np.savetxt(('./quantizer_output/max_activation_count/max_a_count_%s.txt' % self.name),
                           max_activation_count)
                np.savetxt(('./quantizer_output/q_activation_max/q_a_max_%s.txt' % self.name), q_activation_max)
                np.savetxt(('./quantizer_output/q_activation_out/q_activation_%s.txt' % self.name), q_activation_txt,
                           delimiter='\n')

            elif int(self.name[1:4]) == self.layer_idx:

                ##################输出当前激活的量化因子
                activation_scale = - self.activation_quantizer.get_scale()
                np.savetxt(('./quantizer_output/a_scale_out/a_scale_%s.txt' % self.name), activation_scale,
                           delimiter='\n')
                ##################输出当前层的量化激活
                q_activation_txt = self.activation_quantizer.get_quantize_value(output)

                a_para = q_activation_txt
                #############输入特征图重排序
                if self.reorder == True:
                    # 重排序参数
                    # print("use activation reorder!")
                    shape_input = a_para.shape[1]
                    num_TN = int(shape_input / self.TN)
                    remainder_TN = shape_input % self.TN
                    first = True
                    reorder_a_para = None
                    if self.activate == 'linear':
                        print('layer-linear reorder!')
                        temp = a_para[:, 0:remainder_TN, :, :]
                        temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                        temp = temp.permute(1, 2, 0).contiguous().view(-1)
                        if first:
                            reorder_a_para = temp.clone().cpu().data.numpy()
                            first = False
                        else:
                            reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())
                    else:
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                    a_para_flatten = reorder_a_para
                    #####验证重排序结果的正确性
                    '''if a_para_flatten.size == a_para.shape[0] * a_para.shape[1] * a_para.shape[2] * a_para.shape[3]:
                        print("activation convert correctly!")
                    else:
                        print("activation convert mismatchingly!")'''

                    q_activation_reorder = a_para_flatten
                    q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_reorder/a_reorder_%s.txt' % self.name),
                               q_activation_reorder, delimiter='\n')
                    ###保存重排序的二进制文件
                    activation_flat = q_activation_reorder.astype(np.int8)
                    writer = open('./quantizer_output/q_activation_reorder/%s_activation_q_bin' % self.name, "wb")
                    writer.write(activation_flat)
                    writer.close()
                ##########特征图重排序结束

                q_activation_txt = np.array(q_activation_txt.cpu()).reshape(1, -1)
                q_activation_max = [np.max(q_activation_txt)]  # 统计该层的最大值(即查看是否有溢出)
                max_activation_count = [np.sum(abs(q_activation_txt) >= (1 << (self.w_bits - 1)) - 1)]  # 统计该层溢出的数目
                # q_weight_max = np.argmax(q_weight_txt)
                np.savetxt(('./quantizer_output/max_activation_count/max_a_count_%s.txt' % self.name),
                           max_activation_count)
                np.savetxt(('./quantizer_output/q_activation_max/q_a_max_%s.txt' % self.name), q_activation_max)
                np.savetxt(('./quantizer_output/q_activation_out/q_activation_%s.txt' % self.name), q_activation_txt,
                           delimiter='\n')

        output = self.activation_quantizer(output)
        if self.training and self.activate != 'linear':
            return [output, float_output]
        else:
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


class COSPTQuantizedShortcut_min(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False, bits=8, FPGA=False,
                 quantizer_output=False, reorder=False, TM=32, TN=32, name='', layer_idx=-1, ):
        super(COSPTQuantizedShortcut_min, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        self.bits = bits
        self.FPGA = FPGA

        self.register_buffer('scale_x', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_range_x', torch.zeros(1))
        self.scale_list_x = [0 for i in range(bits)]

        self.register_buffer('scale_a', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_range_a', torch.zeros(1))
        self.scale_list_a = [0 for i in range(bits)]

        self.register_buffer('scale_sum', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_range_sum', torch.zeros(1))
        self.scale_list_sum = [0 for i in range(bits)]

        self.quantizer_output = quantizer_output
        self.reorder = reorder
        self.TM = TM
        self.TN = TN
        self.name = name
        self.layer_idx = layer_idx

        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    # 量化
    def quantize(self, input, type):
        if type == "a":
            output = input / self.scale_a
        elif type == "x":
            output = input / self.scale_x
        elif type == "sum":
            output = input / self.scale_sum
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
    def dequantize(self, input, type):
        if type == "a":
            output = (input) * self.scale_a
        elif type == "x":
            output = (input) * self.scale_x
        elif type == "sum":
            output = (input) * self.scale_sum
        return output

    # 更新参数
    def update_params(self, step, type):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val))  # 量化后范围
        if type == "a":
            temp = self.float_range_a
            self.float_range_a.add_(-temp).add_(2 ** step)
            self.scale_a = self.float_range_a / quantized_range  # 量化比例因子
        elif type == "x":
            temp = self.float_range_x
            self.float_range_x.add_(-temp).add_(2 ** step)
            self.scale_x = self.float_range_x / quantized_range  # 量化比例因子
        elif type == "sum":
            temp = self.float_range_sum
            self.float_range_sum.add_(-temp).add_(2 ** step)
            self.scale_sum = self.float_range_sum / quantized_range  # 量化比例因子

    def forward(self, x, outputs):
        if self.training:
            float = x[1]
            x = x[0]
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]
        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            if self.training:
                a = outputs[self.layers[i]][0] * w[i + 1] if self.weight else outputs[self.layers[i]][
                    0]  # feature to add
            else:
                a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels
            if self.training == True:
                # 得到输入两个feature和输出的scale
                max_metrics = -1
                max_step = 0
                for i in range(self.bits):
                    self.update_params(i, type="a")
                    output = self.quantize(a, type="a")  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output, type="a")  # 反量化
                    cosine_similarity = torch.cosine_similarity(a.view(-1), output.view(-1), dim=0)
                    if cosine_similarity > max_metrics:
                        max_metrics = cosine_similarity
                        max_step = i
                self.scale_list_a[max_step] += 1
                Global_max_step = self.scale_list_a.index(max(self.scale_list_a))
                self.update_params(Global_max_step, type="a")

                max_metrics = -1
                max_step = 0
                for i in range(self.bits):
                    self.update_params(i, type="x")
                    output = self.quantize(x, type="x")  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output, type="x")  # 反量化
                    cosine_similarity = torch.cosine_similarity(x.view(-1), output.view(-1), dim=0)
                    if cosine_similarity > max_metrics:
                        max_metrics = cosine_similarity
                        max_step = i
                self.scale_list_x[max_step] += 1
                Global_max_step = self.scale_list_x.index(max(self.scale_list_x))
                self.update_params(Global_max_step, type="x")

                float_max_val = min(self.float_range_a, self.float_range_x)
                self.update_params(float_max_val.log2(), type="a")
                self.update_params(float_max_val.log2(), type="x")

            # 量化x
            x = self.quantize(x, type="x")  # 量化
            x = self.round(x)
            x = self.dequantize(x, type="x")  # 反量化

            # 量化a
            a = self.quantize(a, type="a")  # 量化
            a = self.round(a)
            a = self.dequantize(a, type="a")  # 反量化

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]
            # 量化和
            if self.training == True:
                max_metrics = -1
                max_step = 0
                for i in range(self.bits):
                    self.update_params(i, type="sum")
                    output = self.quantize(x, type="sum")  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output, type="sum")  # 反量化
                    cosine_similarity = torch.cosine_similarity(x.view(-1), output.view(-1), dim=0)
                    if cosine_similarity > max_metrics:
                        max_metrics = cosine_similarity
                        max_step = i
                self.scale_list_sum[max_step] += 1
                Global_max_step = self.scale_list_sum.index(max(self.scale_list_sum))
                self.update_params(Global_max_step, type="sum")
            x = self.quantize(x, type="sum")  # 量化
            x = self.round(x)
            x = self.clamp(x)  # 截断
            # 量化因子数据输出
            if self.quantizer_output == True:
                if not os.path.isdir('./quantizer_output/q_activation_out'):
                    os.makedirs('./quantizer_output/q_activation_out')
                if not os.path.isdir('./quantizer_output/a_scale_out'):
                    os.makedirs('./quantizer_output/a_scale_out')
                if not os.path.isdir('./quantizer_output/q_activation_max'):
                    os.makedirs('./quantizer_output/q_activation_max')
                if not os.path.isdir('./quantizer_output/max_activation_count'):
                    os.makedirs('./quantizer_output/max_activation_count')
                if not os.path.isdir('./quantizer_output/q_activation_reorder'):
                    os.makedirs('./quantizer_output/q_activation_reorder')

                if self.layer_idx == -1:

                    move_scale = math.log2(self.scale_sum)
                    shortcut_scale = - np.array(move_scale).reshape(1, -1)
                    np.savetxt(('./quantizer_output/a_scale_out/shortcut_scale_%s.txt' % self.name), shortcut_scale,
                               delimiter='\n')

                elif int(self.name[1:4]) == self.layer_idx:

                    move_scale = math.log2(self.scale_sum)
                    shortcut_scale = - np.array(move_scale).reshape(1, -1)
                    np.savetxt(('./quantizer_output/a_scale_out/shortcut_scale_%s.txt' % self.name), shortcut_scale,
                               delimiter='\n')
            # 特征图量化数据输出
            if self.quantizer_output == True:
                if not os.path.isdir('./quantizer_output/q_activation_out'):
                    os.makedirs('./quantizer_output/q_activation_out')
                if not os.path.isdir('./quantizer_output/a_scale_out'):
                    os.makedirs('./quantizer_output/a_scale_out')
                if not os.path.isdir('./quantizer_output/q_activation_max'):
                    os.makedirs('./quantizer_output/q_activation_max')
                if not os.path.isdir('./quantizer_output/max_activation_count'):
                    os.makedirs('./quantizer_output/max_activation_count')
                if not os.path.isdir('./quantizer_output/q_activation_reorder'):
                    os.makedirs('./quantizer_output/q_activation_reorder')

                if self.layer_idx == -1:

                    q_x_shortcut = x

                    if self.reorder == True:
                        a_para = q_x_shortcut
                        # 重排序参数
                        # print("use activation reorder!")
                        shape_input = a_para.shape[1]
                        num_TN = int(shape_input / self.TN)
                        remainder_TN = shape_input % self.TN
                        first = True
                        reorder_a_para = None
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                        a_para_flatten = reorder_a_para
                        q_activation_reorder = a_para_flatten
                        q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                        np.savetxt(('./quantizer_output/q_activation_reorder/r_shortcut_%s.txt' % self.name),
                                   q_activation_reorder, delimiter='\n')
                        ###保存重排序的二进制文件
                        activation_flat = q_activation_reorder.astype(np.int8)
                        writer = open('./quantizer_output/q_activation_reorder/%s_shortcut_q_bin' % self.name, "wb")
                        writer.write(activation_flat)
                        writer.close()
                    ##########shortcut重排序结束

                    Q_shortcut = np.array(q_x_shortcut.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_out/Q_shortcut_%s.txt' % self.name), Q_shortcut,
                               delimiter='\n')

                elif int(self.name[1:4]) == self.layer_idx:

                    q_x_shortcut = x

                    if self.reorder == True:
                        a_para = q_x_shortcut
                        # 重排序参数
                        # print("use activation reorder!")
                        shape_input = a_para.shape[1]
                        num_TN = int(shape_input / self.TN)
                        remainder_TN = shape_input % self.TN
                        first = True
                        reorder_a_para = None
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                        a_para_flatten = reorder_a_para
                        q_activation_reorder = a_para_flatten
                        q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                        np.savetxt(('./quantizer_output/q_activation_reorder/r_shortcut_%s.txt' % self.name),
                                   q_activation_reorder, delimiter='\n')
                        ###保存重排序的二进制文件
                        activation_flat = q_activation_reorder.astype(np.int8)
                        writer = open('./quantizer_output/q_activation_reorder/%s_shortcut_q_bin' % self.name, "wb")
                        writer.write(activation_flat)
                        writer.close()
                    ##########shortcut重排序结束
                    Q_shortcut = np.array(q_x_shortcut.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_out/Q_shortcut_%s.txt' % self.name), Q_shortcut,
                               delimiter='\n')

            x = self.dequantize(x, type="sum")  # 反量化

        if self.training:
            # float compute
            # Weights
            if self.weight:
                w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
                float = float * w[0]

            # Fusion
            nx = float.shape[1]  # input channels
            for i in range(self.n - 1):
                a = outputs[self.layers[i]][1] * w[i + 1] if self.weight else outputs[self.layers[i]][
                    1]  # feature to add
                na = a.shape[1]  # feature channels

                # Adjust channels
                if nx == na:  # same shape
                    float = float + a
                elif nx > na:  # slice input
                    float[:, :na] = float[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
                else:  # slice feature
                    float = float + a[:, :nx]

            return [x, float]
        else:
            return x


class COSPTQuantizedShortcut_max(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False, bits=8, FPGA=False,
                 quantizer_output=False, reorder=False, TM=32, TN=32, name='', layer_idx=-1, ):
        super(COSPTQuantizedShortcut_max, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        self.bits = bits
        self.FPGA = FPGA

        self.register_buffer('scale_x', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_range_x', torch.zeros(1))

        self.register_buffer('scale_a', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_range_a', torch.zeros(1))

        self.register_buffer('scale_sum', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_range_sum', torch.zeros(1))
        self.scale_list = [0 for i in range(bits)]

        self.quantizer_output = quantizer_output
        self.reorder = reorder
        self.TM = TM
        self.TN = TN
        self.name = name
        self.layer_idx = layer_idx

        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    # 量化
    def quantize(self, input, type):
        if type == "a":
            output = input / self.scale_a
        elif type == "x":
            output = input / self.scale_x
        elif type == "sum":
            output = input / self.scale_sum
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
    def dequantize(self, input, type):
        if type == "a":
            output = (input) * self.scale_a
        elif type == "x":
            output = (input) * self.scale_x
        elif type == "sum":
            output = (input) * self.scale_sum
        return output

    # 更新参数
    def update_params(self, step, type):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val))  # 量化后范围
        if type == "a":
            temp = self.float_range_a
            self.float_range_a.add_(-temp).add_(2 ** step)
            self.scale_a = self.float_range_a / quantized_range  # 量化比例因子
        elif type == "x":
            temp = self.float_range_x
            self.float_range_x.add_(-temp).add_(2 ** step)
            self.scale_x = self.float_range_x / quantized_range  # 量化比例因子
        elif type == "sum":
            temp = self.float_range_sum
            self.float_range_sum.add_(-temp).add_(2 ** step)
            self.scale_sum = self.float_range_sum / quantized_range  # 量化比例因子

    def forward(self, x, outputs):
        if self.training:
            float = x[1]
            x = x[0]
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            if self.training:
                a = outputs[self.layers[i]][0] * w[i + 1] if self.weight else outputs[self.layers[i]][
                    0]  # feature to add
            else:
                a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels
            if self.training == True:
                # 得到输入两个feature和输出的scale
                max_metrics = -1
                max_step = 0
                for i in range(self.bits):
                    cosine_similarity = 0
                    self.update_params(i, type="a")
                    output = self.quantize(a, type="a")  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output, type="a")  # 反量化
                    cosine_similarity = cosine_similarity + torch.cosine_similarity(a.view(-1), output.view(-1), dim=0)

                    self.update_params(i, type="x")
                    output = self.quantize(x, type="x")  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output, type="x")  # 反量化
                    cosine_similarity = cosine_similarity + torch.cosine_similarity(x.view(-1), output.view(-1), dim=0)
                    # Adjust channels
                    if nx == na:  # same shape
                        temp_x = x + a
                    elif nx > na:  # slice input
                        temp_x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
                    else:  # slice feature
                        temp_x = x + a[:, :nx]

                    self.update_params(i, type="sum")
                    output = self.quantize(temp_x, type="sum")  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output, type="sum")  # 反量化
                    cosine_similarity = cosine_similarity + torch.cosine_similarity(temp_x.view(-1), output.view(-1),
                                                                                    dim=0)
                    del temp_x

                    if cosine_similarity > max_metrics:
                        max_metrics = cosine_similarity
                        max_step = i
                self.scale_list[max_step] += 1
                Global_max_step = self.scale_list.index(max(self.scale_list))
                self.update_params(Global_max_step, type="x")
                self.update_params(Global_max_step, type="a")
                self.update_params(Global_max_step, type="sum")

            # 量化x
            x = self.quantize(x, type="x")  # 量化
            x = self.round(x)
            x = self.dequantize(x, type="x")  # 反量化

            # 量化a
            a = self.quantize(a, type="a")  # 量化
            a = self.round(a)
            a = self.dequantize(a, type="a")  # 反量化

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]
            # 量化和
            x = self.quantize(x, type="sum")  # 量化
            x = self.round(x)
            x = self.clamp(x)  # 截断
            # 量化因子数据输出
            if self.quantizer_output == True:
                if not os.path.isdir('./quantizer_output/q_activation_out'):
                    os.makedirs('./quantizer_output/q_activation_out')
                if not os.path.isdir('./quantizer_output/a_scale_out'):
                    os.makedirs('./quantizer_output/a_scale_out')
                if not os.path.isdir('./quantizer_output/q_activation_max'):
                    os.makedirs('./quantizer_output/q_activation_max')
                if not os.path.isdir('./quantizer_output/max_activation_count'):
                    os.makedirs('./quantizer_output/max_activation_count')
                if not os.path.isdir('./quantizer_output/q_activation_reorder'):
                    os.makedirs('./quantizer_output/q_activation_reorder')

                if self.layer_idx == -1:

                    move_scale = math.log2(self.scale_sum)
                    shortcut_scale = - np.array(move_scale).reshape(1, -1)
                    np.savetxt(('./quantizer_output/a_scale_out/shortcut_scale_%s.txt' % self.name), shortcut_scale,
                               delimiter='\n')

                elif int(self.name[1:4]) == self.layer_idx:

                    move_scale = math.log2(self.scale_sum)
                    shortcut_scale = - np.array(move_scale).reshape(1, -1)
                    np.savetxt(('./quantizer_output/a_scale_out/shortcut_scale_%s.txt' % self.name), shortcut_scale,
                               delimiter='\n')
            # 特征图量化数据输出
            if self.quantizer_output == True:
                if not os.path.isdir('./quantizer_output/q_activation_out'):
                    os.makedirs('./quantizer_output/q_activation_out')
                if not os.path.isdir('./quantizer_output/a_scale_out'):
                    os.makedirs('./quantizer_output/a_scale_out')
                if not os.path.isdir('./quantizer_output/q_activation_max'):
                    os.makedirs('./quantizer_output/q_activation_max')
                if not os.path.isdir('./quantizer_output/max_activation_count'):
                    os.makedirs('./quantizer_output/max_activation_count')
                if not os.path.isdir('./quantizer_output/q_activation_reorder'):
                    os.makedirs('./quantizer_output/q_activation_reorder')

                if self.layer_idx == -1:

                    q_x_shortcut = x

                    if self.reorder == True:
                        a_para = q_x_shortcut
                        # 重排序参数
                        # print("use activation reorder!")
                        shape_input = a_para.shape[1]
                        num_TN = int(shape_input / self.TN)
                        remainder_TN = shape_input % self.TN
                        first = True
                        reorder_a_para = None
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                        a_para_flatten = reorder_a_para
                        q_activation_reorder = a_para_flatten
                        q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                        np.savetxt(('./quantizer_output/q_activation_reorder/r_shortcut_%s.txt' % self.name),
                                   q_activation_reorder, delimiter='\n')
                        ###保存重排序的二进制文件
                        activation_flat = q_activation_reorder.astype(np.int8)
                        writer = open('./quantizer_output/q_activation_reorder/%s_shortcut_q_bin' % self.name, "wb")
                        writer.write(activation_flat)
                        writer.close()
                    ##########shortcut重排序结束

                    Q_shortcut = np.array(q_x_shortcut.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_out/Q_shortcut_%s.txt' % self.name), Q_shortcut,
                               delimiter='\n')

                elif int(self.name[1:4]) == self.layer_idx:

                    q_x_shortcut = x

                    if self.reorder == True:
                        a_para = q_x_shortcut
                        # 重排序参数
                        # print("use activation reorder!")
                        shape_input = a_para.shape[1]
                        num_TN = int(shape_input / self.TN)
                        remainder_TN = shape_input % self.TN
                        first = True
                        reorder_a_para = None
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                        a_para_flatten = reorder_a_para
                        q_activation_reorder = a_para_flatten
                        q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                        np.savetxt(('./quantizer_output/q_activation_reorder/r_shortcut_%s.txt' % self.name),
                                   q_activation_reorder, delimiter='\n')
                        ###保存重排序的二进制文件
                        activation_flat = q_activation_reorder.astype(np.int8)
                        writer = open('./quantizer_output/q_activation_reorder/%s_shortcut_q_bin' % self.name, "wb")
                        writer.write(activation_flat)
                        writer.close()
                    ##########shortcut重排序结束
                    Q_shortcut = np.array(q_x_shortcut.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_out/Q_shortcut_%s.txt' % self.name), Q_shortcut,
                               delimiter='\n')

            x = self.dequantize(x, type="sum")  # 反量化
        if self.training:
            # float compute
            # Weights
            if self.weight:
                w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
                float = float * w[0]

            # Fusion
            nx = float.shape[1]  # input channels
            for i in range(self.n - 1):
                a = outputs[self.layers[i]][1] * w[i + 1] if self.weight else outputs[self.layers[i]][
                    1]  # feature to add
                na = a.shape[1]  # feature channels

                # Adjust channels
                if nx == na:  # same shape
                    float = float + a
                elif nx > na:  # slice input
                    float[:, :na] = float[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
                else:  # slice feature
                    float = float + a[:, :nx]

            return [x, float]
        else:
            return x


class COSPTQuantizedFeatureConcat(nn.Module):
    def __init__(self, layers, groups, bits=8, FPGA=False,
                 quantizer_output=False, reorder=False, TM=32, TN=32, name='', layer_idx=-1, ):
        super(COSPTQuantizedFeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.groups = groups
        self.multiple = len(layers) > 1  # multiple layers flag
        self.register_buffer('scale', torch.zeros(1))  # 量化比例因子
        self.register_buffer('float_max_list', torch.zeros(len(layers)))
        self.bits = bits
        self.FPGA = FPGA
        self.momentum = 0.1
        self.quantizer_output = quantizer_output
        self.reorder = reorder
        self.TM = TM
        self.TN = TN
        self.name = name
        self.layer_idx = layer_idx
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

    def forward(self, x, outputs):
        if self.training:
            float = x[1]
            x = x[0]
        if self.multiple:
            if self.training == True:
                quantized_min_val = torch.tensor(-(1 << (self.bits - 1)))
                quantized_max_val = torch.tensor((1 << (self.bits - 1)) - 1)
                quantized_range = torch.max(torch.abs(quantized_min_val), torch.abs(quantized_max_val))  # 量化后范围
                j = 0
                for i in self.layers:
                    temp = outputs[i][0].detach()
                    if self.float_max_list[j] == 0:
                        self.float_max_list[j].add_(
                            torch.max(torch.max(temp), torch.abs(torch.min(temp))))
                    else:
                        self.float_max_list[j].mul_(1 - self.momentum).add_(
                            torch.max(torch.max(temp), torch.abs(torch.min(temp))) * self.momentum)
                    j = j + 1

                    del temp
                    torch.cuda.empty_cache()
                if self.FPGA == False:
                    float_range = max(self.float_max_list).unsqueeze(0)  # 量化前范围
                else:
                    float_max = max(self.float_max_list).unsqueeze(0)  # 量化前范围
                    floor_float_range = 2 ** float_max.log2().floor()
                    ceil_float_range = 2 ** float_max.log2().ceil()
                    if abs(ceil_float_range - float_max) < abs(floor_float_range - float_max):
                        float_range = ceil_float_range
                    else:
                        float_range = floor_float_range
                self.scale = float_range / quantized_range  # 量化比例因子

            if self.quantizer_output == True:

                if self.layer_idx == -1:
                    q_a_concat = copy.deepcopy(outputs[0])

                    move_scale = math.log2(self.scale)
                    concat_scale = -np.array(move_scale).reshape(1, -1)
                    np.savetxt(('./quantizer_output/a_scale_out/concat_scale_%s.txt' % self.name), concat_scale,
                               delimiter='\n')

                    for i in self.layers:
                        q_a_concat[i] = self.quantize(q_a_concat[i])  # 量化
                        q_a_concat[i] = self.round(q_a_concat[i])
                        q_a_concat[i] = self.clamp(q_a_concat[i])  # 截断
                    Q_concat = torch.cat([q_a_concat[i] for i in self.layers], 1)

                    if self.reorder == True:
                        a_para = Q_concat
                        # 重排序参数
                        # print("use activation reorder!")
                        shape_input = a_para.shape[1]
                        num_TN = int(shape_input / self.TN)
                        first = True
                        reorder_a_para = None
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                        a_para_flatten = reorder_a_para
                        q_activation_reorder = a_para_flatten
                        q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                        np.savetxt(('./quantizer_output/q_activation_reorder/r_concat_%s.txt' % self.name),
                                   q_activation_reorder, delimiter='\n')
                        ###保存重排序的二进制文件
                        activation_flat = q_activation_reorder.astype(np.int8)
                        writer = open('./quantizer_output/q_activation_reorder/%s_concat_q_bin' % self.name, "wb")
                        writer.write(activation_flat)
                        writer.close()
                    ##########concat重排序结束

                    Q_concat = np.array(Q_concat.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_out/a_concat_%s.txt' % self.name), Q_concat,
                               delimiter='\n')
                elif int(self.name[1:4]) == self.layer_idx:
                    q_a_concat = copy.deepcopy(outputs[0])

                    move_scale = math.log2(self.scale)
                    concat_scale = -np.array(move_scale).reshape(1, -1)
                    np.savetxt(('./quantizer_output/a_scale_out/concat_scale_%s.txt' % self.name), concat_scale,
                               delimiter='\n')

                    for i in self.layers:
                        q_a_concat[i] = self.quantize(q_a_concat[i])  # 量化
                        q_a_concat[i] = self.round(q_a_concat[i])
                        q_a_concat[i] = self.clamp(q_a_concat[i])  # 截断
                    Q_concat = torch.cat([q_a_concat[i] for i in self.layers], 1)

                    if self.reorder == True:
                        a_para = Q_concat
                        # 重排序参数
                        # print("use activation reorder!")
                        shape_input = a_para.shape[1]
                        num_TN = int(shape_input / self.TN)
                        first = True
                        reorder_a_para = None
                        for k in range(num_TN):
                            temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                            temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                            temp = temp.permute(1, 2, 0).contiguous().view(-1)
                            if first:
                                reorder_a_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                        a_para_flatten = reorder_a_para
                        q_activation_reorder = a_para_flatten
                        q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                        np.savetxt(('./quantizer_output/q_activation_reorder/r_concat_%s.txt' % self.name),
                                   q_activation_reorder, delimiter='\n')
                        ###保存重排序的二进制文件
                        activation_flat = q_activation_reorder.astype(np.int8)
                        writer = open('./quantizer_output/q_activation_reorder/%s_concat_q_bin' % self.name, "wb")
                        writer.write(activation_flat)
                        writer.close()
                    ##########concat重排序结束
                    Q_concat = np.array(Q_concat.cpu()).reshape(1, -1)
                    np.savetxt(('./quantizer_output/q_activation_out/a_concat_%s.txt' % self.name), Q_concat,
                               delimiter='\n')

            # 量化
            if self.training:
                for i in self.layers:
                    outputs[i][0] = self.quantize(outputs[i][0])  # 量化
                    outputs[i][0] = self.round(outputs[i][0])
                    outputs[i][0] = self.clamp(outputs[i][0])  # 截断
                    outputs[i][0] = self.dequantize(outputs[i][0])  # 反量化
                return [torch.cat([outputs[i][0] for i in self.layers], 1),
                        torch.cat([outputs[i][1] for i in self.layers], 1)]
            else:
                for i in self.layers:
                    outputs[i] = self.quantize(outputs[i])  # 量化
                    outputs[i] = self.round(outputs[i])
                    outputs[i] = self.clamp(outputs[i])  # 截断
                    outputs[i] = self.dequantize(outputs[i])  # 反量化
                return torch.cat([outputs[i] for i in self.layers], 1)
        else:
            if self.groups:
                if self.training:
                    return [x[:, (x.shape[1] // 2):], float[:, (x.shape[1] // 2):]]
                else:
                    return x[:, (x.shape[1] // 2):]
            else:
                return outputs[self.layers[0]]
