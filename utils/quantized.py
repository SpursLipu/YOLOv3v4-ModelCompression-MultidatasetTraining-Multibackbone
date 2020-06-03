import torch
from torch import nn
from torch import autograd
from .quantizers_QAT import *
from .range_trackers import *
import numpy as np

FREEZE_BN_DELAY_DEFAULT = 2000


class QuantizedConv2d(nn.Conv2d):

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
            enable_quant=True,
            w_bit=8,
            a_bit=8,
            activation_quantizer=None,
            weight_quantizer=None
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
        self.activation_quantizer = Signed_Asymmetric_Quantizer(
            bits_precision=a_bit,
            range_tracker=AveragedRangeTracker(per_dim=None, track_type='activation')
        )
        self.weight_quantizer = Signed_Asymmetric_Quantizer(
            bits_precision=w_bit,
            range_tracker=GlobalRangeTracker(per_dim=0, track_type='weight_per_channel')  # weight_per_channel
        )
        # self.activation_quantizer = activation_quantizer or Activation_Quantizer(bits_precision=a_bit, in_channels = in_channels)
        # self.weight_quantizer = weight_quantizer or Weight_Quantizer(bits_precision=w_bit, in_channels = in_channels)
        self.quantization = enable_quant

    def forward(self, inputs):

        weight = self.weight
        if self.quantization:
            # inputs = self.activation_quantizer(inputs)

            ## do not quantize input images
            if inputs.size(1) != 3:
                inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)
            # print('x unique',np.unique(inputs.detach().cpu().numpy()).shape)
            # print('w unique',np.unique(weight.detach().cpu().numpy()).shape)

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        return outputs


class BatchNormFoldedQuantizedConv2d(QuantizedConv2d):

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
            w_bit=8,
            a_bit=8,
            enable_quant=True,
            activation_quantizer=None,
            weight_quantizer=None
    ):
        assert bias is False

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            activation_quantizer=activation_quantizer,
            weight_quantizer=weight_quantizer
        )
        self.activation_quantizer = Signed_Asymmetric_Quantizer(
            bits_precision=a_bit,
            range_tracker=AveragedRangeTracker(per_dim=None, track_type='activation')
        )
        self.weight_quantizer = Signed_Asymmetric_Quantizer(
            bits_precision=w_bit,
            range_tracker=GlobalRangeTracker(per_dim=0, track_type='weight_per_channel')
        )
        # self.activation_quantizer = Activation_Quantizer(bits_precision=a_bit, in_channels = in_channels)
        # self.weight_quantizer = Weight_Quantizer(bits_precision=w_bit, in_channels = in_channels)

        self.eps = eps
        self.momentum = momentum

        self.register_parameter('beta', nn.Parameter(torch.zeros(out_channels)))
        self.register_parameter('gamma', nn.Parameter(torch.ones(out_channels)))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.ones(out_channels))
        self.register_buffer('batch_std', torch.ones(out_channels))
        self.register_buffer('num_batch_tracked', torch.zeros(1))

        self.batch_stats = False
        self.quantization = enable_quant

    def use_batch_stats(self):
        self.batch_stats = True

    def use_running_stats(self):
        self.batch_stats = False

    def forward(self, inputs):

        def reshape_to_activation(inputs):
            return inputs.reshape(1, -1, 1, 1)

        def reshape_to_weight(inputs):
            return inputs.reshape(-1, 1, 1, 1)

        def reshape_to_bias(inputs):
            return inputs.reshape(-1)

        if self.training:

            outputs = nn.functional.conv2d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )

            dims = [dim for dim in range(4) if dim != 1]
            channel_size = outputs.size(1)
            n = outputs.numel() / channel_size

            self.batch_mean = torch.mean(outputs, dim=dims)
            self.batch_var = torch.var(outputs, dim=dims)
            corrected_var = self.batch_var * (n / (n - 1))  ## update running_var use unbiased variance
            self.batch_std = torch.sqrt(self.batch_var + self.eps)
            self.num_batch_tracked.add_(1)

            # # fix running stat when finetune quantized model
            if not self.quantization:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * self.batch_mean)
                    self.running_mean.div_(1 - (1 - self.momentum) ** self.num_batch_tracked)

                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * self.batch_var)
                    self.running_var.div_(1 - (1 - self.momentum) ** self.num_batch_tracked)

            # with torch.no_grad():
            #     self.running_mean.mul_(1 - self.momentum).add_(self.momentum * self.batch_mean)
            #     self.running_var.mul_(1 - self.momentum).add_(self.momentum * self.batch_var)
            #     self.num_batch_tracked.add_(1)

        running_mean = self.running_mean
        running_var = self.running_var
        running_std = torch.sqrt(running_var + self.eps)

        # ******************** calculate quantized and float bn-fold's weight ***************************
        if self.quantization:

            ## bn fold use running stat
            weight = self.weight * reshape_to_weight(self.gamma / running_std)
            bias = reshape_to_bias(self.beta - self.gamma * running_mean / running_std)

            ## do not quantize input images
            if inputs.size(1) != 3:
                inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(weight)

            # print('x unique',np.unique(inputs.detach().cpu().numpy()).shape)
            # print('w unique',np.unique(weight.detach().cpu().numpy()).shape)

            print(weight)

            ## freeze_bn_delay control quantization finetune
            if self.num_batch_tracked < FREEZE_BN_DELAY_DEFAULT:
                self.use_batch_stats()
            else:
                self.use_running_stats()

        else:
            if self.training:
                weight = self.weight * reshape_to_weight(self.gamma / self.batch_std)
                bias = reshape_to_bias(self.beta - self.gamma * self.batch_mean / self.batch_std)
            else:
                weight = self.weight * reshape_to_weight(self.gamma / running_std)
                bias = reshape_to_bias(self.beta - self.gamma * running_mean / running_std)
        # **************************************************************************************************

        # if self.quantization:
        #     ## do not quantize input images
        #     if inputs.size(1) != 3:
        #         inputs = self.activation_quantizer(inputs)
        #     weight = self.weight_quantizer(weight)

        # if self.freeze_bn_delay < 1000:
        #     self.use_batch_stats()
        # else:
        #     self.use_running_stats()

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        if self.training and self.batch_stats and self.quantization:
            outputs *= reshape_to_activation(running_std / self.batch_std)
            outputs += reshape_to_activation(
                self.gamma * (running_mean / running_std - self.batch_mean / self.batch_std))

        return outputs
