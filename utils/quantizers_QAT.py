import torch
from torch import nn
from torch import autograd
import numpy as np

NUM_BATCHES_CALIBRATION = 200


class Round(autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grads):
        return grads


class Quantizer(nn.Module):

    def __init__(self, bits_precision, range_tracker):
        super().__init__()
        self.bits_precision = bits_precision
        self.range_tracker = range_tracker
        self.register_buffer('scale', torch.tensor(1.))
        self.register_buffer('zero_point', torch.tensor(0.))
        self.register_buffer('num_batches_calibration', torch.tensor(0))

    def _reset_parameters(self):
        self.scale.fill_(0.)
        self.zero_point.zero_()
        # self.min_val.zero_()
        # self.max_val.zero_()
        # self.quant_min_th.zero_()
        # self.quant_max_th.zero_()

    def update_params(self):
        raise NotImplementedError

    def quantize(self, inputs):
        outputs = inputs / self.scale
        return outputs

    def round(self, inputs):
        # outputs = torch.round(inputs) + inputs - inputs.detach()
        outputs = Round.apply(inputs)
        return outputs

    def clamp(self, inputs):
        # outputs = torch.clamp(inputs, self.min_val, self.max_val)
        outputs = torch.clamp(inputs, self.min_val, self.max_val)
        # print('clamp unique',np.unique(outputs.detach().cpu().numpy()).shape)
        return outputs

    def dequantize(self, inputs):
        outputs = inputs * self.scale
        # print('dequantize unique',np.unique(outputs.detach().cpu().numpy()).shape)
        return outputs

    def forward(self, inputs):
        if self.num_batches_calibration <= NUM_BATCHES_CALIBRATION:
            self.range_tracker(inputs)
            self.update_params()
        outputs = self.quantize(inputs)
        outputs = self.round(outputs)
        outputs = self.clamp(outputs)
        outputs = self.dequantize(outputs)
        return outputs


class Signed_Symmetric_Quantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits_precision - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits_precision - 1)) - 1))

    def update_params(self):
        self._reset_parameters()
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))
        self.scale = float_range / quantized_range
        self.zero_point = torch.round(-self.range_tracker.min_val / self.scale)
        self.num_batches_calibration.add_(1)


class Signed_Asymmetric_Quantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits_precision - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits_precision - 1)) - 1))

    def update_params(self):
        self._reset_parameters()
        quantized_range = self.max_val - self.min_val
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        self.scale = float_range / quantized_range
        self.zero_point = torch.round(-self.range_tracker.min_val / self.scale)
        self.num_batches_calibration.add_(1)


class Unsigned_Quantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits_precision) - 1))

    def update_params(self):
        self._reset_parameters()
        quantized_range = self.max_val - self.min_val
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        self.scale = float_range / quantized_range
        self.zero_point = torch.round(-self.range_tracker.min_val / self.scale)
        self.num_batches_calibration.add_(1)


def test():
    from .range_trackers import GlobalRangeTracker
    net = Unsigned_Quantizer(8, GlobalRangeTracker(per_dim=0, track_type='weight'))
    out = net(torch.randn([1, 3, 16, 16]))
    print(net)
    for k, v in net.state_dict().items():
        print(k, v)


if __name__ == '__main__':
    test()
