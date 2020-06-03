import torch
from torch import nn


class RangeTracker(nn.Module):

    def __init__(self, per_dim):
        super().__init__()
        self.per_dim = per_dim
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)
        self.register_buffer('step', torch.tensor(0))
        # self.register_buffer('min_val', torch.tensor(0.))
        # self.register_buffer('max_val', torch.tensor(0.))

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, inputs):

        if self.track_type == 'activation':
            min_val = torch.min(inputs)
            max_val = torch.max(inputs)

        elif self.track_type == 'weight':
            min_val = torch.min(inputs)
            max_val = torch.max(inputs)

        elif self.track_type == 'weight_per_channel':
            view_dims = [inputs.shape[i] for i in range(self.per_dim + 1)] + [-1]
            tv = inputs.view(*view_dims)
            min_val = tv.min(dim=-1)[0].reshape([inputs.shape[self.per_dim], 1, 1, 1])
            max_val = tv.max(dim=-1)[0].reshape([inputs.shape[self.per_dim], 1, 1, 1])
        else:
            pass

        self.update_range(min_val, max_val)


class GlobalRangeTracker(RangeTracker):

    def __init__(self, per_dim, track_type):
        super().__init__(per_dim)
        self.track_type = track_type

    def update_range(self, min_val, max_val):
        self.min_val = torch.min(self.min_val, min_val) if self.min_val is not None else min_val
        self.max_val = torch.max(self.max_val, max_val) if self.max_val is not None else max_val


class AveragedRangeTracker(RangeTracker):

    def __init__(self, per_dim, track_type, momentum=0.01):
        super().__init__(per_dim)
        self.momentum = momentum
        self.track_type = track_type

    def update_range(self, min_val, max_val):
        # self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum if self.min_val is not None else min_val
        # self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum if self.max_val is not None else max_val

        self.step.add_(1)

        if self.min_val is not None:
            self.min_val.mul_(1 - self.momentum).add_(self.momentum * min_val)
            self.min_val.div_(1 - (1 - self.momentum) ** self.step)  # Bias correction
        else:
            self.min_val = min_val

        if self.max_val is not None:
            self.max_val.mul_(1 - self.momentum).add_(self.momentum * max_val)
            self.max_val.div_(1 - (1 - self.momentum) ** self.step)  # Bias correction
        else:
            self.max_val = max_val


def test():
    net = GlobalRangeTracker([1, 3, 1, 1])
    inputs = torch.randn([2, 3, 4, 4])

    print(inputs)
    # out = net(inputs)
    # print (net.min_val)
    # for k,v in net.state_dict().items():
    #     print(k, v)
    print(torch.min(inputs))
    print(inputs)


if __name__ == '__main__':
    test()
