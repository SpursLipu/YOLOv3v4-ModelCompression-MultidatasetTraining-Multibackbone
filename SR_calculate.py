# Author:LiPu
from models import *

cfg = "cfg/yolov3/yolov3-hand.cfg"
weights = 'weights/best.pt'
model = Darknet(cfg)
if weights.endswith('.pt'):
    chkpt = torch.load(weights)
    chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(chkpt['model'], strict=False)
else:
    load_darknet_weights(model, weights)
count = 0
total = 0
for i, (mdef, module) in enumerate(zip(model.module_defs[:], model.module_list[:])):
    if mdef['type'] == 'convolutional':
        if mdef['activation'] != 'linear':
            conv_layer = module[0]
            if conv_layer.in_channels != 3:
                weights = conv_layer.weight.data.view(-1)
                weights[weights != 0] = 1
                count = count + (weights.shape[0] - torch.sum(weights))
                total = total + weights.shape[0]
print("SR = ", count / total)
