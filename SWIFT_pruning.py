# Author:LiPu
from models import *


def SSP(weights, pof=16, alpha=0.3):
    weights_rtrn = weights.numpy()
    temp_eff = np.absolute((weights))
    shape = temp_eff.shape
    sum = 0
    sum_array = []
    out_feat = int(np.ceil(shape[0] / pof))
    F = np.zeros(shape=[out_feat, shape[1], shape[2], shape[3]])
    a = np.ones(shape=shape)
    temp_eff = np.divide(a, temp_eff, out=np.zeros_like(a), where=temp_eff != 0)
    for b in range(shape[1]):  # Inp feat
        for d in range(shape[2]):  # Row
            for c in range(shape[3]):  # Col
                for a in range(shape[0]):  # Out Feat
                    if ((a != 0) & (a % pof == 0)):
                        sum_array.append(sum)
                        sum = temp_eff[a][b][d][c]
                    else:
                        sum += temp_eff[a][b][d][c]
                if (a % pof != 0):
                    sum_array.append(sum)
                sum = 0
    sum_array = np.asarray(sum_array)
    sum_array = np.reshape(sum_array, (out_feat, shape[1], shape[2], shape[3]))
    sum = np.sum(sum_array, axis=0)

    for b in range(shape[1]):  # Inp feat
        for d in range(shape[2]):  # Row
            for c in range(shape[3]):  # Col
                for a in range(out_feat):  # Out Feat
                    sum_array[a][b][d][c] = sum_array[a][b][d][c] / sum[b][d][c]

    for b in range(shape[1]):  # Inp feat
        for d in range(shape[2]):  # Row
            for c in range(shape[3]):  # Col
                for a in range(out_feat):  # Out Feat
                    temp = sum_array[a][b][d][c]
                    sum_temp = 0
                    count = 0
                    while (count < out_feat):
                        if (sum_array[count][b][d][c] < temp):
                            if (count == a):
                                pass
                            else:
                                sum_temp += 1
                        else:
                            pass
                        count += 1
                    F[a][b][d][c] = (1 / out_feat) * (sum_temp)
    sparse = 0
    for b in range(shape[1]):  # Inp feat
        for d in range(shape[2]):  # Row
            for c in range(shape[3]):  # Col
                min = np.amin(F[:, b, d, c])
                max = np.amax(F[:, b, d, c])
                cnt = 0
                a = 0
                while (a < shape[0]):  # Out Feat
                    if (a == 0):
                        cnt = 0
                    elif (a % pof == 0):
                        cnt += 1
                    else:
                        cnt = cnt
                    if ((max - F[cnt][b][d][c]) < alpha):
                        a = a + pof
                    else:
                        count = 1
                        while ((count <= pof) & (a < shape[0])):
                            weights_rtrn[a][b][d][c] = 0
                            sparse += 1
                            a += 1
                            count += 1

    total = shape[0] * shape[1] * shape[2] * shape[3]
    print("Number of zero is %f percent" % ((sparse / total) * 100))
    return weights_rtrn


cfg = "cfg/yolov3/yolov3-hand.cfg"
weights = 'weights/pretrain_weights/yolov3-hand-best.weights'

model = Darknet(cfg)
if weights.endswith('.pt'):
    chkpt = torch.load(weights)
    chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(chkpt['model'], strict=False)
else:
    load_darknet_weights(model, weights)

for i, (mdef, module) in enumerate(zip(model.module_defs[:], model.module_list[:])):
    if mdef['type'] == 'convolutional':
        if mdef['activation'] != 'linear':
            conv_layer = module[0]
            if conv_layer.in_channels != 3:
                weights = conv_layer.weight.data
                weights = SSP(weights)
                conv_layer.weight.data.copy_(torch.from_numpy(weights))
print("finish pruning!")
save_weights(model, path='weights/SSP.weights')

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
