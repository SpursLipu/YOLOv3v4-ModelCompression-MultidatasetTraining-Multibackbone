from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.utils import *
from utils.prune_utils import *
import os
import argparse


# 该函数有很重要的意义：
# ①先用深拷贝将原始模型拷贝下来，得到model_copy
# ②将model_copy中，BN层中低于阈值的α参数赋值为0
# ③在BN层中，输出y=α*x+β，由于α参数的值被赋值为0，因此输入仅加了一个偏置β
# ④很神奇的是，network slimming中是将α参数和β参数都置0，该处只将α参数置0，但效果却很好：其实在另外一篇论文中，已经提到，可以先将β参数的效果移到
# 下一层卷积层，再去剪掉本层的α参数

# 该函数用最简单的方法，让我们看到了，如何快速看到剪枝后的效果


def prune_and_eval(model, sorted_bn, percent=.0):
    model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * percent)
    # 获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
    thre = sorted_bn[thre_index]

    print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

    remain_num = 0
    for idx in prune_idx:
        bn_module = model_copy.module_list[idx][1]
        # 根据BN的阈值找到BN的mask
        mask = obtain_bn_mask(bn_module, thre)

        remain_num += int(mask.sum())
        bn_module.weight.data.mul_(mask)
    # with torch.no_grad():
    #     mAP = eval_model(model_copy)[1].mean()

    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1 - remain_num / len(sorted_bn):.3f}')
    # print(f'mAP of the pruned model is {mAP:.4f}')

    return thre


def obtain_filters_mask(model, thre, CBL_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    # CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:

            mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain

            if remain == 0:
                print("Channels would be all pruned!")
                raise Exception

            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        else:
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.copy())

    # 因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask


def obtain_avg_forward_time(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.5, help='global channel prune percent')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    percent = opt.percent
    # 指定GPU
    # torch.cuda.set_device(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg).to(device)

    if opt.weights:
        if opt.weights.endswith(".pt"):
            model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
        else:
            _ = load_darknet_weights(model, opt.weights)

    data_config = parse_data_cfg(opt.data)

    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    # test model
    eval_model = lambda model: test(model=model, cfg=opt.cfg, data=opt.data, imgsz=opt.img_size)
    # 获取参数个数
    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    with torch.no_grad():
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)

    # CBL表示后接BN的卷积层，Other_idx表示不接BN的卷积层和其他层
    CBL_idx, Other_idx, prune_idx = parse_module_defs(model.module_defs)

    # 将所有要剪枝的BN层的γ参数，拷贝到bn_weights列表
    bn_weights = gather_bn_weights(model.module_list, prune_idx)

    # torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
    sorted_bn = torch.sort(bn_weights)[0]
    # 对BN中的γ参数排序
    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for idx in prune_idx:
        # .item()可以得到张量里的元素值
        # 获取每一层中γ参数的最大值
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    # 获取所有层中的最小值
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_thre).nonzero().item() / len(bn_weights)

    print(f'Threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}.')
    # 获得在目标百分百比下的剪植阈值
    threshold = prune_and_eval(model, sorted_bn, percent)

    # 获得保留的卷积核的个数和每层对应的mask
    num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)

    # CBLidx2mask存储CBL_idx中，每一层BN层对应的mask，将需要被剪植的层和剪植后的mask结合起来
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

    # 返回剪植后的模型
    pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)

    with torch.no_grad():
        mAP = eval_model(pruned_model)[1].mean()
    print('after prune_model_keep_size map is {}'.format(mAP))

    # 获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)
    # 生成新模型
    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)
    # 拷贝权重
    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Other_idx, CBLidx2mask)
    # 测试运行速度
    random_input = torch.rand((16, 3, 416, 416)).to(device)
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)

    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[1].mean():.6f}', f'{compact_model_metric[1].mean():.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{percent}_')
    # 创建存储目录
    dir_name = pruned_cfg_name.split('/')[0] + '/' + pruned_cfg_name.split('/')[1]
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # 由于原始的compact_module_defs将anchor从字符串变为了数组，因此这里将anchors重新变为字符串
    file = open(opt.cfg, 'r')
    lines = file.read().split('\n')
    for line in lines:
        if line.split(' = ')[0] == 'anchors':
            anchor = line.split(' = ')[1]
            break
    file.close()
    for item in compact_module_defs:
        if item['type'] == 'shortcut':
            item['from'] = str(item['from'][0])
        elif item['type'] == 'route':
            item['layers'] = ",".join('%s' % i for i in item['layers'])
        elif item['type'] == 'yolo':
            item['mask'] = ",".join('%s' % i for i in item['mask'])
            item['anchors'] = anchor
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    # compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
    compact_model_name = opt.weights.replace('/', f'/prune_{str(percent)}_percent_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')

    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')
