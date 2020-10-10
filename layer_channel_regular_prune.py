from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse

filter_switch = [each for each in range(1024) if (each % 8 == 0)]


# %%
def obtain_filters_mask(model, thre, CBL_idx, shortcut_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    idx_new = dict()
    # CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            if idx not in shortcut_idx:

                mask = obtain_bn_mask(bn_module, thre).cpu().numpy()

                # 保证通道数为的倍数
                mask_cnt = int(mask.sum())
                if mask_cnt == 0:
                    this_layer_sort_bn = bn_module.weight.data.abs().clone()
                    sort_bn_values = torch.sort(this_layer_sort_bn)[0]
                    bn_cnt = bn_module.weight.shape[0]
                    this_layer_thre = sort_bn_values[bn_cnt - 8]
                    mask = obtain_bn_mask(bn_module, this_layer_thre).cpu().numpy()
                else:
                    for i in range(len(filter_switch)):
                        if mask_cnt <= filter_switch[i]:
                            mask_cnt = filter_switch[i]
                            break
                    this_layer_sort_bn = bn_module.weight.data.abs().clone()
                    sort_bn_values = torch.sort(this_layer_sort_bn)[0]
                    bn_cnt = bn_module.weight.shape[0]
                    this_layer_thre = sort_bn_values[bn_cnt - mask_cnt]
                    mask = obtain_bn_mask(bn_module, this_layer_thre).cpu().numpy()

                idx_new[idx] = mask
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                # if remain == 0:
                #     print("Channels would be all pruned!")
                #     raise Exception

                # print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                #     f'remaining channel: {remain:>4d}')
            else:
                # 如果idx在shortcut_idx之中，则试跳连层的两层的mask相等
                mask = idx_new[shortcut_idx[idx]]
                idx_new[idx] = mask
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


def prune_and_eval(model, sorted_bn, shortcut_idx, percent=.0):
    model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * percent)
    # 获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
    thre1 = sorted_bn[thre_index]

    print(f'Channels with Gamma value less than {thre1:.8f} are pruned!')

    remain_num = 0
    idx_new = dict()
    for idx in prune_idx:

        if idx not in shortcut_idx:

            bn_module = model_copy.module_list[idx][1]

            mask = obtain_bn_mask(bn_module, thre1)
            # 记录剪枝后，每一层卷积层对应的mask
            # idx_new[idx]=mask.cpu().numpy()
            idx_new[idx] = mask
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
            # bn_module.bias.data.mul_(mask*0.0001)
        else:

            bn_module = model_copy.module_list[idx][1]

            mask = idx_new[shortcut_idx[idx]]
            idx_new[idx] = mask

            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)

        # print(int(mask.sum()))

    # with torch.no_grad():
    #     mAP = eval_model(model_copy)[0][2]

    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1 - remain_num / len(sorted_bn):.3f}')
    # print(f'mAP of the pruned model is {mAP:.4f}')

    return thre1


def prune_and_eval2(model, prune_shortcuts=[]):
    model_copy = deepcopy(model)
    for idx in prune_shortcuts:
        for i in [idx, idx - 1]:
            bn_module = model_copy.module_list[i][1]

            mask = torch.zeros(bn_module.weight.data.shape[0]).cuda()
            bn_module.weight.data.mul_(mask)

    with torch.no_grad():
        mAP = eval_model(model_copy)[0][2]

    print(f'simply mask the BN Gama of to_be_pruned CBL as zero, now the mAP is {mAP:.4f}')


# %%
def obtain_filters_mask2(model, CBL_idx, prune_shortcuts):
    filters_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        mask = np.ones(bn_module.weight.data.shape[0], dtype='float32')
        filters_mask.append(mask.copy())
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    for idx in prune_shortcuts:
        for i in [idx, idx - 1]:
            bn_module = model.module_list[i][1]
            mask = np.zeros(bn_module.weight.data.shape[0], dtype='float32')
            CBLidx2mask[i] = mask.copy()
    return CBLidx2mask


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
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--shortcuts', type=int, default=8, help='how many shortcut layers will be pruned,\
        pruning one shortcut will also prune two CBL,yolov3 has 23 shortcuts')
    parser.add_argument('--percent', type=float, default=0.6, help='global channel prune percent')
    parser.add_argument('--layer_keep', type=float, default=0.01, help='channel keep percent per layer')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ', opt.weights)

    eval_model = lambda model: test(model=model, cfg=opt.cfg, data=opt.data, batch_size=16, imgsz=img_size)
    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)

    ##############################################################
    # 先剪通道
    # 与normal_prune不同的是这里需要获得shortcu_idx和short_all
    # 其中shortcut_idx存储的是对应关系，故shortcut[x]就对应的是与第x-1卷积层相加层的索引值
    # shortcut_all存储的是所有相加层
    CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all = parse_module_defs2(model.module_defs)

    # 将所有要剪枝的BN层的γ参数，拷贝到bn_weights列表
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    # 对BN中的γ参数排序
    # torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉一层中的所有channel的最高阈值(每个BN层中gamma的最大值在所有层中最小值即为阈值上限)
    highest_thre = []
    for idx in prune_idx:
        # .item()可以得到张量里的元素值
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_thre).nonzero().item() / len(bn_weights)

    print(f'Threshold should be less than {highest_thre:.8f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}.')

    percent = opt.percent
    threshold = prune_and_eval(model, sorted_bn, shortcut_idx, percent)

    num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, shortcut_idx, prune_idx)

    # CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

    pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)

    with torch.no_grad():
        mAP = eval_model(pruned_model)[0][2]
    print('after prune_model_keep_size map is {}'.format(mAP))

    # 获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)

    # for item_def in compact_module_defs:
    #     print(item_def)

    compact_model1 = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
    compact_nparameters1 = obtain_num_parameters(compact_model1)

    init_weights_from_loose_model(compact_model1, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

    print('testing the channel pruned model...')
    with torch.no_grad():
        compact_model_metric1 = eval_model(compact_model1)

    #########################################################
    # 再剪层
    print('\nnow we prune shortcut layers and corresponding CBLs')

    CBL_idx, Conv_idx, shortcut_idx = parse_module_defs4(compact_model1.module_defs)
    print('all shortcut_idx:', [i + 1 for i in shortcut_idx])

    bn_weights = gather_bn_weights(compact_model1.module_list, shortcut_idx)

    sorted_bn = torch.sort(bn_weights)[0]

    # highest_thre = torch.zeros(len(shortcut_idx))
    # for i, idx in enumerate(shortcut_idx):
    #     highest_thre[i] = compact_model1.module_list[idx][1].weight.data.abs().max().clone()
    # _, sorted_index_thre = torch.sort(highest_thre)

    # 这里更改了选层策略，由最大值排序改为均值排序，均值一般表现要稍好，但不是绝对，可以自己切换尝试；前面注释的四行为原策略。
    bn_mean = torch.zeros(len(shortcut_idx))
    for i, idx in enumerate(shortcut_idx):
        bn_mean[i] = compact_model1.module_list[idx][1].weight.data.abs().mean().clone()
    _, sorted_index_thre = torch.sort(bn_mean)

    prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:opt.shortcuts]]]
    prune_shortcuts = [int(x) for x in prune_shortcuts]

    index_all = list(range(len(compact_model1.module_defs)))
    index_prune = []
    for idx in prune_shortcuts:
        index_prune.extend([idx - 1, idx, idx + 1])
    index_remain = [idx for idx in index_all if idx not in index_prune]

    print('These shortcut layers and corresponding CBL will be pruned :', index_prune)

    prune_and_eval2(compact_model1, prune_shortcuts)

    CBLidx2mask = obtain_filters_mask2(compact_model1, CBL_idx, prune_shortcuts)

    pruned_model = prune_model_keep_size(compact_model1, CBL_idx, CBL_idx, CBLidx2mask)

    with torch.no_grad():
        mAP = eval_model(pruned_model)[0][2]
    print("after transfering the offset of pruned CBL's activation, map is {}".format(mAP))

    compact_module_defs = deepcopy(compact_model1.module_defs)

    for module_def in compact_module_defs:
        if module_def['type'] == 'route':
            from_layers = [int(s) for s in module_def['layers']]
            if len(from_layers) == 2:
                count = 0
                for i in index_prune:
                    if i <= from_layers[1]:
                        count += 1
                from_layers[1] = from_layers[1] - count
                # from_layers = ', '.join([str(s) for s in from_layers])
                module_def['layers'] = from_layers

    compact_module_defs = [compact_module_defs[i] for i in index_remain]
    compact_model2 = Darknet([compact_model1.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)

    compact_nparameters2 = obtain_num_parameters(compact_model2)

    print('testing the final model')
    with torch.no_grad():
        compact_model_metric2 = eval_model(compact_model2)

    ################################################################
    # 剪枝完毕，测试速度

    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    print('testing inference time...')
    pruned_forward_time, output = obtain_avg_forward_time(random_input, model)
    compact_forward_time1, compact_output1 = obtain_avg_forward_time(random_input, compact_model1)
    compact_forward_time2, compact_output2 = obtain_avg_forward_time(random_input, compact_model2)

    metric_table = [
        ["Metric", "Before", "After prune channels", "After prune layers(final)"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric1[0][2]:.6f}',
         f'{compact_model_metric2[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters1}", f"{compact_nparameters2}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time1:.4f}', f'{compact_forward_time2:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    pruned_cfg_name = opt.cfg.replace('/',
                                      f'/prune_regular_{opt.percent}_keep_{opt.layer_keep}_{opt.shortcuts}_shortcut_')
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

    compact_model_name = opt.weights.replace('/',
                                             f'/prune_regular_{opt.percent}_keep_{opt.layer_keep}_{opt.shortcuts}_shortcut_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model2, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')
