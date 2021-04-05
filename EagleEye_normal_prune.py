from models import *
from test import test
from utils.utils import *
from utils.prune_utils import *
from utils.datasets import *
import os


def obtain_avg_forward_time(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output


def obtain_filters_mask2(model, CBL_idx, prune_idx, idx_mask):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    # CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            mask = idx_mask[idx]
            # mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain

            if remain == 0:
                print("Channels would be all pruned!")
                raise Exception

            # print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
            #       f'remaining channel: {remain:>4d}')
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())

    # 因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
    # prune_ratio = pruned / total
    # print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask


def obtain_l1_mask(conv_module, random_rate):
    w_copy = conv_module.weight.data.abs().clone()
    w_copy = torch.sum(w_copy, dim=(1, 2, 3))
    length = w_copy.cpu().numpy().shape[0]
    num_retain = int(length * (1 - random_rate))
    if num_retain == 0:
        num_retain = 1
    _, y = torch.topk(w_copy, num_retain)
    mask = torch.zeros(length, dtype=torch.float32).to(w_copy.device)
    mask[y] = 1

    return mask


def rand_prune_and_eval(model, min_rate, max_rate):
    np.random.seed(int(time.time()))
    candidates = 0
    max_mAP = 0

    while True:
        model_copy = deepcopy(model)
        remain_num = 0
        idx_new = dict()
        for idx in prune_idx:
            # bn_module = model_copy.module_list[idx][1]
            conv_module = model_copy.module_list[idx][0]

            random_rate = (max_rate - min_rate) * (np.random.rand(1)) + min_rate
            mask = obtain_l1_mask(conv_module, random_rate)

            idx_new[idx] = mask
            remain_num += int(mask.sum())
            conv_module.weight.data = conv_module.weight.data.permute(1, 2, 3, 0).mul(mask).float().permute(3, 0, 1, 2)
            # bn_module.weight.data.mul_(mask)

        # ---------------
        num_filters, filters_mask = obtain_filters_mask2(model_copy, CBL_idx, prune_idx, idx_new)
        CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

        compact_module_defs = deepcopy(model.module_defs)
        for idx, num in zip(CBL_idx, num_filters):
            assert compact_module_defs[idx]['type'] == 'convolutional'
            compact_module_defs[idx]['filters'] = str(num)
        compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
        current_parameters = obtain_num_parameters(compact_model)
        remain_ratio = 0.3  # 0.2
        delta = 0.05  # 0.03
        # print(current_parameters/origin_nparameters, end='；')
        if current_parameters / origin_nparameters > remain_ratio + delta or current_parameters / origin_nparameters < remain_ratio - delta:
            # print('missing')
            continue

        print("yes---")

        for i in CBLidx2mask:
            CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()
        pruned_model = prune_model_keep_size_forEagleEye(model, prune_idx, CBLidx2mask)
        init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

        compact_model.train()
        with torch.no_grad():
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader)):
                imgs = imgs.cuda().float() / 255.0
                compact_model(imgs)
                if batch_i > 1000:
                    break

        with torch.no_grad():
            mAP = test.test(opt.cfg,
                            opt.data,
                            batch_size=batch_size,
                            imgsz=img_size,
                            model=compact_model,
                            dataloader=testloader,
                            rank=-1,
                            plot=False)[0][2]

        print('candidate: ' + str(candidates), end=" ")
        print('remain_ratio: ' + str(current_parameters / origin_nparameters))
        print(f'mAP of the pruned model is {mAP:.4f}')
        if mAP > max_mAP:
            max_mAP = mAP
            compact_model_winnner = deepcopy(compact_model)
            cfg_name = 'cfg_backup/' + str(candidates) + '.cfg'
            if not os.path.isdir('cfg_backup/'):
                os.makedirs('cfg_backup/')
            pruned_cfg_file = write_cfg(cfg_name, [model.hyperparams.copy()] + compact_module_defs)

        candidates = candidates + 1
        if candidates > 1000:  # 20
            break

    return model_copy, compact_model_winnner


if __name__ == '__main__':
    class opt():
        model_def = "cfg/yolov3/yolov3.cfg"
        data_config = "data/coco2017.data"
        model = 'weights/pretrain_weights/yolov3.weights'
        rect = False
        img_weights = False


    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.20,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.0005,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.05 * 0,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def).to(device)

    if opt.model:
        if opt.model.endswith(".pt"):
            model.load_state_dict(torch.load(opt.model, map_location=device)['model'])
        else:
            _ = load_darknet_weights(model, opt.model)

    data_config = parse_data_cfg(opt.data_config)

    valid_path = data_config["valid"]
    train_path = data_config["train"]
    class_names = load_classes(data_config["names"])

    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    img_size = 416
    batch_size = 16
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 8]),
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(valid_path, img_size, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True),
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 8]),
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    with torch.no_grad():
        origin_model_metric = test.test(opt.cfg,
                                        opt.data,
                                        batch_size=batch_size,
                                        imgsz=img_size,
                                        model=model,
                                        dataloader=testloader,
                                        rank=-1,
                                        plot=False)
    origin_nparameters = obtain_num_parameters(model)

    CBL_idx, Conv_idx, prune_idx = parse_module_defs(model.module_defs)

    print("-------------------------------------------------------")

    percent = 1
    pruning_model, compact_model = rand_prune_and_eval(model, 0, percent)

    # 获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(compact_model.module_defs)

    compact_nparameters = obtain_num_parameters(compact_model)

    random_input = torch.rand((16, 3, 416, 416)).to(device)

    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruning_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    with torch.no_grad():
        compact_model_metric = test.test(opt.cfg,
                                         opt.data,
                                         batch_size=batch_size,
                                         imgsz=img_size,
                                         model=compact_model,
                                         dataloader=testloader,
                                         rank=-1,
                                         plot=False)

    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[1].mean():.6f}', f'{compact_model_metric[1].mean():.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = 'cfg/rand-normal_0-' + str(percent) + '_1000/' + 'rand-normal_0-' + str(percent) + '_1000.cfg'
    # 创建存储目录
    dir_name = 'cfg/rand-normal_0-' + str(percent) + '_1000/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # 由于原始的compact_module_defs将anchor从字符串变为了数组，因此这里将anchors重新变为字符串
    file = open(opt.model_def, 'r')
    lines = file.read().split('\n')
    for line in lines:
        if line.split(' = ')[0] == 'anchors':
            anchor = line.split(' = ')[1]
            break
        if line.split('=')[0] == 'anchors':
            anchor = line.split('=')[1]
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
    weights_dir_name = dir_name.replace('cfg', 'weights')
    if not os.path.isdir(weights_dir_name):
        os.makedirs(weights_dir_name)
    compact_model_name = weights_dir_name + 'rand-normal_0-' + str(percent) + '_1000.weights'

    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')
