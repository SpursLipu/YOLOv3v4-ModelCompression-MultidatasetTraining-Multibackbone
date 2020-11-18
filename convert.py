import argparse
import struct

from models import *  # set ONNX_EXPORT in models.py

from utils.utils import *


def convert():
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights, half = opt.weights, opt.half

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Initialize model
    model = Darknet(opt.cfg, img_size, quantized=opt.quantized, a_bit=opt.a_bit, w_bit=opt.w_bit, BN_Fold=opt.BN_Fold,
                    FPGA=opt.FPGA)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights, BN_Fold=opt.BN_Fold)
    if opt.quantized == 0:
        save_weights(model, path='weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '-best.weights')
    else:
        # Fuse Conv2d + BatchNorm2d layers
        if not opt.BN_Fold:
            model.fuse(quantized=opt.quantized, BN_Fold=opt.BN_Fold, FPGA=opt.FPGA)

        w_file = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_weights.bin', 'wb')
        b_file = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_bias.bin', 'wb')
        if opt.quantized == 1:
            w_scale = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_w_scale.bin', 'wb')
            a_scale = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_a_scale.bin', 'wb')
            if opt.FPGA == True:
                b_scale = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_b_scale.bin', 'wb')
        for _, (mdef, module) in enumerate(zip(model.module_defs, model.module_list)):
            print(mdef)
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]

                if opt.quantized != 0 and opt.BN_Fold:
                    # 使用BN训练中量化，融合BN参数
                    weight, bias = conv_layer.BN_fuse()
                else:
                    weight, bias = conv_layer.weight, conv_layer.bias
                if opt.quantized == 1:
                    # 得到缩放因子
                    activate_scale = -math.log(conv_layer.activation_quantizer.scale.cpu().data.numpy()[0], 2)
                    weight_scale = -math.log(conv_layer.weight_quantizer.scale.cpu().data.numpy()[0], 2)
                    a = struct.pack('<i', int(activate_scale))
                    a_scale.write(a)
                    a = struct.pack('<i', int(weight_scale))
                    w_scale.write(a)
                # 处理weights
                para = conv_layer.weight_quantizer.get_weights(weight)

                if opt.reorder:
                    # 重排序参数
                    print("use reorder!")
                    shape_output = para.shape[0]
                    shape_input = para.shape[1]
                    num_TN = int(shape_input / opt.TN)
                    remainder_TN = shape_input % opt.TN
                    num_TM = int(shape_output / opt.TM)
                    remainder_TM = shape_output % opt.TM
                    first = True
                    for j in range(num_TM):
                        for k in range(num_TN):
                            temp = para[j * opt.TM:(j + 1) * opt.TM, k * opt.TN:(k + 1) * opt.TN, :, :]
                            temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                            temp = temp.permute(2, 0, 1).contiguous().view(-1)
                            if first:
                                reorder_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_para = np.append(reorder_para, temp.cpu().data.numpy())
                        temp = para[j * opt.TM:(j + 1) * opt.TM, num_TN * opt.TN:num_TN * opt.TN + remainder_TN, :, :]
                        temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                        temp = temp.permute(2, 0, 1).contiguous().view(-1)
                        if first:
                            reorder_para = temp.clone().cpu().data.numpy()
                            first = False
                        else:
                            reorder_para = np.append(reorder_para, temp.cpu().data.numpy())

                    for k in range(num_TN):
                        temp = para[num_TM * opt.TM:num_TM * opt.TM + remainder_TM, k * opt.TN:(k + 1) * opt.TN, :, :]
                        temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                        temp = temp.permute(2, 0, 1).contiguous().view(-1)
                        if first:
                            reorder_para = temp.clone().cpu().data.numpy()
                            first = False
                        else:
                            reorder_para = np.append(reorder_para, temp.cpu().data.numpy())
                    temp = para[num_TM * opt.TM:num_TM * opt.TM + remainder_TM,
                           num_TN * opt.TN:num_TN * opt.TN + remainder_TN, :, :]
                    temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                    temp = temp.permute(2, 0, 1).contiguous().view(-1)
                    if first:
                        reorder_para = temp.clone().cpu().data.numpy()
                        first = False
                    else:
                        reorder_para = np.append(reorder_para, temp.cpu().data.numpy())

                    para_flatten = reorder_para
                else:
                    para_flatten = para.cpu().data.numpy().flatten()  # 展开

                # 存储weights
                for i in para_flatten:
                    if opt.w_bit == 16:
                        # Dorefa量化为非对称量化 Google量化为对称量化
                        if opt.quantized == 1:
                            a = struct.pack('<h', int(i))
                        if opt.quantized == 2:
                            a = struct.pack('<H', int(i))
                    elif opt.w_bit == 8:
                        # Dorefa量化为非对称量化 Google量化为对称量化
                        if opt.quantized == 1:
                            a = struct.pack('b', int(i))
                        if opt.quantized == 2:
                            a = struct.pack('B', int(i))
                    else:
                        a = struct.pack('<f', i)
                    w_file.write(a)

                # 处理bias
                if bias != None:
                    if opt.FPGA == True:
                        # 生成量化后的参数
                        para = conv_layer.bias_quantizer.get_weights(bias)
                        bias_scale = -math.log(conv_layer.bias_quantizer.scale.cpu().data.numpy()[0], 2)
                        a = struct.pack('<i', int(bias_scale))
                        b_scale.write(a)
                    else:
                        para = bias
                    # print(para.shape)
                    para_flatten = para.cpu().data.numpy().flatten()  # 展开
                    # 存储bias
                    for i in para_flatten:
                        if opt.quantized == 0:
                            a = struct.pack('<f', i)  # 小端浮点                 大端，浮点32>f
                        else:
                            if opt.w_bit == 16:
                                # Dorefa量化为非对称量化 Google量化为对称量化
                                if opt.quantized == 1:
                                    a = struct.pack('<h', int(i))
                                if opt.quantized == 2:
                                    a = struct.pack('<H', int(i))
                            elif opt.w_bit == 8:
                                # Dorefa量化为非对称量化 Google量化为对称量化
                                if opt.quantized == 1:
                                    a = struct.pack('b', int(i))
                                if opt.quantized == 2:
                                    a = struct.pack('B', int(i))
                            else:
                                a = struct.pack('<f', i)
                        b_file.write(a)
            if mdef['type'] == 'yolo':
                yolo_layer = module
                activate_scale = -math.log(yolo_layer.scale.data.numpy(), 2)
                a = struct.pack('<i', int(activate_scale))
                a_scale.write(a)
        if opt.quantized == 1:
            w_scale.close()
            a_scale.close()
            if opt.FPGA == True:
                b_scale.close()
        w_file.close()
        b_file.close()
    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.8, help='iou threshold for non-maximum suppression')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--quantized', type=int, default=0,
                        help='0:quantization way one Ternarized weight and 8bit activation')
    parser.add_argument('--a-bit', type=int, default=8,
                        help='a-bit')
    parser.add_argument('--w-bit', type=int, default=8,
                        help='w-bit')
    parser.add_argument('--BN_Fold', action='store_true', help='BN_Fold')
    parser.add_argument('--FPGA', action='store_true', help='FPGA')
    parser.add_argument('--reorder', action='store_true', help='reorder')
    parser.add_argument('--TN', type=int, default=8, help='TN')
    parser.add_argument('--TM', type=int, default=64, help='TN')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        convert()
