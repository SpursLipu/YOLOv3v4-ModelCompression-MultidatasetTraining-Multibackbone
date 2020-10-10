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
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    save_weights(model, path='weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '-best.weights')

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    w_file = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_weights.bin', 'wb')
    b_file = open('weights/' + opt.cfg.split('/')[-1].replace('.cfg', '') + '_bias.bin', 'wb')
    stat = model.state_dict()
    for name in stat:
        # print(name)
        if name.find('weight') >= 0:
            file = w_file
        elif name.find('bias') >= 0:
            file = b_file
        else:
            continue

        para = stat[name]
        # print(para.shape)
        para_flatten = para.cpu().data.numpy().flatten()  # 展开

        for i in para_flatten:
            a = struct.pack('<f', i)  # 小端浮点                 大端，浮点32>f
            file.write(a)

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
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        convert()
