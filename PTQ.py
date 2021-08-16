import argparse
import test
from torch.utils.data import DataLoader
from models import *
from utils.datasets import *
from utils.utils import *

wdir = 'weights' + os.sep  # weights dir
PTQ_weights = wdir + 'PTQ.pt'


def PTQ(cfg,
        t_data,
        c_data,
        weights=None,
        batch_size=64,
        imgsz=416,
        single_cls=False,
        augment=False,
        a_bit=8,
        w_bit=8, ):
    # Initialize/load model and set device
    device = torch_utils.select_device(opt.device, batch_size=batch_size)
    print('PTQ only support for one gpu!')
    print('')  # skip a line
    # Initialize model
    model = Darknet(cfg, is_gray_scale=opt.gray_scale, maxabsscaler=opt.maxabsscaler)
    q_model = Darknet(cfg, quantized=5, a_bit=a_bit, w_bit=w_bit, FPGA=True, is_gray_scale=opt.gray_scale,
                      maxabsscaler=opt.maxabsscaler,
                      shortcut_way=opt.shortcut_way)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        q_model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)
        load_darknet_weights(q_model, weights, FPGA=True)

    model.to(device)
    q_model.to(device)

    # Configure run
    t_data = parse_data_cfg(t_data)
    t_path = t_data['valid']  # path to test images
    c_data = parse_data_cfg(c_data)
    c_path = c_data['valid']  # path to test images

    # Dataloader
    c_dataset = LoadImagesAndLabels(c_path, imgsz, batch_size, rect=True, single_cls=single_cls,
                                    is_gray_scale=True if opt.gray_scale else False, subset_len=opt.subset_len)
    c_batch_size = min(batch_size, len(c_dataset))
    c_dataloader = DataLoader(c_dataset,
                              batch_size=c_batch_size,
                              num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                              pin_memory=True,
                              collate_fn=c_dataset.collate_fn)

    t_dataset = LoadImagesAndLabels(t_path, imgsz, batch_size, rect=True, single_cls=single_cls,
                                    is_gray_scale=True if opt.gray_scale else False)
    t_batch_size = min(batch_size, len(t_dataset))
    t_dataloader = DataLoader(t_dataset,
                              batch_size=t_batch_size,
                              num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                              pin_memory=True,
                              collate_fn=t_dataset.collate_fn)
    print('')  # skip a line
    print('<.....................test original model.......................>')
    test.test(cfg,
              data=opt.t_data,
              batch_size=batch_size,
              imgsz=imgsz,
              model=model,
              dataloader=t_dataloader,
              rank=-1,
              maxabsscaler=opt.maxabsscaler)

    q_model.train()
    print('')  # skip a line
    print('<.....................Quantize.......................>')

    for batch_i, (imgs, _, _, _) in enumerate(tqdm(c_dataloader)):
        if opt.maxabsscaler:
            imgs = imgs.to(device).float() / 256.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            imgs = imgs * 2 - 1
        else:
            imgs = imgs.to(device).float() / 256.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # Disable gradients
        with torch.no_grad():
            _, _ = q_model(imgs, augment=augment)  # inference and training outputs
    print('')  # skip a line
    print('<.....................test quantized model.......................>')
    print('')  # skip a line
    test.test(cfg,
              data=opt.t_data,
              batch_size=batch_size,
              imgsz=imgsz,
              model=q_model,
              dataloader=t_dataloader,
              quantized=4,
              a_bit=opt.a_bit,
              w_bit=opt.w_bit,
              rank=-1,
              maxabsscaler=opt.maxabsscaler)
    # Save model

    if hasattr(q_model, 'module'):
        model_temp = q_model.module.state_dict()
    else:
        model_temp = q_model.state_dict()
    chkpt = {'epoch': None,
             'best_fitness': None,
             'training_results': None,
             'model': model_temp,
             'optimizer': None}
    # Save last, best and delete
    torch.save(chkpt, PTQ_weights)
    del chkpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--t_data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--c_data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--a-bit', type=int, default=8,
                        help='a-bit')
    parser.add_argument('--w-bit', type=int, default=8,
                        help='w-bit')
    parser.add_argument('--subset_len', type=int, default=-1, help='calibration set len')
    parser.add_argument('--gray_scale', action='store_true', help='gray scale trainning')
    parser.add_argument('--maxabsscaler', '-mas', action='store_true', help='Standarize input to (-1,1)')
    parser.add_argument('--shortcut_way', type=int, default=1, help='--shortcut quantization way')
    opt = parser.parse_args()
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.t_data = list(glob.iglob('./**/' + opt.t_data, recursive=True))[0]  # find file
    opt.c_data = list(glob.iglob('./**/' + opt.c_data, recursive=True))[0]  # find file

    print(opt)

    PTQ(opt.cfg,
        opt.t_data,
        opt.c_data,
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.single_cls,
        opt.augment,
        a_bit=opt.a_bit,
        w_bit=opt.w_bit)
