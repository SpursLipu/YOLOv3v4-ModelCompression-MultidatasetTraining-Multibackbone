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
        batch_size=16,
        imgsz=416,
        single_cls=False,
        augment=False,
        a_bit=8,
        w_bit=8,
        FPGA=False):
    # Initialize/load model and set device
    device = torch_utils.select_device(opt.device, batch_size=batch_size)

    # Initialize model
    model = Darknet(cfg)
    q_model = Darknet(cfg, quantized=4, a_bit=a_bit, w_bit=w_bit, FPGA=FPGA)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        q_model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)
        load_darknet_weights(q_model, weights, FPGA=FPGA)

    model.to(device)
    q_model.to(device)

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        q_model = nn.DataParallel(q_model)
    # Configure run
    t_data = parse_data_cfg(t_data)
    t_path = t_data['valid']  # path to test images
    c_data = parse_data_cfg(c_data)
    c_path = c_data['valid']  # path to test images

    # Dataloader
    c_dataset = LoadImagesAndLabels(c_path, imgsz, batch_size, rect=True, single_cls=single_cls)
    c_batch_size = min(batch_size, len(c_dataset))
    c_dataloader = DataLoader(c_dataset,
                              batch_size=c_batch_size,
                              num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                              pin_memory=True,
                              collate_fn=c_dataset.collate_fn)

    t_dataset = LoadImagesAndLabels(t_path, imgsz, batch_size, rect=True, single_cls=single_cls)
    t_batch_size = min(batch_size, len(t_dataset))
    t_dataloader = DataLoader(t_dataset,
                              batch_size=t_batch_size,
                              num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                              pin_memory=True,
                              collate_fn=t_dataset.collate_fn)

    test.test(cfg,
              data=opt.t_data,
              batch_size=batch_size,
              imgsz=imgsz,
              model=model,
              dataloader=t_dataloader)

    q_model.train()

    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(c_dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # Disable gradients
        with torch.no_grad():
            _, _ = q_model(imgs, augment=augment)  # inference and training outputs

    test.test(cfg,
              data=opt.t_data,
              batch_size=batch_size,
              imgsz=imgsz,
              model=q_model,
              dataloader=t_dataloader,
              quantized=3,
              a_bit=opt.a_bit,
              w_bit=opt.w_bit,
              FPGA=opt.FPGA)
    # Save model

    if hasattr(q_model, 'module'):
        model_temp = q_model.module.state_dict()
    else:
        model_temp = q_model.state_dict()
    chkpt = {'model': model_temp, }

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
    parser.add_argument('--FPGA', action='store_true', help='FPGA')

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
        w_bit=opt.w_bit,
        FPGA=opt.FPGA)
