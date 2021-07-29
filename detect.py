import argparse
import subprocess
import Val_upsample
import os

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect(save_img=False):
    if opt.quantizer_output == True:
        tmp_dir = 'quantizer_output'
        subprocess.Popen("rm -rf %s" % tmp_dir, shell=True)
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, view_img, save_txt = opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz, quantized=opt.quantized, quantizer_output=opt.quantizer_output,
                    layer_idx=opt.layer_idx,
                    reorder=opt.reorder, TN=opt.TN, TM=opt.TM, a_bit=opt.a_bit, w_bit=opt.w_bit, FPGA=opt.FPGA,
                    is_gray_scale=opt.gray_scale, maxabsscaler=opt.maxabsscaler, shortcut_way=opt.shortcut_way)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)
    #################打印model_list
    '''AWEIGHT = torch.load(weights, map_location=device)['model']
    for k,v in AWEIGHT.items():
        print(k)'''

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, is_gray_scale=opt.gray_scale, rect=opt.rect)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        if opt.maxabsscaler:
            # 输出原始图片
            if opt.quantizer_output == True:
                if not os.path.isdir('./quantizer_output/'):
                    os.makedirs('./quantizer_output/')
                ori_img = copy.deepcopy(img)
                ori_img_input = np.array(ori_img.cpu()).reshape(1, -1)
                np.savetxt('./quantizer_output/img_input.txt', ori_img_input, delimiter='\n')
                ori_img_input = ori_img_input.astype(np.int8)
                writer = open('./quantizer_output/img_bin', "wb")
                writer.write(ori_img_input)
                writer.close()

                val_img = copy.deepcopy(img)
                val_img = val_img - 128
            img /= 256
            img = img * 2 - 1
            # 输出第一层的要送入卷积的量化数据
            if opt.quantizer_output == True:
                if not os.path.isdir('./quantizer_output/'):
                    os.makedirs('./quantizer_output/')
                q_img_input = copy.deepcopy(img)
                q_img_input = q_img_input * (2 ** (opt.a_bit - 1))

                # 软硬件处理方式对比
                delt = val_img - q_img_input
                delt = np.array(delt.cpu()).reshape(1, -1)
                delt_count = [np.sum(abs(delt) > 0)]
                np.savetxt(('./quantizer_output/not0_count.txt'), delt_count)

                q_img_input = np.array(q_img_input.cpu()).reshape(1, -1)
                np.savetxt('./quantizer_output/q_img_input.txt', q_img_input, delimiter='\n')
                q_img_input = q_img_input.astype(np.int8)
                writer = open('./quantizer_output/q_img_bin', "wb")
                writer.write(q_img_input)
                writer.close()
        else:
            img /= 256.0  # 0 - 255 to 0.0 - 1.0
        if opt.quantized != -1:
            if opt.a_bit == 16:
                img = img * (2 ** 14)
                sign = torch.sign(img)
                img = sign * torch.floor(torch.abs(img) + 0.5)
                img = img / (2 ** 14)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--rect', action='store_true', help='rectangular detecting')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--quantized', type=int, default=-1, help='quantization way')
    parser.add_argument('--shortcut_way', type=int, default=1, help='--shortcut quantization way')
    parser.add_argument('--a_bit', type=int, default=8, help='a-bit')
    parser.add_argument('--w_bit', type=int, default=8, help='w-bit')
    parser.add_argument('--FPGA', action='store_true', help='FPGA')
    parser.add_argument('--quantizer_output', action='store_true', help='quantizer output')
    parser.add_argument('--layer_idx', type=int, default=-1, help='output')
    parser.add_argument('--reorder', action='store_true', help='reorder')
    parser.add_argument('--TN', type=int, default=32, help='TN')
    parser.add_argument('--TM', type=int, default=32, help='TM')
    parser.add_argument('--gray-scale', action='store_true', help='gray scale trainning')
    parser.add_argument('--maxabsscaler', '-mas', action='store_true', help='Standarize input to (-1,1)')
    opt = parser.parse_args()
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.names = list(glob.iglob('./**/' + opt.names, recursive=True))[0]  # find file
    print(opt)

    with torch.no_grad():
        detect()

        if opt.quantizer_output == True and opt.layer_idx == -1:
            Val_upsample.Val_upsample(opt.cfg, opt.TN)
