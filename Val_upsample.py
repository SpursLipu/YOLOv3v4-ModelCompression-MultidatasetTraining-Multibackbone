import numpy as np
import torch
import torch.nn.functional as F
import os
from utils.parse_config import *


# cfg = './cfg/prune_regular_0.8_keep_0.01_10_shortcut_yolov3-ship.cfg'
def Val_upsample(cfg):

    # if not os.path.isdir('./validation'):
    #     os.makedirs('./validation')

    module_defs = parse_model_cfg(cfg)
    #_ = module_defs.pop(0)  # cfg training hyperparams (unused)
    upsample_times = 0  # 上采样次数（第几次上采样）
    for i, mdef in enumerate(module_defs):
        if mdef['type'] == 'net':
            width = mdef['width']
            height = mdef['height']
            channels = mdef['channels']
        elif mdef['type'] == 'upsample':

            upsample_times = upsample_times + 1

            layer_idx = i - 1

            activation_input = np.loadtxt('./quantizer_output/q_activation_reorder/a_reorder_00%d_conv.txt'%(layer_idx-1))

            input_scale = np.loadtxt('./quantizer_output/a_scale_out/a_scale_00%d_conv.txt'%(layer_idx-1))

            Up_channels = int(256 / upsample_times)
            Up_width = int((width * upsample_times) /32)
            Up_height = int((height * upsample_times) /32)
            activation_input = torch.from_numpy(activation_input).view(1, Up_channels, Up_width, Up_height)


            input_scale = torch.from_numpy(input_scale)


            #stride为上采样的倍数
            stride = 2
            temp_out = F.upsample(input=activation_input,scale_factor=stride)

            #保存txt文件
            val_results = np.array(temp_out.cpu()).reshape(1, -1)
            np.savetxt(('./quantizer_output/q_activation_reorder/%d_upsample_output.txt'%layer_idx), val_results,delimiter='\n')

            output_scale = input_scale
            output_scale = np.array(output_scale.cpu()).reshape(1, -1)
            np.savetxt(('./quantizer_output/a_scale_out/%d_upsample_scale.txt'%layer_idx), output_scale,delimiter='\n')

            ###保存二进制文件
            activation_flat = val_results.astype(np.int8)
            writer = open('./quantizer_output/q_activation_reorder/%d_upsample_q_bin'%layer_idx, "wb")
            writer.write(activation_flat)
            writer.close()

# Val_upsample(cfg)


# import argparse
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
#     opt = parser.parse_args()
#     opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file