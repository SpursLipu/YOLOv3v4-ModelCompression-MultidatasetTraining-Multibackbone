import numpy as np
import torch
import torch.nn.functional as F
import os

if not os.path.isdir('./validation'):
    os.makedirs('./validation')

###55层和67层是上采样层
layer_idx = 55

activation_input = np.loadtxt('./quantizer_output/q_activation_reorder/a_reorder_00%d_conv.txt'%(layer_idx-1))

input_scale = np.loadtxt('./quantizer_output/a_scale_out/a_scale_00%d_conv.txt'%(layer_idx-1))

if layer_idx == 55:
    activation_input = torch.from_numpy(activation_input).view(1, 256, 24, 18)
elif layer_idx == 67:
    activation_input = torch.from_numpy(activation_input).view(1, 128, 48, 36)

input_scale = torch.from_numpy(input_scale)


#stride为上采样的倍数
stride = 2
temp_out = F.upsample(input=activation_input,scale_factor=stride)

#保存txt文件
val_results = np.array(temp_out.cpu()).reshape(1, -1)
np.savetxt(('./validation/%d_upsample_output.txt'%layer_idx), val_results,delimiter='\n')

output_scale = input_scale
output_scale = np.array(output_scale.cpu()).reshape(1, -1)
np.savetxt(('./validation/%d_upsample_scale.txt'%layer_idx), output_scale,delimiter='\n')

###保存二进制文件
activation_flat = val_results.astype(np.int8)
writer = open('./validation/%d_upsample_q_bin'%layer_idx, "wb")
writer.write(activation_flat)
writer.close()