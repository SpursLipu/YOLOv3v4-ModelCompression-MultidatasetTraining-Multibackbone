import numpy as np
import torch
import torch.nn.functional as F

activation_input = np.loadtxt()
activation_output = np.loadtxt()
input_scale = np.loadtxt()
output_scale = np.loadtxt()
w_para = np.loadtxt()
b_para = np.loadtxt()
weight_scale = np.loadtxt()
bias_scale = loadtxt()
activation_input = torch.from_numpy(activation_input).view(1,64,72,96)
activation_output = torch.from_numpy(activation_output).view(1,64,72,96)
input_scale = torch.from_numpy(input_scale)
output_scale = torch.from_numpy(output_scale)
w_para = torch.from_numpy(w_para).view(64,64,3,3)
b_para = torch.from_numpy(b_para).view(64)
weight_scale = torch.from_numpy(weight_scale)
bias_scale = torch.from_numpy(bias_scale)

tmp_out = F.conv2d(
    input = activation_input * (2**input_scale),
    weight = w_para * (2**weight_scale),
    bias = b_para * (2**bias_scale),
    stride =1,
    padding =1,
)
tmp_out = F.leaky_relu(tmp_out,0.125,implace = True)

delt = tmp_out/(2 ** output_scale) - activation_output
max = torch.max(tmp_out / (2** output_scale))
min = torch.min(tmp_out / (2** output_scale))
shape_fmout = tmp_out.shape
delt_not_0 = (torch.abs(delt) != 0)
sum = (delt_not_0.shape[0]*delt_not_0.shape[1]*delt_not_0.shape[2]*delt_not_0.shape[3])
percent = count / sum
delt_overflow_idx = torch.nonzero(delt_gt_1)

if delt_overflow_idx != None:
    path = ''
    dump_file_delt_txt = path + "_delt.txt"
    np.savetxt(dump_file_delt_txt,delt_overflow_idx,fmt = "%f\n",delimiter=',')