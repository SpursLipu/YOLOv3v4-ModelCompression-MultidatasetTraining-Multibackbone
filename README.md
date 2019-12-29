# YOLOv3-ModelCompression-MultidatasetTraining
YOLOv3 ModelCompression MultidatasetTraining

本项目以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3的Pytorch实现，剪植方法由[coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning)实现，感谢学长在在模型压缩领域的探索。


# 环境部署
1.由于采用[ultralytics/yolov3](https://github.com/ultralytics/yolov3)的YOLO实现，环境搭建见[ultralytics/yolov3](https://github.com/ultralytics/yolov3)。这里重复介绍一下：

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

可直接`pip3 install -U -r requirements.txt`搭建环境，或根据该.txt文件使用conda搭建。

# 项目特点

1.采用的YOLO-v3实现较为准确，mAP相对较高。

2.提供对YOLOv3及Tiny的多种剪枝版本，以适应不同的需求。

3.剪枝后保存为.weights格式，可在任何框架下继续训练、推理，或以图像视频展示。

4.提供多种精度的量化

5.目前支持情况

|<center>功能</center>|<center>单卡</center>|<center>多卡</center>|
| --- | --- | --- |
|<center>正常训练</center>|<center>√</center>|<center>√</center>|
|<center>Dior数据集训练</center>|<center>√</center>|<center>√</center>|
|<center>bdd100k数据集训练</center>|<center>√</center>|<center>√</center>|
|<center>稀疏化训练</center>|<center>√</center>|<center>√</center>  |
|<center>正常剪枝</center>|<center>√</center>|<center>√</center>|
|<center>规整剪枝</center>  | <center>√</center> |<center>√</center>  |
|<center>极限剪枝(shortcut)</center>  | <center>√</center> | <center>√</center> |
|<center>Tiny剪枝</center>|<center>√</center>|<center>√</center>  |
|<center>BNN量化</center>|<center>√</center>|<center>√</center>  |
|<center>BWN量化</center>|<center>√</center>|<center>√</center>  |
|<center>stage-wise 逐层量化</center>|<center>√</center>|<center>√</center>  |

# 推理展示

这里，我们不仅可以使用原始的YOLOV3用来推理展示，还可使用我们剪枝后的模型来推理展示。（修改cfg，weights的指向即可）

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

# 多数据集训练

本项目已完成并开放针对YOLOv3仓库的数据集预处理，并提供配置文件(.cfg)，数据集所以文件(.data)，数据集类别文件(.names)以及使用k-means算法重新聚类的anchor box尺寸(包含用于yolov3的9框和tiny-yolov3的6框)。

下载地址如下，下载并解压后将文件夹拷贝至data目录下即可使用。

- [Dior遥感数据集](https://pan.baidu.com/s/1z0IQPBN16I-EctjwN9Idyg)
  
  提取码：vnuq
- [bdd100k无人驾驶数据集](https://pan.baidu.com/s/157Md2qeFgmcOv5UmnIGI_g)
  
  提取码：8duw
  
## Dior数据集
DIRO数据集是地球观测社区中最大、最多样化和公开可用的目标检测数据集之一。其中船舶和车辆的实例数较高，在小型实例和大型实例之间实现了良好的平衡。图片采集自Google Earth。

[数据集详细介绍](https://cloud.tencent.com/developer/article/1509762)

## bdd100k数据集
Bdd100是一个大规模、多样化的驾驶视频数据集，共包含十万个视频。每个视频大约40秒长，研究者为所有10万个关键帧中常出现在道路上的对象标记了边界框。数据集涵盖了不同的天气条件，包括晴天、阴天和雨天、以及白天和晚上的不同时间。

[官网](http://bair.berkeley.edu/blog/2018/05/30/bdd/)

[原数据集下载](http://bdd-data.berkeley.edu)

[论文](https://arxiv.org/abs/1805.04687)

# 模型压缩

## 剪植

### 剪植特点
|剪枝方案 |<center>优点</center>|<center>缺点</center> |
| --- | --- | --- |
|正常剪枝 |不对shortcut剪枝，拥有可观且稳定的压缩率，无需微调。  |压缩率达不到极致。  |
|极限剪枝 |极高的压缩率。  |需要微调。  |
|规整剪枝 |专为硬件部署设计，剪枝后filter个数均为8的倍数，无需微调。 | 为规整牺牲了部分压缩率。 |
|Tiny剪枝 |稳定的压缩率。  |由于Tiny本来已很小，压缩率中规中矩。  |


### 步骤

1.正常训练

```bash
python3 train.py --data data/oxfordhand.data --batch-size 32 --accumulate 1 --weights weights/yolov3.weights --cfg cfg/yolov3-hand.cfg
```

2.稀疏化训练

`-sr`开启稀疏化，`--s`指定稀疏因子大小，`--prune`指定稀疏类型。

其中：

`--prune 0`为正常剪枝和规整剪枝的稀疏化

`--prune 1`为极限剪枝的稀疏化

`--prune 2`为Tiny剪枝的稀疏化

```bash
python3 train.py --data data/oxfordhand.data --batch-size 32 --accumulate 1 --weights weights/yolov3.weights --cfg cfg/yolov3-hand.cfg -sr --s 0.001 --prune 0 
```

3.模型剪枝

- 正常剪枝
```bash
python3 normal_prune.py
```
- 规整剪枝
```bash
python3 regular_prune.py
```
- 极限剪枝
```bash
python3 shortcut_prune.py
```
- Tiny剪枝
```bash
python3 prune_tiny_yolo.py
```
需要注意的是，这里需要在.py文件内，将opt内的cfg和weights变量指向第2步稀疏化后生成的cfg文件和weights文件。
此外，可通过增大代码中percent的值来获得更大的压缩率。（若稀疏化不到位，且percent值过大，程序会报错。）

## 量化


