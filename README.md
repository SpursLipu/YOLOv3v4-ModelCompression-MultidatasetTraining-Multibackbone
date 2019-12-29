# YOLOv3-ModelCompression-MultidatasetTraining

本项目包含两部分内容：

1、提供多个主流目标检测数据集的预处理后文件及训练方法。

2、提供包括剪植，量化，知识蒸馏的主流模型压缩算法实现。

其中：

源码使用Pytorch实现，以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3源码仓库。基于BN层的剪植方法由[coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning)提供，感谢学长在模型压缩领域的探索。

# 环境部署
1.由于采用[ultralytics/yolov3](https://github.com/ultralytics/yolov3)的YOLO实现，环境搭建详见[ultralytics/yolov3](https://github.com/ultralytics/yolov3)。这里简要说明：

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

可直接`pip3 install -U -r requirements.txt`搭建环境，或根据该.txt文件使用conda搭建。

# 目前支持功能

|<center>功能</center>|<center>多卡</center>|
| --- |--- |
|<center>正常训练</center>|<center>√</center>|
|<center>tiny训练</center>|<center>√</center>|
|<center>Dior数据集训练</center>|<center>√</center>|
|<center>bdd100k数据集训练</center>|<center>√</center>|
|<center>稀疏化训练</center>|<center>√</center>  |
|<center>正常剪枝</center>|<center>√</center>|
|<center>规整剪枝</center>|<center>√</center>  |
|<center>极限剪枝(shortcut)</center>|<center>√</center> |
|<center>Tiny剪枝</center>|<center>√</center>  |
|<center>BNN量化</center>|<center>√</center>  |
|<center>BWN量化</center>|<center>√</center>  |
|<center>stage-wise 逐层量化</center>|<center>√</center>  |

# 功能支持

# 多数据集训练

本项目提供针对YOLOv3仓库的预处理数据集，配置文件(.cfg)，数据集索引文件(.data)，数据集类别文件(.names)以及使用k-means算法重新聚类的anchor box尺寸(包含用于yolov3的9框和tiny-yolov3的6框)。

下载地址如下，下载并解压后将文件夹拷贝至data目录下即可使用。

- [Dior遥感数据集](https://pan.baidu.com/s/1z0IQPBN16I-EctjwN9Idyg)
  
  提取码：vnuq

  训练指令

```bash
python3 train.py --data cfg/dior.data --batch-size 30 --weights weights/yolov3.weights --cfg cfg/yolov3-onDIOR.cfg --img-size 608 --epochs 200
```


- [bdd100k无人驾驶数据集](https://pan.baidu.com/s/157Md2qeFgmcOv5UmnIGI_g)
  
  提取码：8duw

  训练指令

```bash
python3 train.py --data cfg/bdd100k.data --batch-size 20 --weights weights/yolov3.weights --cfg cfg/yolov3-bdd100k.cfg --img-size 608 --epochs 200
```
  
## Dior数据集
DIRO数据集是地球观测社区中最大、最多样化和公开可用的目标检测数据集之一。其中船舶和车辆的实例数较高，在小型实例和大型实例之间实现了良好的平衡。图片采集自Google Earth。

[数据集详细介绍](https://cloud.tencent.com/developer/article/1509762)

### 检测效果
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/2.jpg)
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/3.jpg)

## bdd100k数据集
Bdd100是一个大规模、多样化的驾驶视频数据集，共包含十万个视频。每个视频大约40秒长，研究者为所有10万个关键帧中常出现在道路上的对象标记了边界框。数据集涵盖了不同的天气条件，包括晴天、阴天和雨天、以及白天和晚上的不同时间。

[官网](http://bair.berkeley.edu/blog/2018/05/30/bdd/)

[原数据集下载](http://bdd-data.berkeley.edu)

[论文](https://arxiv.org/abs/1805.04687)

### 检测效果
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/1.jpg)

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
### 量化方法
`--quantized` 表示选取量化方法，默认值为-1，表示不采用任何量化方法。

`--quantized 0` 表示使用BNN量化方法。

`--quantized 1` 表示选使用BWN量化方法

### stage-wise 训练策略
`--qlayers`可以用于选取Darknet中的量化区间，默认为自深层到浅层, 默认值为-1表示无量化层，有效范围为0-74，取0时表示量化所有层，取74时表示无量化层，大于74则无意义。

![Darknet](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/Darknet.png)
如：

`--qlayers 63` 表示量化Darknet主体网络中最后四个重复的残差块。

`--qlayers 38` 表示量化Darknet主体网络中从倒数第二八个重复的残差块开始，量化到Darknet主体网络结束。

以此类推，量化时可根据具体情况选择何是的量化层数，以及量化进度，推荐`--qlayers`值自74逐渐下降。

量化指令范例：

```bash
python train.py --data cfg/bdd100k.data --batch-size 20 --weights weights/best.pt --cfg cfg/yolov3-bdd100k.cfg --img-size 608 --epochs 200 --quantized 1 --qlayers 72

```

## 知识蒸馏
