# YOLOv3-ModelCompression-MultidatasetTraining

本项目包含三部分内容：

1、提供多个主流目标检测数据集的预处理后文件及训练方法。

2、提供包括剪植，量化，知识蒸馏的主流模型压缩算法实现。

3、提供多backbone训练目前包括Darknet-YOLOv3，Tiny-YOLOv3，Mobilenetv3-YOLOv3。

其中：

源码使用Pytorch实现，以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3源码仓库。基于BN层的剪植方法由[coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning)提供，感谢学长在模型压缩领域的探索。

# 最近更新
 2020年1月4日 提供Visdrone数据集剪裁后的下载链接和训练方法。
 
 2020年1月19日 提供Dior，Bdd100k，visdrone训练完成，并完成转化的.weights文件。
 
 2020年3月1日 实现基于mobilenetv3 backbone的YOLOv3。
 
 2020年4月7日 实现基于mobilenetv3的两种backbone模型，YOLOv3-mobilenet和YOLOv3tiny-mobilene-small
 ，提供预训练模型，将正常剪植算法扩展到基于mobilenet的两个模型和YOLOv3tiny模型，删除tiny剪植。

 2020年4月27日 更新mobilenetv3的模型预训练，添加了层剪植方法，方法来自于[tanluren/yolov3-channel-and-layer-pruning/yolov3](https://github.com/tanluren/yolov3-channel-and-layer-pruning)，
 感谢大佬的分享。
 
 2020年5月22日 更新了[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3源码仓库的最新优化，更新YOLOv4网络结构和权重文件。

 2020年5月22日 更新了8位定点量化方法，修复一些bug。
 
 2020年7月12日 修复了YOLOv3-mobilenet剪植后map归0的问题，详见issue#41。

 2020年7月14日 更新mobilenet支持基于shortcut的两种极限剪植方法和depthwise卷积的bn融合方法。
 
 2020年9月30日 更新BN融合的边融合边训练方法，减少BN融合所带来的精度损失，更新针对FPGA的pow(2)量化方法，详见量化篇。
# 环境部署
1.由于采用[ultralytics/yolov3](https://github.com/ultralytics/yolov3)的YOLO实现，环境搭建详见[ultralytics/yolov3](https://github.com/ultralytics/yolov3)。这里简要说明：

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

可直接`pip3 install -U -r requirements.txt`搭建环境，或根据该.txt文件使用conda搭建。

# 目前支持功能

|<center>功能</center>|<center></center>|
| --- |--- |
|<center>多种主干网络</center>|<center>√</center>  |
|<center>多种数据集</center>|<center>√</center>  |
|<center>剪植</center>|<center>√</center>  |
|<center>量化</center>|<center>√</center>  |
|<center>知识蒸馏</center>|<center>√</center>  |

# 可用指令

`python3 train.py --data ... --cfg ... `为训练模型指令，使用coco预训练模型时需要-pt指令。

`python3 test.py --data ... --cfg ... ` 为mAP测试指令。

`python3 detect.py --data ... --cfg ... --source ...`为推理检测指令，source默认地址为data/samples,输出结果保存在output文件中，检测资源可以为图片，视频等。

# 一、多数据集训练
本项目提供针对YOLOv3仓库的预处理数据集，配置文件(.cfg)，数据集索引文件(.data)，数据集类别文件(.names)以及使用k-means算法重新聚类的anchor box尺寸(包含用于yolov3的9框和tiny-yolov3的6框)。

mAP统计

|<center>数据集</center>|<center>YOLOv3-640</center>|<center>YOLOv4-640</center>|<center>YOLOv3-mobilenet-640</center>|
| --- |--- |--- |--- |
|<center>Dior遥感数据集</center>|<center>0.749</center>|
|<center>bdd100k自动驾驶数据集</center>|<center>0.543</center>|
|<center>visdrone无人机航拍数据集</center>|<center>0.311</center>|<center>0.383</center>|<center>0.348</center>|


下载地址如下，下载并解压后将文件夹拷贝至data目录下即可使用。

- [COCO2017](https://pan.baidu.com/s/1KysFL6AmdbCBq4tHDebqlw)
  
  提取码：hjln

- [COCO2014](https://pan.baidu.com/s/1EoXOR77yEVokqPCaxg8QGg)
  
  提取码：rhqx

- [COCO权重文件](https://pan.baidu.com/s/1JZylwRQIgAd389oWUu0djg)

  提取码：k8ms
  
训练指令

```bash
python3 train.py --data data/coco2017.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3.cfg --img-size ... --epochs ...
```


- [Dior遥感数据集](https://pan.baidu.com/s/1z0IQPBN16I-EctjwN9Idyg)
  
  提取码：vnuq

- [Dior权重文件](https://pan.baidu.com/s/12lYOgBAo1R5VkOZqDqCFJQ)

  提取码：l8wz
  
训练指令

```bash
python3 train.py --data data/dior.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-onDIOR.cfg --img-size ... --epochs ...
```


- [bdd100k无人驾驶数据集](https://pan.baidu.com/s/157Md2qeFgmcOv5UmnIGI_g)
  
  提取码：8duw
  
- [bdd100k权重文件](https://pan.baidu.com/s/1wWiHlLxIaK_WHy_mG2wmAA)

  提取码：xeqo
  
训练指令

```bash
python3 train.py --data data/bdd100k.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-bdd100k.cfg --img-size ... --epochs ...
```

- [visdrone数据集](https://pan.baidu.com/s/1CPGmS3tLI7my4_m7qDhB4Q)
  
  提取码：dy4c
  
- [YOLOv3-visdrone权重文件](https://pan.baidu.com/s/1N4qDP3b0tt8TIWuTFefDEw)

  提取码：87lf

- [YOLOv4-visdrone权重文件](https://pan.baidu.com/s/1zOFyt_AFiNk0fAFa8yE9RQ)

  提取码：xblu
  
 - [YOLOv3-mobilenet-visdrone权重文件](https://pan.baidu.com/s/1BHC8b6xHmTuN8h74QJFt1g)

  提取码：fb6y

训练指令

```bash
python train.py --data data/visdrone.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-visdrone.cfg  --img-size ... --epochs ...
```

- [oxfordhand数据集](https://pan.baidu.com/s/1JL4gFGh-W_gYEEsiIQssZw)
  
  提取码：3du4

训练指令

```bash
python train.py --data data/oxfordhand.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-visdrone.cfg  --img-size ... --epochs ...
```

## 1、Dior数据集
DIRO数据集是地球观测社区中最大、最多样化和公开可用的目标检测数据集之一。其中船舶和车辆的实例数较高，在小型实例和大型实例之间实现了良好的平衡。图片采集自Google Earth。

[数据集详细介绍](https://cloud.tencent.com/developer/article/1509762)

### 检测效果
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/2.jpg)
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/3.jpg)

## 2、bdd100k数据集
bdd100是一个大规模、多样化的驾驶视频数据集，共包含十万个视频。每个视频大约40秒长，研究者为所有10万个关键帧中常出现在道路上的对象标记了边界框。数据集涵盖了不同的天气条件，包括晴天、阴天和雨天、以及白天和晚上的不同时间。

[官网](http://bair.berkeley.edu/blog/2018/05/30/bdd/)

[原数据集下载](http://bdd-data.berkeley.edu)

[论文](https://arxiv.org/abs/1805.04687)

### 检测效果
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/1.jpg)

## 3、Visdrone数据集
VisDrone2019数据集由中国天津大学机器学习和数据挖掘实验室的AISKYEYE团队收集。基准数据集包含288个视频片段，由261,908个帧和10,209个帧组成静态图像，由各种安装在无人机上的摄像头捕获，涵盖了广泛的方面，包括位置（从中国相距数千公里的14个不同城市中拍摄），环境（城市和乡村），物体（行人，车辆，自行车等）和密度（稀疏和拥挤的场景）。该数据集是在各种情况下以及在各种天气和光照条件下使用各种无人机平台（即具有不同模型的无人机）收集的。这些框架使用超过260 万个边界框手动标注，这些边界框是人们经常感兴趣的目标，例如行人，汽车，自行车和三轮车。还提供了一些重要属性，包括场景可见性，对象类别和遮挡，以提高数据利用率.

[官网](http://www.aiskyeye.com/)

### 检测效果YOLOv3
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/4.jpg)

### 检测效果YOLOv4
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/5.jpg)
![检测效果](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/6.png)


# 二、多种网络结构
在mobilenetv3基础上设计了两种网络结构

|结构名称 |<center>backbone</center>|<center>后处理</center> |<center>总参数</center> |<center>GFLOPS</center> |<center>mAP0.5</center> |<center>mAP0.5:0.95</center> |<center>speed(inference/NMS/total)</center> |<center>FPS</center> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|YOLOv3                      |38.74M  |20.39M  |59.13M  |117.3   |0.580  |0.340  |12.3/1.7/14.0 ms|71.4fps  |
|YOLOv3tiny                  |6.00M   |2.45M   |8.45M   |9.9     |0.347  |0.168  |3.5/1.8/5.3 ms  |188.7fps |
|YOLOv3-mobilenetv3          |2.84M   |20.25M  |23.09M  |32.2    |0.547  |0.346  |7.9/1.8/9.7 ms  |103.1fps |
|YOLOv3tiny-mobilenetv3-small|0.92M   |2.00M   |2.92M   |2.9     |0.379  |0.214  |5.2/1.9/7.1 ms  |140.8fps |
|YOLOv4                      |-       |-       |61.35M  |107.1   |0.650  |0.438  |13.5/1.8/15.3 ms|65.4fps  |
|YOLOv4-tiny                 |-       |-       |5.78M   |12.3    |0.435  |0.225  |4.1/1.7/5.8 ms  |172.4fps |

注：

1、YOLOv3,YOLOv3tiny和YOLOv4是在coco2014上训练和测试的，YOLOv3-mobilenetv3和YOLOv3tiny-mobilenetv3-small是在coco2017上训练和测试的。
    
2、推理速度在GTX2080ti*4上测试,输入图片尺寸608。
    
3、训练测试集与训练集应当相匹配，不匹配会造成map虚高的问题。原因参照[issue](https://github.com/ultralytics/yolov3/issues/970)

## 训练指令
1、YOLOv3
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3-608.weights --cfg cfg/yolov3/yolov3.cfg --img_size ...
```

权重文件下载
- [COCO预训练权重文件](https://pan.baidu.com/s/1JZylwRQIgAd389oWUu0djg)

  提取码：k8ms

2、YOLOv3tiny
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3tiny.weights --cfg cfg/yolov3tiny/yolov3-tiny.cfg --img_size ...
```

- [COCO预训练权重文件](https://pan.baidu.com/s/1iWGxdjR3TWxEe37__msyRA)

  提取码：udfe
  
3、YOLOv3tiny-mobilenet-small
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3tiny-mobilenet-small.weights --cfg cfg/yolov3tiny-mobilenet-small/yolov3tiny-mobilenet-small-coco.cfg --img_size ...
```

- [COCO预训练权重文件](https://pan.baidu.com/s/1mSFjWLU91H2OhNemsAeiiQ)

  提取码：pxz4

4、YOLOv3-mobilenet
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3-mobilenet.weights --cfg cfg/yolov3-mobilenet/yolov3-mobilenet-coco.cfg --img_size ...
```

- [COCO预训练权重文件](https://pan.baidu.com/s/1EI2Xh1i18CRLoZo_P3NVHw)

  提取码：3vm8

5、YOLOv4
```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov4.weights --cfg cfg/yolov4/yolov4.cfg --img_size ...
```

- [COCO预训练权重文件](https://pan.baidu.com/s/1jAGNNC19oQhAIgBfUrkzmQ)

  提取码：njdg
  
# 三、模型压缩

## 1、剪植

### 剪植特点
|<center>剪枝方案</center> |<center>优点</center>|<center>缺点</center> |
| --- | --- | --- |
|正常剪枝   |不对shortcut剪枝，拥有可观且稳定的压缩率，无需微调，支持tiny-yolov3和mobilenet系列。  |压缩率达不到极致。  |
|极限剪枝   |极高的压缩率。  |需要微调。  |
|极限剪枝2  |采用shortcut融合的方法提升剪植精度。  |针对shortcut最优的方法。|
|规整剪枝   |专为硬件部署设计，剪枝后filter个数均为8的倍数，无需微调，支持tiny-yolov3和mobilenet系列。 |为规整牺牲了部分压缩率。 |
|层剪枝     |以ResBlock为基本单位剪植，利于硬件部署。 |但是只能剪backbone，剪植率有限。 |
|层通道剪植 |先进行通道剪植再进行层剪植，剪植率非常高。 |可能会影响精度。 |

### 步骤

1.正常训练

```bash
python3 train.py --data ... -pt --batch-size ... --weights ... --cfg ...
```

2.稀疏化训练

`-sr`开启稀疏化，`--s`指定稀疏因子大小，`--prune`指定稀疏类型。

其中：

`--prune 0`为正常剪枝和规整剪枝的稀疏化

`--prune 1`为极限剪枝的稀疏化

`--prune 2`为层剪植稀疏化

指令范例：

```bash
python3 train.py --data ... -pt --batch-size 32  --weights ... --cfg ... -sr --s 0.001 --prune 0 
```

3.模型剪枝

- 正常剪枝
```bash
python3 normal_prune.py --cfg ... --data ... --weights ... --percent ...
```
- 规整剪枝
```bash
python3 regular_prune.py --cfg ... --data ... --weights ... --percent ...
```
- 极限剪枝
```bash
python3 shortcut_prune.py --cfg ... --data ... --weights ... --percent ...
```

- 极限剪枝2
```bash
python3 slim_prune.py --cfg ... --data ... --weights ... --percent ...
```

- 层剪植
```bash
python3 layer_prune.py --cfg ... --data ... --weights ... --shortcut ...
```

- 层剪植+通道剪植
```bash
python3 layer_channel_prune.py --cfg ... --data ... --weights ... --shortcut ... --percent ...
```


需要注意的是，这里需要在.py文件内，将opt内的cfg和weights变量指向第2步稀疏化后生成的cfg文件和weights文件。
此外，可通过增大代码中percent的值来获得更大的压缩率。（若稀疏化不到位，且percent值过大，程序会报错。）

### 剪植实验
1、正常剪植 
oxfordhand数据集，img_size = 608，在GTX2080Ti*4上计算推理时间

|<center>模型</center> |<center>剪植前参数量</center> |<center>剪植前mAP</center>|<center>剪植前推理时间</center>|<center>剪植率</center> |<center>剪植后参数量</center> |<center>剪植后mAP</center> |<center>剪植后推理时间</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(不微调)           |58.67M   |0.806   |0.1139s   |0.8    |10.32M |0.802 |0.0844s |
|yolov3-mobilenet(微调)   |22.75M   |0.812   |0.0345s   |0.97   |2.72M  |0.795 |0.0211s |
|yolov3tiny(微调)         |8.27M    |0.708   |0.0144s   |0.5    |1.13M  |0.641 |0.0116s |

2、规则剪植
oxfordhand数据集，img_size = 608，在GTX2080Ti*4上计算推理时间

|<center>模型</center> |<center>剪植前参数量</center> |<center>剪植前mAP</center>|<center>剪植前推理时间</center>|<center>剪植率</center> |<center>剪植后参数量</center> |<center>剪植后mAP</center> |<center>剪植后推理时间</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(不微调)           |58.67M   |0.806   |0.1139s   |0.8    |12.15M |0.805 |0.0874s |
|yolov3-mobilenet(微调)   |22.75M   |0.812   |0.0345s   |0.97   |2.75M  |0.803 |0.0208s |
|yolov3tiny(微调)         |8.27M    |0.708   |0.0144s   |0.5    |1.82M  |0.703 |0.0122s |

## 2、量化
### 定点量化
`--quantized 2` 使用Dorefa8位定点量化方法

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 2
```

`--quantized 1` 使用Google白皮书8位定点量化方法

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 1
```

加入`--BN_Flod`使用BN融合训练，加入`--FPGA`使用针对FPGA的pow(2)量化。
### 量化实验
oxfordhand数据集，使用yolov3网络，640image-size
|<center>方法</center> |<center>teacher模型mAP</center> |
| --- | --- |
|Baseline                     |0.847    |
|Google8bit                   |0.851    |
|Google8bit + BN融合训练       |0.851    |
|Google8bit + BN融合训练 + FPGA|0.852    |
|Google4bit + BN融合训练 + FPGA|0.842    |
## 3、知识蒸馏

### 蒸馏方法
蒸馏方法采用基于Hinton于2015年提出的基本蒸馏方法，并结合检测网络做了部分改进。

Distilling the Knowledge in a Neural Network
[参考论文](https://arxiv.org/abs/1503.02531)

`--t_cfg --t_weights --KDstr` 在命令中加入这两个选项即可以开始蒸馏训练。

其中：

`--t_cfg`表示教师网络配置文件。

`--t_weights`表示教师网络权重文件。

`--KDstr`表示使用的蒸馏策略。

    `--KDstr 1` 直接在tencher网络的输出和student网络的输出求KLloss并加入到整体的loss中
    `--KDstr 2` 对boxloss和classloss有所区分，student不直接向teacher学习。student，teacher和GT分别求l2距离，当student大于teacher时附加一项student和gt的loss。
    `--KDstr 3` 对boxloss和classloss有所区分，student直接向teacher学习。
    `--KDstr 4` 将KDloss分为三项，boxloss，classloss和featureloss。
    `--KDstr 5` 在KDstr 4的基础上，feature中加入Fine-grain-mask
蒸馏指令范例：

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --t_cfg ... --t_weights ...
```

通常以压缩前模型为teacher模型，压缩后模型为student模型进行蒸馏训练，提高学生网络的mAP。

### 蒸馏实验
oxfordhand数据集，使用yolov3tiny作为teacher网络，normal剪植后的yolov3tiny作为学生网络

|<center>teacher模型</center> |<center>teacher模型mAP</center> |<center>student模型</center>|<center>直接微调</center>|<center>KDstr 1</center> |<center>KDstr 2</center> |<center>KDstr 3</center>  |<center>KDstr 4(L1)</center> |<center>KDstr 5(L1)</center> |
| --- | --- | --- | --- | --- | --- | --- |--- |--- |
|yolov3tiny608   |0.708    |normal剪植yolov3tiny608    |0.658     |0.666    |0.661  |0.672   |0.673   |0.674   |