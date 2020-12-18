# YOLOv3-ModelCompression-MultidatasetTraining

[README in chinese](https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone/blob/master/README_in_Chinese.md)

This project mainly include three parts.

1.Provides training methods for multiple mainstream object detection datasets(coco2017, coco2014, BDD100k, Visdrone,
Hand)

2.Provides a mainstream model compression algorithm including pruning, quantization, and knowledge distillation.

3.Provides multiple backbone for yolov3 including Darknet-YOLOv3，Tiny-YOLOv3，Mobilenetv3-YOLOv3

Source using Pytorch implementation to [ultralytics/yolov3](https://github.com/ultralytics/yolov3) for yolov3 source
code. Pruning method based on BN layer
by [coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning), thanks to both of you.

# Update

January 4, 2020. Provides download links and training methods to the Visdrone dataset.

January 19, 2020. Dior, Bdd100k and Visdrone training will be provided, as well as the converted weights file.

March 1, 2020. Provides Mobilenetv3 backbone.

April 7, 2020. Implement two models based on Mobilenetv3: Yolov3-Mobilenet, and Yolov3tin-Mobilene-small, provide
pre-training weights, extend the normal pruning methods to the two Mobilenet-based models.

April 27, 2020. Update mobilenetv3 pre-training weights, add a layer pruning method, methods from
the [tanluren/yolov3-channel-and-layer-pruning/yolov3](https://github.com/tanluren/yolov3-channel-and-layer-pruning),
Thanks for sharing.

May 22, 2020. Updated some new optimizations from [ultralytics/yolov3](https://github.com/ultralytics/yolov3), update
cfg file and weights of YOLOv4.

May 22, 2020. The 8-bit quantization method was updated and some bugs were fixed.

July 12, 2020. The problem of mAP returning to 0 after pruning in yolov3-mobilenet was fixed. See issue#41 for more
details.

September 30, 2020. The BN_Fold training method was updated to reduce the precision loss caused by BN fusion, and the
POW (2) quantization method targeted at FPGA was updated. See the quantization section for details.

# Requirements

Our project based on [ultralytics/yolov3](https://github.com/ultralytics/yolov3),
see [ultralytics/yolov3](https://github.com/ultralytics/yolov3) for details. Here is a brief explanation:

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

# Current support

|<center>Function</center>|<center></center>|
| --- |--- |
|<center>Multi-Backbone training</center>|<center>√</center>  |
|<center>Multi-Datasets</center>|<center>√</center>  |
|<center>Pruning</center>|<center>√</center>  |
|<center>Quantization</center>|<center>√</center>  |
|<center>Knowledge Distillation</center>|<center>√</center>  |

# Training

`python3 train.py --data ... --cfg ... `For training model command, the -pt command is required when using coco
pre-training model.

`python3 test.py --data ... --cfg ... ` For testing model command

`python3 detect.py --data ... --cfg ... --source ...` For detecting model command, the default address of source is
data/samples, the output result is saved in the /output, and the detection resource can be pictures and videos.

# Multi-Datasets

This project provides preprocessed datasets for the YOLOv3, configuration files (.cfg), dataset index files (.data),
dataset category files (.names), and anchor box sizes (including 9 boxes for YOLOv3 and 6 boxes for tiny- YOLOv3) that
are reclustered using the K-means algorithm.

mAP

|<center>Dataset</center>|<center>YOLOv3-640</center>|<center>YOLOv4-640</center>|<center>YOLOv3-mobilenet-640</center>|
| --- |--- |--- |--- |
|<center>Dior</center>|<center>0.749</center>|
|<center>bdd100k</center>|<center>0.543</center>|
|<center>visdrone</center>|<center>0.311</center>|<center>0.383</center>|<center>0.348</center>|

Datasets, download and unzip to /data.

- [COCO2017](https://pan.baidu.com/s/1KysFL6AmdbCBq4tHDebqlw)

  Extract code：hjln

- [COCO2014](https://pan.baidu.com/s/1EoXOR77yEVokqPCaxg8QGg)

  Extract code：rhqx

- [COCO weights](https://pan.baidu.com/s/1JZylwRQIgAd389oWUu0djg)

  Extract code：k8ms

Training command

```bash
python3 train.py --data data/coco2017.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3.cfg --img-size ... --epochs ...
```

- [Dior](https://pan.baidu.com/s/1z0IQPBN16I-EctjwN9Idyg)

  Extract code：vnuq

- [Dior weights](https://pan.baidu.com/s/12lYOgBAo1R5VkOZqDqCFJQ)

  Extract code：l8wz

Training command

```bash
python3 train.py --data data/dior.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-onDIOR.cfg --img-size ... --epochs ...
```

- [bdd100k](https://pan.baidu.com/s/157Md2qeFgmcOv5UmnIGI_g)

  Extract code：8duw

- [bdd100k weights](https://pan.baidu.com/s/1wWiHlLxIaK_WHy_mG2wmAA)

  Extract code：xeqo

Training command

```bash
python3 train.py --data data/bdd100k.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-bdd100k.cfg --img-size ... --epochs ...
```

- [visdrone](https://pan.baidu.com/s/1CPGmS3tLI7my4_m7qDhB4Q)

  Extract code：dy4c

- [YOLOv3-visdrone weights](https://pan.baidu.com/s/1N4qDP3b0tt8TIWuTFefDEw)

  Extract code：87lf

- [YOLOv4-visdrone weights](https://pan.baidu.com/s/1zOFyt_AFiNk0fAFa8yE9RQ)

  Extract code：xblu

- [YOLOv3-mobilenet-visdrone weights](https://pan.baidu.com/s/1BHC8b6xHmTuN8h74QJFt1g)

Extract code：fb6y

Training command

```bash
python train.py --data data/visdrone.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-visdrone.cfg  --img-size ... --epochs ...
```

- [oxfordhand](https://pan.baidu.com/s/1JL4gFGh-W_gYEEsiIQssZw)

  Extract code：3du4

Training command

```bash
python train.py --data data/oxfordhand.data --batch-size ... --weights weights/yolov3-608.weights -pt --cfg cfg/yolov3/yolov3-visdrone.cfg  --img-size ... --epochs ...
```

## 1.Dior

The DIRO dataset is one of the largest, most diverse, and publicly available object detection datasets in the Earth
observation community. Among them, the number of instances of ships and vehicles is high, which achieves a good balance
between small instances and large ones. The images were collected from Google Earth.

[Introduction](https://cloud.tencent.com/developer/article/1509762)

### Test results

![Test results](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/2.jpg)
![Test results](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/3.jpg)

## 2.bdd100k

Bdd100 is a large, diverse data set of driving videos containing 100,000 videos. Each video was about 40 seconds long,
and the researchers marked bounding boxes for all 100,000 key frames of objects that often appeared on the road. The
data set covers different weather conditions, including sunny, cloudy and rainy days, and different times of day and
night.

[Website](http://bair.berkeley.edu/blog/2018/05/30/bdd/)

[Download](http://bdd-data.berkeley.edu)

[Paper](https://arxiv.org/abs/1805.04687)

### Test results

![Test results](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/1.jpg)

## 3.Visdrone

The VisDrone2019 dataset was collected by AISKYEYE team at the Machine Learning and Data Mining Laboratory at Tianjin
University, China. Benchmark data set contains 288 video clips, and consists of 261908 frames and 10209 frames a static
image, by all sorts of installed on the unmanned aerial vehicle (uav) camera capture, covers a wide range of aspects,
including location (thousands of kilometers apart from China in 14 different cities), environment (city and country),
object (pedestrians, vehicles, bicycles, etc.) and density (sparse and crowded scenario). This data set was collected
using a variety of uav platforms (i.e., uAvs with different models) in a variety of situations and under various weather
and light conditions. These frames are manually marked with more than 2.6 million border frames, which are often targets
of interest, such as pedestrians, cars, bicycles and tricycles. Some important attributes are also provided, including
scene visibility, object categories, and occlusion, to improve data utilization.

[Website](http://www.aiskyeye.com/)

### Test results of YOLOv3

![Test results](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/4.jpg)

### Test results of YOLOv4

![Test results](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/5.jpg)
![Test results](https://github.com/SpursLipu/YOLOv3-ModelCompression-MultidatasetTraining/blob/master/image_in_readme/6.png)

# Multi-Backbone

Based on mobilenetv3, two network structures are designed.

|Structure |<center>backbone</center>|<center>Postprocessing</center> |<center>Parameters</center> |<center>GFLOPS</center> |<center>mAP0.5</center> |<center>mAP0.5:0.95</center> |<center>speed(inference/NMS/total)</center> |<center>FPS</center> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|YOLOv3                      |38.74M  |20.39M  |59.13M  |117.3   |0.580  |0.340  |12.3/1.7/14.0 ms|71.4fps  |
|YOLOv3tiny                  |6.00M   |2.45M   |8.45M   |9.9     |0.347  |0.168  |3.5/1.8/5.3 ms  |188.7fps |
|YOLOv3-mobilenetv3          |2.84M   |20.25M  |23.09M  |32.2    |0.547  |0.346  |7.9/1.8/9.7 ms  |103.1fps |
|YOLOv3tiny-mobilenetv3-small|0.92M   |2.00M   |2.92M   |2.9     |0.379  |0.214  |5.2/1.9/7.1 ms  |140.8fps |
|YOLOv4                      |-       |-       |61.35M  |107.1   |0.650  |0.438  |13.5/1.8/15.3 ms|65.4fps  |
|YOLOv4-tiny                 |-       |-       |5.78M   |12.3    |0.435  |0.225  |4.1/1.7/5.8 ms  |172.4fps |

1. YOLOv3,YOLOv3tiny and YOLOv4 were trained and tested on coco2014, and Yolov3-Mobilenetv3 and YOLOv3tiny
   Mobilenetv3-Small were trained and tested on coco2017.

2. The reasoning speed is tested on GTX2080ti*4, and the image size is 608.

3. The training set should match the testing set, because mismatch will cause the mistakes of mAP.
   Read [issue](https://github.com/ultralytics/yolov3/issues/970) for detial.

## Train command

1.YOLOv3

```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3-608.weights --cfg cfg/yolov3/yolov3.cfg --img_size ...
```

Weights Download

- [COCO pretraining weights](https://pan.baidu.com/s/1JZylwRQIgAd389oWUu0djg)

  Extract code：k8ms

2.YOLOv3tiny

```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3tiny.weights --cfg cfg/yolov3tiny/yolov3-tiny.cfg --img_size ...
```

- [COCO pretraining weights](https://pan.baidu.com/s/1iWGxdjR3TWxEe37__msyRA)

  Extract code：udfe

3.YOLOv3tiny-mobilenet-small

```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3tiny-mobilenet-small.weights --cfg cfg/yolov3tiny-mobilenet-small/yolov3tiny-mobilenet-small-coco.cfg --img_size ...
```

- [COCO pretraining weights](https://pan.baidu.com/s/1mSFjWLU91H2OhNemsAeiiQ)

  Extract code：pxz4

4.YOLOv3-mobilenet

```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov3-mobilenet.weights --cfg cfg/yolov3-mobilenet/yolov3-mobilenet-coco.cfg --img_size ...
```

- [COCO pretraining weights](https://pan.baidu.com/s/1EI2Xh1i18CRLoZo_P3NVHw)

  Extract code：3vm8

5.YOLOv4

```bash
python3 train.py --data data/... --batch-size ... -pt --weights weights/yolov4.weights --cfg cfg/yolov4/yolov4.cfg --img_size ...
```

- [COCO pretraining weights](https://pan.baidu.com/s/1jAGNNC19oQhAIgBfUrkzmQ)

  Extract code：njdg

# Model Compression

## 1. Pruning

### Features

|<center>method</center> |<center>advantage</center>|<center>disadvantage</center> |
| --- | --- | --- |
|Normal pruning        |Not prune for shortcut layer. It has a considerable and stable compression rate that requires no fine tuning.|The compression rate is limited.  |
|Shortcut pruning      |Very high compression rate.  |Fine-tuning is necessary.  |
|Silmming              |Shortcut fusion method was used to improve the precision of shear planting.|Best way for shortcut pruning|
|Regular pruning       |Designed for hardware deployment, the number of filters after pruning is a multiple of 2, no fine-tuning, support tiny-yolov3 and Mobilenet.|Part of the compression ratio is sacrificed for regularization. |
|layer pruning         |ResBlock is used as the basic unit for purning, which is conducive to hardware deployment. |It can only cut backbone. |
|layer-channel pruning |First, use channel pruning and then use layer pruning, and pruning rate was very high. |Accuracy may be affected. |

### Step

1.Training

```bash
python3 train.py --data ... -pt --batch-size ... --weights ... --cfg ...
```

2.Sparse training

`-sr`Sparse training，`--s`Specifies the sparsity factor，`--prune`Specify the sparsity type.

`--prune 0` is the sparsity of normal pruning and regular pruning.

`--prune 1` is the sparsity of shortcut pruning.

`--prune 2` is the sparsity of layer pruning.

command：

```bash
python3 train.py --data ... -pt --batch-size 32  --weights ... --cfg ... -sr --s 0.001 --prune 0 
```

3.Pruning

- normal pruning

```bash
python3 normal_prune.py --cfg ... --data ... --weights ... --percent ...
```

- regular pruning

```bash
python3 regular_prune.py --cfg ... --data ... --weights ... --percent ...
```

- shortcut pruning

```bash
python3 shortcut_prune.py --cfg ... --data ... --weights ... --percent ...
```

- silmming

```bash
python3 slim_prune.py --cfg ... --data ... --weights ... --percent ...
```

- layer pruning

```bash
python3 layer_prune.py --cfg ... --data ... --weights ... --shortcut ...
```

- layer-channel pruning

```bash
python3 layer_channel_prune.py --cfg ... --data ... --weights ... --shortcut ... --percent ...
```

It is important to note that the cfg and weights variables in OPT need to be pointed to the cfg and weights files
generated by step 2.

In addition, you can get more compression by increasing the percent value in the code.
(If the sparsity is not enough and the percent value is too high, the program will report an error.)

### Pruning experiment

1.normal pruning oxfordhand，img_size = 608，test on GTX2080Ti*4

|<center>model</center> |<center>parameter before pruning</center> |<center>mAP before pruning</center>|<center>inference time before pruning</center>|<center>percent</center> |<center>parameter after pruning</center> |<center>mAP after pruning</center> |<center>inference time after pruning</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(without fine tuning)     |58.67M   |0.806   |0.1139s   |0.8    |10.32M |0.802 |0.0844s |
|yolov3-mobilenet(fine tuning)   |22.75M   |0.812   |0.0345s   |0.97   |2.72M  |0.795 |0.0211s |
|yolov3tiny(fine tuning)         |8.27M    |0.708   |0.0144s   |0.5    |1.13M  |0.641 |0.0116s |

2.regular pruning oxfordhand，img_size = 608，test ong GTX2080Ti*4

|<center>model</center> |<center>parameter before pruning</center> |<center>mAP before pruning</center>|<center>inference time before pruning</center>|<center>percent</center> |<center>parameter after pruning</center> |<center>mAP after pruning</center> |<center>inference time after pruning</center>
| --- | --- | --- | --- | --- | --- | --- | --- |
|yolov3(without fine tuning)           |58.67M   |0.806   |0.1139s   |0.8    |12.15M |0.805 |0.0874s |
|yolov3-mobilenet(fine tuning)   |22.75M   |0.812   |0.0345s   |0.97   |2.75M  |0.803 |0.0208s |
|yolov3tiny(fine tuning)         |8.27M    |0.708   |0.0144s   |0.5    |1.82M  |0.703 |0.0122s |

## 2.quantization

`--quantized 2` Dorefa quantization method

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 2
```

`--quantized 1` Google quantization method

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --quantized 1
```

`--BN_Flod` using BN Flod training, `--FPGA` Pow(2) quantization for FPGA.

### experiment

oxfordhand, yolov3, 640image-size |<center>method</center> |<center>mAP</center> | | --- | --- | |Baseline |0.847 |
|Google8bit |0.851 | |Google8bit + BN Flod |0.851 | |Google8bit + BN Flod + FPGA |0.852 | |Google4bit + BN Flod + FPGA
|0.842 |

## 3.Knowledge Distillation

### Knowledge Distillation

The distillation method is based on the basic distillation method proposed by Hinton in 2015, and has been partially
improved in combination with the detection network.

Distilling the Knowledge in a Neural Network
[paper](https://arxiv.org/abs/1503.02531)

command : `--t_cfg --t_weights --KDstr`

`--t_cfg` cfg file of teacher model

`--t_weights` weights file of teacher model

`--KDstr` KD strategy

    `--KDstr 1` KLloss can be obtained directly from the output of teacher network and the output of student network and added to the overall loss.
    `--KDstr 2` To distinguish between box loss and class loss, the student does not learn directly from the teacher. L2 distance is calculated respectively for student, teacher and GT. When student is greater than teacher, an additional loss is added for student and GT.
    `--KDstr 3` To distinguish between Boxloss and ClassLoss, the student learns directly from the teacher.
    `--KDstr 4` KDloss is divided into three categories, box loss, class loss and feature loss.
    `--KDstr 5` On the basis of KDstr 4, the fine-grain-mask is added into the feature

example:

```bash
python train.py --data ... --batch-size ... --weights ... --cfg ... --img-size ... --epochs ... --t_cfg ... --t_weights ...
```

Usually, the pre-compression model is used as the teacher model, and the post-compression model is used as the student
model for distillation training to improve the mAP of student network.

### experiment

oxfordhand，yolov3tiny as teacher model，normal pruning yolov3tiny as student model

|<center>teacher model</center> |<center>mAP of teacher model</center> |<center>student model</center>|<center>directly fine tuning</center>|<center>KDstr 1</center> |<center>KDstr 2</center> |<center>KDstr 3</center>  |<center>KDstr 4(L1)</center> |<center>KDstr 5(L1)</center> |
| --- | --- | --- | --- | --- | --- | --- |--- |--- |
|yolov3tiny608   |0.708    |normal pruning yolov3tiny608    |0.658     |0.666    |0.661  |0.672   |0.673   |0.674   |