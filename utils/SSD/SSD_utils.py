# Author:LiPu

from utils.layers import *
from utils.SSD import box_utils
from utils.SSD.container import Container
from utils.SSD.nms import batched_nms
from utils.SSD.prior_box import PriorBox


def cfg_backbone(cfg):
    model = Backbone(cfg)
    return model


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.module_defs = self.parse_model_cfg(cfg)
        self.module_list, self.l2_norm_index, self.features, self.l2_norm = self.create_backbone(self.module_defs)
        self.reset_parameters()

    def parse_model_cfg(self, path):
        # Parses the ssd layer configuration file and returns module definitions
        # print(os.getcwd())#绝对路径
        # print(os.path.abspath(os.path.join(os.getcwd(), "../../..")))  #获取上上上级目录
        # file = open(os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../../..")),path), 'r')  #测试本文件时使用
        file = open(path, 'r')
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        mdefs = []  # module definitions
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                mdefs.append({})
                mdefs[-1]['type'] = line[1:-1].rstrip()
                if mdefs[-1]['type'] == 'convolutional':
                    mdefs[-1]['batch_normalize'] = '0'  # pre-populate with zeros (may be overwritten later)
                    mdefs[-1]['feature'] = 'no'
                    mdefs[-1]['dilation'] = '1'
            else:
                key, val = line.split("=")
                key = key.rstrip()
                mdefs[-1][key] = val.strip()
        return mdefs

    def create_backbone(self, module_defs):
        # Constructs module list of layer blocks from module configuration in module_defs
        output_filters = [3]
        module_list = nn.ModuleList()
        features = []  # list of layers which rout to detection layes
        l2_norm_index = 0
        l2_norm = 0
        for i, mdef in enumerate(module_defs):
            modules = nn.Sequential()
            # print(mdef)
            if mdef['type'] == 'convolutional':
                bn = int(mdef['batch_normalize'])
                filters = int(mdef['filters'])
                kernel_size = int(mdef['size'])
                pad = int(mdef['pad'])

                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=kernel_size,
                                                       stride=int(mdef['stride']),
                                                       padding=pad,
                                                       dilation=int(mdef['dilation'])))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                elif mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU(inplace=True))
                else:
                    print("Error activation!")
                    raise Exception

                if mdef['feature'] == 'linear':  # 传入预测层
                    features.append(i)
                elif mdef['feature'] == 'l2_norm':
                    features.append(i)
                    l2_norm_index = i
                    l2_norm = L2Norm(filters)

            elif mdef['type'] == 'maxpool':
                kernel_size = int(mdef['size'])
                stride = int(mdef['stride'])
                ceil_mode = True if int(mdef['ceil_mode']) else False
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int(mdef['pad']),
                                       ceil_mode=ceil_mode)  # https://www.cnblogs.com/xxxxxxxxx/p/11529343.html ceilmode
                modules = maxpool
            else:
                print("Error type!")
                raise Exception
            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return module_list, l2_norm_index, features, l2_norm

    def reset_parameters(self):
        for m in self.module_list.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for i in range(len(self.features)):
            if i == 0:
                start = 0
            else:
                start = end
            end = self.features[i] + 1
            for j in range(start, end):
                x = self.module_list[j](x)
            if self.features[i] != self.l2_norm_index:
                features.append(x)
            else:
                feature = self.l2_norm(x)
                features.append(feature)
        return tuple(features)


class SSDBoxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = SSDBoxPredictor()
        self.post_processor = PostProcessor()
        self.priors = None

    def forward(self, features):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred):
        detections = (cls_logits, bbox_pred)
        return detections

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox()().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, 0.1, 0.2
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections


class BoxPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(
                zip([4, 6, 6, 6, 4, 4], [512, 1024, 512, 256, 256, 256])):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1,
                                                                                         11)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred


class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 11, kernel_size=3, stride=1,
                         padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class PostProcessor:
    def __init__(self):
        super().__init__()
        self.width = 300
        self.height = 300

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            indices = torch.nonzero(scores > 0.01).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            keep = batched_nms(boxes, scores, labels, 0.45)
            # keep only topk scoring predictions
            keep = keep[:100]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results


def SSD_targets_Convert(imgs, targets):
    transform = SSDTargetTransform(PriorBox()(), 0.1, 0.2, 0.5)
    batch_size, _, h, w = imgs.shape
    for i in range(0, batch_size):
        index = torch.where(targets[:, 0] == i)
        if index[0].shape[0] == 0:
            box = torch.zeros((1, 8732, 4))
            label = torch.zeros((1, 8732))
        else:
            label_pre_img = targets[index]
            box = torch.zeros(label_pre_img.shape[0], 4)
            box[:, 0] = (label_pre_img[:, 2] - label_pre_img[:, 4] / 2)
            box[:, 1] = (label_pre_img[:, 3] - label_pre_img[:, 5] / 2)
            box[:, 2] = (label_pre_img[:, 2] + label_pre_img[:, 4] / 2)
            box[:, 3] = (label_pre_img[:, 3] + label_pre_img[:, 5] / 2)
            box[:, 0] *= w
            box[:, 1] *= h
            box[:, 2] *= w
            box[:, 3] *= h
            label = label_pre_img[:, 1] + 1
            box, label = transform(box, label)
            box, label = box.unsqueeze(0), label.unsqueeze(0)
        if i == 0:
            boxes = box
            labels = label
        else:
            boxes = torch.cat((boxes, box), 0)
            labels = torch.cat((labels, label), 0)
    labels = labels.long()
    result = Container(
        boxes=boxes,
        labels=labels,
    )
    return result


class SSDTargetTransform:
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)

        return locations, labels
