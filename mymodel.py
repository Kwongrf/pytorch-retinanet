from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch
from torch.utils import model_zoo
import math

from collections import OrderedDict

from anchors import Anchors

import losses
from utils import BBoxTransform, ClipBoxes

from model import PyramidFeatures,ClassificationModel,RegressionModel,nms
from senet import SEResNeXtBottleneck,pretrained_settings

class PyramidFeatures2(nn.Module):
    def __init__(self, C3_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
#         # upsample C5 to get P5 from the FPN paper
#         self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
#         self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
#         self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
#         self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
#         self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
#         self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3  = inputs

#         P5_x = self.P5_1(C5)
#         P5_upsampled_x = self.P5_upsampled(P5_x)
#         P5_x = self.P5_2(P5_x)
        
#         P4_x = self.P4_1(C4)
#         P4_x = P5_upsampled_x + P4_x
#         P4_upsampled_x = self.P4_upsampled(P4_x)
#         P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
#         P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C4)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P6_x, P7_x]
    
class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        #TODO ##########################################################
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        
        #self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        self.last_linear = nn.Linear(247808, num_classes)
#         512+32 = 544 544/4 /2 /2 /2 -7+1 = 11 11*11*2048 = 247808
#         torch.Size([1, 3, 544, 544])
#         torch.Size([1, 3, 544, 544])
#         torch.Size([1, 64, 136, 136])
#         torch.Size([1, 64, 136, 136])
#         torch.Size([1, 256, 136, 136])
#         torch.Size([1, 512, 68, 68])
#         torch.Size([1, 256, 136, 136])
#         torch.Size([1, 1024, 34, 34])
#         torch.Size([1, 512, 68, 68])
#         torch.Size([1, 2048, 17, 17])
#         torch.Size([1, 2048, 11, 11])
#         torch.Size([1, 247808])
#         torch.Size([1, 1024, 34, 34])
#         torch.Size([1, 2048, 17, 17])
#         torch.Size([1, 2048, 11, 11])
#         torch.Size([1, 247808])
        #TODO ##########################################################

        if block == SEResNeXtBottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
        
        self.focalLoss = losses.FocalLoss()
        
        self.criterion =  nn.CrossEntropyLoss()
        ##############################################################
#         self.regressionModel2 = RegressionModel(256)
#         self.classificationModel2 = ClassificationModel(256, num_classes=num_classes)

#         self.anchors2 = Anchors()

#         self.regressBoxes2 = BBoxTransform()

#         self.clipBoxes2 = ClipBoxes()
        
#         self.focalLoss2 = losses.FocalLoss()
        ###############################################################      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        #############################################################################
#         self.classificationModel2.output.weight.data.fill_(0)
#         self.classificationModel2.output.bias.data.fill_(-math.log((1.0-prior)/prior))

#         self.regressionModel2.output.weight.data.fill_(0)
#         self.regressionModel2.output.bias.data.fill_(0)
        #############################################################################
        self.freeze_bn()
        
        
    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def logits(self, x):
        x = self.avg_pool(x)
#         print(x.size())
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
#         print(x.size())
        x = self.last_linear(x)
        return x   
    
    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
#         print(img_batch.size())   
        x0 = self.layer0(img_batch)
#         print(x0.shape)
        x1 = self.layer1(x0)
#         print(x1.size())
        x2 = self.layer2(x1)
#         print(x2.size())
        x3 = self.layer3(x2)
#         print(x3.size())
        x4 = self.layer4(x3)
#         print(x4.size())
        
        whole_class = self.logits(x4)
        
        
        features = self.fpn([x2, x3, x4])
        
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            target = annotations[:,0,4].type(torch.cuda.LongTensor)
            whole_class_loss = self.criterion(whole_class, target)#第一个box的class即整张图像的 annotations.shape= (batch_size,box_num,5)
            return self.focalLoss(classification, regression, anchors, annotations),whole_class_loss
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
#                 return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
                return [torch.zeros([1]).cuda(0), torch.zeros([1]).cuda(0), torch.zeros([1, 4]).cuda(0)] #modified by krf 2018/11/29 github issue

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

def initialize_pretrained_model(model, num_classes, settings):
#     assert num_classes == settings['num_classes'], \
#         'num_classes should be {}, but is {}'.format(
#             settings['num_classes'], num_classes)
#    model.load_state_dict(model_zoo.load_url(settings['url']))
#    model.load_state_dict('se_resnext101_32x4d-3b2fe3d8.pth')
    model = torch.load('se_resnext101_32x4d-3b2fe3d8.pth')
#    model = torch.load('se_resnext50_32x4d-a260b3a4.pth')
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    
def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model