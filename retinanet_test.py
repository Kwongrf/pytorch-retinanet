#!/usr/bin/env python
# coding: utf-8

# In[24]:


import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import pydicom
import skimage.io
import skimage.transform
import skimage.color
import skimage
#import cv2
from PIL import Image

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, MyDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#参数
NAME="RSNA"
DATA_PATH = "/data/krf/dataset"
CSV_TRAIN = DATA_PATH + "/csv_train.csv"
CSV_VAL = DATA_PATH + "/csv_val.csv"
CSV_CLASSES = DATA_PATH + "/classes.csv"
TEST_PATH = DATA_PATH +"/stage_2_test_images"




#######################################################KRF CREATED 2018/11/15###################################################
class TestDataset(Dataset):
    """Test dataset.类似于MyDataset，不过没有csv和label"""

    def __init__(self,test_fp, transform=None):
        """
        Args:
            test_fp (string):训练集的文件目录
            
        """
        self.test_fp = test_fp
        self.transform = transform
        self.image_names = os.listdir(self.test_fp) #只测试100张
        for i in range(len(self.image_names)):
            self.image_names[i] =  os.path.join(TEST_PATH,self.image_names[i])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        sample = {'img': img,'name' : self.image_names[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
#         img = skimage.io.imread(self.image_names[image_index])
        ds = pydicom.read_file(self.image_names[image_index])
        img = ds.pixel_array
        
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0
    
    #####################################Modified by KRF######################
    def image_aspect_ratio(self, image_index):
        #image = Image.open(self.image_names[image_index])
        ds = pydicom.read_file(self.image_names[image_index])
        img_arr = ds.pixel_array
        image = Image.fromarray(img_arr).convert('RGB')
        return float(image.width) / float(image.height)
###################################################################################################################################


# In[14]:


#覆盖另一个
class NormalizerTest(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, name = sample['img'], sample['name']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'name': name}
class ResizerTest(object):
    """Convert ndarrays in sample to Tensors."""

    #def __call__(self, sample, min_side=608, max_side=1024):
    ###########################################KRF Modeified###########################################################
    def __call__(self, sample, min_side=512, max_side=1024):
        image, name = sample['img'], sample['name']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        #annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'name': name, 'scale': scale}

dataset_test = TestDataset(TEST_PATH,transform=transforms.Compose([NormalizerTest(), ResizerTest()]))
print(len(dataset_test))




#retinanet.load_state_dict('weights/RSNA_retinanet_3.pt')
def collaterTest(data):

    imgs = [s['img'] for s in data]
    names = [s['name'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

#    max_num_annots = max(annot.shape[0] for annot in annots)
    
#     if max_num_annots > 0:

#         annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

#         if max_num_annots > 0:
#             for idx, annot in enumerate(annots):
#                 #print(annot.shape)
#                 if annot.shape[0] > 0:
#                     annot_padded[idx, :annot.shape[0], :] = annot
#     else:
#         annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'names': names, 'scale': scales}


sampler_test = AspectRatioBasedSampler(dataset_test, batch_size=1, drop_last=False)
dataloader_test = DataLoader(dataset_test, num_workers=1, collate_fn=collaterTest, batch_sampler=sampler_test)

retinanet = torch.load("weights/model_final.pt")

use_gpu = True

if use_gpu:
    retinanet = retinanet.cuda()

retinanet.eval()

unnormalize = UnNormalizer()

#def draw_caption(image, box, caption):

#    b = np.array(box).astype(int)
#    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
#    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


# In[1]:


filepath = "submission.csv"
with open(filepath, 'w') as file:
    file.write("patientId,PredictionString\n")
    for idx, data in enumerate(dataloader_test):
        patientId = os.path.splitext(os.path.basename(data['names'][0]))[0]
        print(patientId)
        file.write(patientId+",")
        with torch.no_grad():
            st = time.time()
            try:
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
                print('Elapsed time: {}'.format(time.time()-st))
                idxs = np.where(scores>0.5)
                img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                img[img<0] = 0
                img[img>255] = 255

                img = np.transpose(img, (1, 2, 0))

                #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                outstr = ""#输出到csv中
                resize_factor = 2 #输入的是512，原图像是1024
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    #label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                    #draw_caption(img, (x1, y1, x2, y2), "1")

                    #cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    #print(label_name)
                    width = x2 - x1;
                    height = y2 - y1
                    outstr += ' '
                    s = scores[idxs[0][j]].cpu().numpy()
                    print(s)
                    outstr += str(s)
                    outstr += ' '
                    bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, width*resize_factor, height*resize_factor)
                    outstr += bboxes_str
            except Exception as e:
                print(e)
                continue 
            
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
            file.write(outstr+"\n")
        


# In[ ]:




