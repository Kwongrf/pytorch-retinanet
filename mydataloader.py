from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import six
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

import pydicom

#######################################################KRF CREATED 2018/11/14######################################################################
class MyDataset(Dataset):
    """My dataset.类似于CSV dataset，差异在于CSV格式稍有不同，以及图片处理过程不同"""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
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

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
#允许出现(0,0,0,0)的情况
#             if (x2-x1) < 1 or (y2-y1) < 1:
#                 continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2) == ('', '', '', ''):#！！！！这里原来没有将normal的图片加载进去
                result[img_file].append({'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0, 'class': class_name})
            else:
                ##################change int to float by KRF######################################
                x1 = self._parse(x1,  float, 'line {}: malformed x1: {{}}'.format(line))
                y1 = self._parse(y1,  float, 'line {}: malformed y1: {{}}'.format(line))
                x2 = self._parse(x2,  float, 'line {}: malformed x2: {{}}'.format(line))
                y2 = self._parse(y2,  float, 'line {}: malformed y2: {{}}'.format(line))

                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                # check if the current class name is correctly present
                if class_name not in classes:
                    raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

                result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1
    #####################################Modified by KRF######################
    def image_aspect_ratio(self, image_index):
        #image = Image.open(self.image_names[image_index])
        ds = pydicom.read_file(self.image_names[image_index])
        img_arr = ds.pixel_array
        image = Image.fromarray(img_arr).convert('RGB')
        return float(image.width) / float(image.height)
###############################################################################################################################################


#######################################################KRF CREATED 2018/11/15###################################################
class TestDataset(Dataset):
    """Test dataset.类似于MyDataset，不过没有csv和label"""

    def __init__(self, test_fp, transform=None):
        """
        Args:
            test_fp (string):训练集的文件目录
            
        """
        self.test_fp = test_fp
        
        self.transform = transform
        self.image_names = os.listdir(self.test_fp) #只测试100张
        for i in range(len(self.image_names)):
            self.image_names[i] =  os.path.join(test_fp,self.image_names[i])

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




