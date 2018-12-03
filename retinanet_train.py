#!/usr/bin/env python
# coding: utf-8

# In[1]:

print("Start importing package...")
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

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, MyDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval
import visdom
vis = visdom.Visdom(env='retinanet')

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#参数
NAME="RSNA"
DATA_PATH = "/data/krf/dataset"
CSV_TRAIN = DATA_PATH + "/csv_train.csv"
CSV_VAL = DATA_PATH + "/csv_val.csv"
CSV_CLASSES = DATA_PATH + "/classes.csv"
DEPTH = 101
EPOCHS = 20
BATCH_SIZE=8
VAL_SIZE = 3000
#TRAIN_SIZE = 100


# In[2]:


#数据预处理
import csv
import random
with open(DATA_PATH+"/stage_1_train_labels.csv") as f:
    reader = csv.reader(f)
    rows=[row for row in  reader]
    rows = rows[1:]
    random.shuffle(rows)
    for row in rows:
        row[0] = DATA_PATH+"/stage_1_train_images/"+row[0]+".dcm"
        if row[1] == '' and row[2] == '' and row[3] == '' and row[4] == '':
            row[5] = ''
        else:
            row[3] = str(float(row[1]) + float(row[3]))# x2 = x1 + w 
            row[4] = str(float(row[2]) + float(row[4]))# y2 = y1 + h
    val_rows = rows[:VAL_SIZE]
    train_rows = rows[VAL_SIZE:]
    print(len(val_rows),len(train_rows))
    with open(CSV_TRAIN,'w') as f2:
        write = csv.writer(f2)
        write.writerows(train_rows)
        print("csv_train 写入完毕")
    with open(CSV_VAL,'w') as f3:
        write = csv.writer(f3)
        write.writerows(val_rows)
        print("csv_val 写入完毕")

with open(CSV_CLASSES,'w') as f:
    write = csv.writer(f)
    row = ['1','0']
    write.writerow(row)
    print("csv_classes 写入完毕")


# In[3]:
print("Loading data...")

#%time
#制作数据loader
dataset_train = MyDataset(train_file=CSV_TRAIN, class_list=CSV_CLASSES, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
dataset_val = MyDataset(train_file=CSV_VAL, class_list=CSV_CLASSES, transform=transforms.Compose([Normalizer(), Resizer()]))

#每次的sampler的参数：来源、batchsize、是否抛弃最后一层？？？
sampler = AspectRatioBasedSampler(dataset_train, batch_size=BATCH_SIZE, drop_last=False)
# num_workers 同时工作的组？collater:校验用的吧
dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)

# if dataset_val is not None:
#     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
#     dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)

dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)



# In[4]:


# Create the model
if DEPTH == 18:
    retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
elif DEPTH == 34:
    retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
elif DEPTH == 50:
    retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
elif DEPTH == 101:
    retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
elif DEPTH == 152:
    retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
#retinanet = torch.load('weights/RSNA_retinanet_5.pt')
use_gpu = True

if use_gpu:
    retinanet = retinanet.cuda()
#变成并行
retinanet = torch.nn.DataParallel(retinanet).cuda()
#训练模式
retinanet.training = True
#学习率0.00001  
optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
#如果3个epoch损失没有减少则降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
# TODO 这是干什么 deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：这里是定义了一个500的队列
loss_hist = collections.deque(maxlen=1000)


# In[5]:


retinanet.train()
#BN层冻结！！
retinanet.module.freeze_bn()

print('Num training images: {}'.format(len(dataset_train)))
count = 0

for epoch_num in range(EPOCHS):

    retinanet.train()
    retinanet.module.freeze_bn()
    
    epoch_loss = []
    
    for iter_num, data in enumerate(dataloader_train):
        try:
            optimizer.zero_grad()

            classification_loss, regression_loss = retinanet([Variable(data['img'].cuda().float()), Variable(data['annot'].cuda())])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss
            
            if bool(loss == 0):
                continue
            #反向传播？
            loss.backward()

            #这是干嘛？？TODO
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            #这?TODO
            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
            vis.line(X=torch.Tensor([count]), Y=torch.Tensor([np.mean(loss_hist)]), win='train loss', update='append' ,opts={'title':'train loss'})
            count += 1
            vis.save(['retinanet'])
            
            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue


    print("Evaluating dataset")
#     retinanet.eval()
#     for index,data in enumerate(dataloader_val):
#         #data = dataset[index]
#         scale = data['scale']
#         # run network
#         #scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
#         print(Variable(data['img'].cuda()))
#         scores, labels, boxes = retinanet(Variable(data['img'].cuda()))
#         #mAP = csv_eval.evaluate(dataset_val,retinanet)
    try:
        mAP = csv_eval.evaluate(dataset_val,retinanet)
        vis.line(X=torch.Tensor([epoch_num]), Y=torch.Tensor([mAP[0][0]]), win='val mAP', update='append' ,opts={'title':'mAP val'})
        vis.save(['retinanet'])
    except Exception as e:
        print(e)
        continue
    #这一步也看不懂？？TODO 
    scheduler.step(np.mean(epoch_loss))
    
    torch.save(retinanet.module, 'weights_stage1/{}_retinanet_{}.pt'.format(NAME, epoch_num))
#     torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

retinanet.eval()

torch.save(retinanet, 'weights_stage1/model_final.pt'.format(epoch_num))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




