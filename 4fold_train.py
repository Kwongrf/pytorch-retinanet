#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load retinanet_train.py
#!/usr/bin/env python

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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
import mymodel
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
from mydataloader import MyDataset,RandomTransformer
import coco_eval
import csv_eval
# import visdom
# vis = visdom.Visdom(env='retinanet_seresnextAngu')
from tensorboardX import SummaryWriter

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#参数
NAME="RSNA"
DATA_PATH = "/data/krf/dataset"
CSV_TRAINS = [DATA_PATH+"/csv_train.csv"]
CSV_VALS = [DATA_PATH +"/csv_val.csv"]
# CSV_TRAINS = [DATA_PATH + "/csv_train0.csv",DATA_PATH + "/csv_train1.csv",DATA_PATH + "/csv_train2.csv",DATA_PATH + "/csv_train3.csv"]
# CSV_VALS = [DATA_PATH + "/csv_val0.csv",DATA_PATH + "/csv_val1.csv",DATA_PATH + "/csv_val2.csv",DATA_PATH + "/csv_val3.csv"]
CSV_CLASSES = DATA_PATH + "/classes.csv"
DEPTH = 101
EPOCHS = 30
BATCH_SIZE=4
VAL_STEP = 1000
MODEL_NUM = 1
#VAL_SIZE = 3000
#TRAIN_SIZE = 100
#数据预处理
import csv
import random
from tqdm import tqdm


# In[15]:


# with open(DATA_PATH+"/stage_1_train_labels.csv") as f:
#     reader = csv.reader(f)
#     rows=[row for row in  reader]
#     rows = rows[1:]
#     random.shuffle(rows)
#     for row in rows:
#         row[0] = DATA_PATH+"/stage_1_train_images/"+row[0]+".dcm"
#         if row[1] == '' and row[2] == '' and row[3] == '' and row[4] == '':
#             row[5] = ''
#         else:
#             row[3] = str(float(row[1]) + float(row[3]))# x2 = x1 + w 
#             row[4] = str(float(row[2]) + float(row[4]))# y2 = y1 + h
#     val_rows = rows[:VAL_SIZE]
#     train_rows = rows[VAL_SIZE:]
#     print(len(val_rows),len(train_rows))
#     with open(CSV_TRAIN,'w') as f2:
#         write = csv.writer(f2)
#         write.writerows(train_rows)
#         print("csv_train 写入完毕")
#     with open(CSV_VAL,'w') as f3:
#         write = csv.writer(f3)
#         write.writerows(val_rows)
#         print("csv_val 写入完毕")

# with open(CSV_CLASSES,'w') as f:
#     write = csv.writer(f)
    
#     row = ['0','0']
#     write.writerow(row)
#     row = ['1','1']
#     write.writerow(row)
#     row = ['2','2']
#     write.writerow(row)
#     print("csv_classes 写入完毕")



# # #每次跑这个函数之前需要先删除之前的

# df = pd.read_csv(DATA_PATH+"/stage_1_train_labels.csv")
# ddf = pd.read_csv(DATA_PATH+'/stage_1_detailed_class_info.csv')
# train_images = os.listdir(DATA_PATH+"/stage_1_train_images")

# random.shuffle(train_images)#打乱图片顺序
# count = 0
# pos_cnt_train = [0,0,0,0]
# pos_cnt_val = [0,0,0,0]
# class_dict = {'Normal':0,'Lung Opacity':1,'No Lung Opacity / Not Normal':2}
# for img_name in tqdm(train_images):
#     results = df[df['patientId']==img_name.split('.')[0]].values
#     detail = ddf[df['patientId']==img_name.split('.')[0]].values
#     for row in results:
#         row[0] = DATA_PATH+"/stage_1_train_images/"+row[0]+".dcm"
#         if row[5] == 1:
#             pos_cnt_val[count % 4] += 1
#         if row[1] >= 0 and row[1] <= 1024:
#             row[3] = str(float(row[1]) + float(row[3]))# x2 = x1 + w 
#             row[4] = str(float(row[2]) + float(row[4]))# y2 = y1 + h
#         else:
#             row[1] = ''
#             row[2] = ''
#             row[3] = ''
#             row[4] = ''
#             row[5] = class_dict[detail[0,1]]
#         with open(CSV_VALS[count % 4],'a') as f:
#             write = csv.writer(f)
#             write.writerow(row)
#         for i in range(4):
#             if count % 4 == i:
#                 continue
#             else:
#                 with open(CSV_TRAINS[count % 4],'a') as f:
#                     write = csv.writer(f)
#                     write.writerow(row) 
#     count += 1
# print(pos_cnt_train,pos_cnt_val) 


# In[2]:



print("Loading data...")

#%time
#制作数据loader
dataset_train = []
dataset_val = []

for i in range(MODEL_NUM): 
    dataset_train.append(MyDataset(train_file=CSV_TRAINS[i], class_list=CSV_CLASSES, transform=transforms.Compose([Normalizer(), RandomTransformer(), Resizer()])))
    dataset_val.append(MyDataset(train_file=CSV_VALS[i], class_list=CSV_CLASSES, transform=transforms.Compose([Normalizer(), Resizer()])))

#每次的sampler的参数：来源、batchsize、是否抛弃最后一层？？？

# num_workers 同时工作的组？collater:校验用的吧
dataloader_train = []
dataloader_val = []

for i in range(MODEL_NUM):
    sampler = AspectRatioBasedSampler(dataset_train[i], batch_size=BATCH_SIZE, drop_last=False)
    sampler_val = AspectRatioBasedSampler(dataset_val[i], batch_size=BATCH_SIZE, drop_last=False)
    dataloader_train.append(DataLoader(dataset_train[i], num_workers=4, collate_fn=collater, batch_sampler=sampler))
    dataloader_val.append(DataLoader(dataset_val[i], num_workers=4, collate_fn=collater, batch_sampler=sampler_val))


# In[3]:


retinanets = []
# Create the model
# if DEPTH == 18:
#     retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
# elif DEPTH == 34:
#     retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
# elif DEPTH == 50:
#     for i in range(4):
#         retinanets.append(model.resnet50(num_classes=dataset_train[i].num_classes(), pretrained=True))
# elif DEPTH == 101:
#     for i in range(4):
#         retinanets.append(model.resnet101(num_classes=dataset_train[i].num_classes(), pretrained=True)) 
# elif DEPTH == 152:
#     for i in range(4):
#         retinanets.append(model.resnet152(num_classes=dataset_train[i].num_classes(), pretrained=True))
# else:
#     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
for i in range(MODEL_NUM):
    retinanets.append(mymodel.se_resnext101_32x4d(num_classes=dataset_train[i].num_classes()))
    
optimizer = []
scheduler = []
loss_hist = []

classloss_hist = []
regressloss_hist = []
wholeclassloss_hist = []

writer = SummaryWriter()
#retinanets[0] = torch.load('weights_stage1/{}_seresnext101_{}.pt'.format(0,3))
#retinanets[1] = torch.load('weights_stage1/{}_seresnext101_{}.pt'.format(1,2))
#retinanets[2] = torch.load('weights_stage1/{}_seresnext101_{}.pt'.format(2,2))
#retinanets[3] = torch.load('weights_stage1/{}_seresnext101_{}.pt'.format(3,2))
for i in range(MODEL_NUM):
    
    retinanets[i] = retinanets[i].cuda()
    #变成并行
    retinanets[i] = torch.nn.DataParallel(retinanets[i]).cuda()
    #训练模式
    retinanets[i].training = True
    #学习率0.0001  
    optimizer.append(optim.Adam(retinanets[i].parameters(), lr=1e-5))
    #如果3个epoch损失没有减少则降低学习率
    scheduler.append(optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], patience=1, verbose=True))
    # TODO 这是干什么 deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：这里是定义了一个500的队列
    loss_hist.append(collections.deque(maxlen=500))
    classloss_hist.append(collections.deque(maxlen=500))
    regressloss_hist.append(collections.deque(maxlen=500))
    wholeclassloss_hist.append(collections.deque(maxlen=500))
    print('Num training images: {}'.format(len(dataset_train[i])))
# In[5]:


# In[6]:

count = np.zeros(MODEL_NUM)
#count = [6418,6418,4812,4812]#TODO 0
import traceback
for epoch_num in range(EPOCHS):#TODO 4

    for i in range(MODEL_NUM):
        retinanets[i].train()
        retinanets[i].module.freeze_bn()
        epoch_loss = []
        
        for iter_num, data in enumerate(dataloader_train[i]):
            try:
                optimizer[i].zero_grad()

                focal_loss, whole_class_loss = retinanets[i]([Variable(data['img'].cuda().float()), Variable(data['annot'].cuda())])
                classification_loss, regression_loss = focal_loss
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                whole_class_loss = whole_class_loss.mean()
                loss = classification_loss + regression_loss + whole_class_loss

                if bool(loss == 0):
                    continue
                #反向传播？
                loss.backward()

                #这是干嘛？？TODO
                torch.nn.utils.clip_grad_norm_(retinanets[i].parameters(), 0.1)

                #这?TODO
                optimizer[i].step()

                loss_hist[i].append(float(loss))
                classloss_hist[i].append(float(classification_loss))
                regressloss_hist[i].append(float(regression_loss))
                wholeclassloss_hist[i].append(float(whole_class_loss))
                
                epoch_loss.append(float(loss))
                if iter_num % 10 == 0:
                    print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist[i])))
    #                 vis.line(X=torch.Tensor([count[i]]), Y=torch.Tensor([np.mean(loss_hist[i])]), win='train loss '+str(i), update='append' ,opts={'title':'train loss '+str(i)})
    #                 vis.line(X=torch.Tensor([count[i]]), Y=torch.Tensor([np.mean(classloss_hist[i])]), win='classification loss '+str(i), update='append' ,opts={'title':'classification loss '+str(i)})
    #                 vis.line(X=torch.Tensor([count[i]]), Y=torch.Tensor([np.mean(regressloss_hist[i])]), win='regression loss '+str(i), update='append' ,opts={'title':'regression loss '+str(i)})
    #                 vis.line(X=torch.Tensor([count[i]]), Y=torch.Tensor([np.mean(wholeclassloss_hist[i])]), win='whole class loss '+str(i), update='append' ,opts={'title':'whole class loss '+str(i)})
                    writer.add_scalar('Model_{}/Loss'.format(i), np.mean(loss_hist[i]), count[i])
                    writer.add_scalar('Model_{}/Classification Loss'.format(i), np.mean(classloss_hist[i]), count[i])
                    writer.add_scalar('Model_{}/Regression Loss'.format(i), np.mean(regressloss_hist[i]), count[i])
                    writer.add_scalar('Model_{}/Whole Class Loss'.format(i), np.mean(wholeclassloss_hist[i]), count[i])
                count[i] += 1
#                 vis.save(['retinanet_seresnextAngu'])
                
                del focal_loss
                del classification_loss
                del regression_loss
                del whole_class_loss
#                 if count[i] % VAL_STEP == 0:
#                     print("Evaluating dataset")
#                     retinanets[i].eval()
#                     mAP = csv_eval.evaluate(dataset_val[i],retinanets[i])
#                     vis.line(X=torch.Tensor([count[i]]), Y=torch.Tensor([mAP[0][0]]), win='val mAP '+str(i), update='append' ,opts={'title':'mAP val '+str(i)})
#                     vis.save(['retinanet_seresnext101'])
#                     torch.save(retinanets[i].module, 'weights_stage1/{}_seresnext101_{}.pt'.format(i, epoch_num))
#                     retinanets[i].train()
            except Exception as e:
                print(e)
                continue
        
        print("Evaluating")
        retinanets[i].eval()
        try:
            mAP = csv_eval.evaluate(dataset_val[i],retinanets[i])
            writer.add_scalar('Model_{}/mAP'.format(i), mAP[1][0], epoch_num)
#             vis.line(X=torch.Tensor([epoch_num]), Y=torch.Tensor([mAP[1][0]]), win='val mAP '+str(i), update='append' ,opts={'title':'mAP val '+str(i)})
#             vis.save(['retinanet_seresnextAngu'])
        except Exception as e:
            print(e)
        #这一步也看不懂？？TODO 
        scheduler[i].step(np.mean(epoch_loss))
        torch.save(retinanets[i].module, 'weights_stage1/{}_seresnext101_{}.pt'.format(i, epoch_num))
    #     torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

for i in range(4):
    retinanets[i].eval()
    torch.save(retinanets[i], 'weights_stage1/{}_final_seresnext101.pt'.format(i))

writer.export_scalars_to_json("./test.json")
writer.close()
