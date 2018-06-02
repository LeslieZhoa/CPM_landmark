
# coding: utf-8

# In[5]:

import os
import tensorflow as tf 
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt


# In[2]:

def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)


# In[9]:

def make_heatmaps_from_joints(input_size, heatmap_size, gaussian_variance, batch_joints):
    # Generate ground-truth heatmaps from ground-truth 2d joints
    scale_factor = input_size // heatmap_size
    batch_gt_heatmap_np = []
    
    gt_heatmap_np = []
    invert_heatmap_np = np.ones(shape=(heatmap_size, heatmap_size))
    for i in range(batch_joints.shape[0]):
        if batch_joints[i,2]==-1:
            cur_joint_heatmap=np.zeros((heatmap_size,heatmap_size))
            
        else:
            cur_joint_heatmap = make_gaussian(heatmap_size,
                                              gaussian_variance,
                                              center=(batch_joints[i,:2] // scale_factor))
        gt_heatmap_np.append(cur_joint_heatmap)
        invert_heatmap_np -= cur_joint_heatmap
    gt_heatmap_np.append(invert_heatmap_np)
    batch_gt_heatmap_np = np.asarray( gt_heatmap_np)
    batch_gt_heatmap_np = np.transpose(batch_gt_heatmap_np, (1,2,0))

    return batch_gt_heatmap_np


# In[18]:

inpaths=['TC/data/blouse/val/']

for j,path in enumerate(inpaths):
    files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    count=0
    writer = tf.python_io.TFRecordWriter("TC/tfrecord/val/"+"blouse"+".tfrecords")
    for i,f in enumerate(files):
#         if i<100:
            with open(path+f,'r') as fid:
                dat=json.load(fid)
            pts=np.array(dat)

            img = cv2.imread(path+f[0:-5]+'.jpg')
            scale=img.shape[0]/256.0
            pts[:,:2]=pts[:,:2]/scale
            label=make_heatmaps_from_joints(256,32,1.0,pts)
            label_raw=label.tobytes()
            img = cv2.resize(img,(256, 256))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())  #序列化为字符串
            print i
writer.close()


# In[ ]:



