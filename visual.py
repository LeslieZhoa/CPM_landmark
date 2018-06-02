
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import h5py
import time
import os
slim = tf.contrib.slim
from tensorflow.python.ops import control_flow_ops
import cv2
import json
import pandas as pd
phase=False


# In[2]:

train = pd.read_csv("F:/data/TC/TC/fashionAI_key_points_test_a_20180227/test/test.csv")
types=train.image_category
image=train.image_id
inpath='F:/data/TC/TC/fashionAI_key_points_test_a_20180227/test/'
image_name=[]
for i in range(image.shape[0]):
    if types[i]=='trousers':
        image_name.append(image[i])
       


# In[3]:


def dethwise_conv(inputs,num,kernel_size=[3,3],is_training='True',scope=''):
    with slim.arg_scope([slim.conv2d,slim.separable_conv2d],padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95}):
            net=slim.separable_conv2d(inputs,num_outputs=None,depth_multiplier=1,kernel_size=kernel_size,scope=scope+'_dw')
            net=slim.conv2d(net,num,kernel_size=[1,1],scope=scope)
    return net


# In[4]:

class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 16
    def build_model(self,input_image,iftrain):
            with tf.variable_scope('pooled_center_map'):
            # center map is a gaussion template which gather the respose
                self.center_map = slim.avg_pool2d(input_image,
                                              [9, 9], stride=8,
                                              padding='SAME',
                                              scope='center_map')
            with slim.arg_scope([slim.conv2d,slim.separable_conv2d],
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with tf.variable_scope('sub_stages'):
                    net = slim.conv2d(input_image, 64, [3, 3],activation_fn=tf.nn.relu, scope='sub_conv1')
                    net = dethwise_conv(net,64,is_training=iftrain,scope='sub_conv2')
                    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool1')
                    net = dethwise_conv(net,128,is_training=iftrain,scope='sub_conv3')
                    net = dethwise_conv(net,128,is_training=iftrain,scope='sub_conv4')
                    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool2')
                    net = dethwise_conv(net,256,is_training=iftrain,scope='sub_conv5')
                    net = dethwise_conv(net,256,is_training=iftrain,scope='sub_conv6')
                    net = dethwise_conv(net,256,is_training=iftrain,scope='sub_conv7')
                    net = dethwise_conv(net,256,is_training=iftrain,scope='sub_conv8')
                    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool3')
                    net = dethwise_conv(net,512,is_training=iftrain,scope='sub_conv9')
                    net = dethwise_conv(net,512,is_training=iftrain,scope='sub_conv10')
                    net = dethwise_conv(net,512,is_training=iftrain,scope='sub_conv11')
                    net = dethwise_conv(net,512,is_training=iftrain,scope='sub_conv12')
                    net = dethwise_conv(net,512,is_training=iftrain,scope='sub_conv13')
                    net = dethwise_conv(net,512,is_training=iftrain,scope='sub_conv14')
                    self.sub_stage_img_feature = dethwise_conv(net, 128, is_training=iftrain,
                                                             scope='sub_stage_img_feature')

                with tf.variable_scope('stage_1'):
                    conv1 = slim.conv2d(self.sub_stage_img_feature, 512, [1, 1],activation_fn=tf.nn.relu,
                                        scope='conv1')
                    self.stage_heatmap.append(slim.conv2d(conv1, self.joints, [1, 1],activation_fn=None,
                                                          scope='stage_heatmap'))

                for stage in range(2, self.stages+1):
                    self._middle_conv(stage,iftrain)

    def _middle_conv(self,stage,iftrain):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage-2],
                                                 self.sub_stage_img_feature,
                                                 self.center_map],
                                                axis=3)
            with slim.arg_scope([slim.conv2d,slim.separable_conv2d],
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                mid_net = dethwise_conv(self.current_featuremap, 128, kernel_size=[7, 7],is_training=iftrain, scope='mid_conv1')
                mid_net = dethwise_conv(mid_net, 128, kernel_size=[7, 7],is_training=iftrain, scope='mid_conv2')
                mid_net = dethwise_conv(mid_net, 128, kernel_size=[7, 7], is_training=iftrain,scope='mid_conv3')
                mid_net = dethwise_conv(mid_net, 128, kernel_size=[7, 7], is_training=iftrain,scope='mid_conv4')
                mid_net = dethwise_conv(mid_net, 128, kernel_size=[7, 7], is_training=iftrain,scope='mid_conv5')
                mid_net = slim.conv2d(mid_net, 128, [1, 1],activation_fn=tf.nn.relu, scope='mid_conv6')
                self.current_heatmap = slim.conv2d(mid_net, self.joints, [1, 1],activation_fn=None,
                                                   scope='mid_conv7')
                self.stage_heatmap.append(self.current_heatmap)
                
    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step,optimizer='Adam'):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer=optimizer

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage+1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap[:,:,:,:24],
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage+1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total_loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.cur_lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning_rate', self.cur_lr)
            
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.cur_lr)
            self.train_step = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                self.total_loss = control_flow_ops.with_dependencies([updates], self.total_loss)
        self.merged_summary = tf.summary.merge_all()


# In[5]:

x=tf.placeholder(tf.float32,[None,256,256,3])
y_=tf.placeholder(tf.float32,[None,32,32,25])
# phase = tf.placeholder(tf.bool, name='phase')
model=CPM_Model(3,24)
WHITE=[255,255,255]


# In[6]:

model.build_model(x,phase)
model.build_loss(y_,0.001,0.1,10000,optimizer='RMSProp')
merged_summary=model.merged_summary
min_loss=10000
label_test=[]
with tf.Session() as sess:
#     train_writer=tf.summary.FileWriter('/workspace/TC/Graph/blouse/train/',sess.graph)
#     test_writer = tf.summary.FileWriter('/workspace/TC/Graph/blouse/test/', sess.graph)
    saver=tf.train.Saver()
    
    model_file=tf.train.latest_checkpoint('F:/data/TC/model/trousers/')
    saver.restore(sess,model_file)
    
    for i,f in enumerate(image_name):
        if i<100:
            
                x0=cv2.imread(inpath+f)
                w,h,_=x0.shape
                if w!=512 or h!=512:
                    a=512-w
                    b=512-h
                    if a % 2==0:
                        a1=a2=a//2
                    else:
                        a1=a//2
                        a2=a1+1
                    if b % 2==0:
                        b1=b2=b//2
                    else:
                        b1=b//2
                        b2=b1+1
                    x1=cv2.copyMakeBorder(x0,a1,a2,b1,b2,cv2.BORDER_CONSTANT,value=WHITE)
                else:
                    a1=b1=0
                    x1=x0
                x2=cv2.resize(x1,(256,256))
                x4=cv2.flip(x2,1)
                x3=cv2.cvtColor(x0,cv2.COLOR_BGR2RGB)
                x4=x4[np.newaxis,:,:,:]/255.0-0.5

                x2=x2[np.newaxis,:,:,:]/255.0-0.5

                stage_heatmap_np= sess.run(model.stage_heatmap,
                                             feed_dict={x:x2})
                stage_heatmap_np1= sess.run(model.stage_heatmap,
                                             feed_dict={x:x4})

        #     print stage_heatmap_np[0].shape
                l_t=[]
                heatmap1=stage_heatmap_np[0]+stage_heatmap_np[1]+stage_heatmap_np[2]
                heatmap2=stage_heatmap_np1[0]+stage_heatmap_np1[1]+stage_heatmap_np1[2]
                heatmap2=cv2.flip(heatmap2[0],1)
                heatmap=heatmap1[0]+heatmap2
                pmap=cv2.resize(heatmap[:,:,:24],(256,256))
                pt=np.zeros((24,2))
#                 d=np.sqrt(np.sum(np.square(pts[5,:2]-pts[6,:2])))
                for j in range(24):
                    tmp_map=pmap[:,:,j]
                    coord=np.unravel_index(np.argmax(tmp_map),(256,256))
                    coord = np.array(coord).astype(np.int32)
                    pt[j,:]=[coord[1],coord[0]]
                pt=pt.astype(np.int32) 
                pt=pt*2-[b1,a1]
                for k in range(24):
                    l_t.append(str(pt[k,0])+'_'+str(pt[k,1])+'_1')
                label_test.append(l_t)

                
                for k in range(15,17):
                    cv2.circle(x0,center=(pt[k,0],pt[k,1]),radius=3,color=[255,0,0], thickness=-1)
                for k in range(19,24):
                    cv2.circle(x0,center=(pt[k,0],pt[k,1]),radius=3,color=[255,0,0], thickness=-1)
               
                cv2.imwrite('F:/data/TC/out/'+str(i)+'.jpg',x0)
               
                if i==0:
                    print(pt)

                    
                    
         


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



