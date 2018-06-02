
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import time
import os
slim = tf.contrib.slim
from tensorflow.python.ops import control_flow_ops


# In[ ]:


def dethwise_conv(inputs,num,kernel_size=[3,3],is_training='True',scope=''):
    with slim.arg_scope([slim.conv2d,slim.separable_conv2d],padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95}):
            net=slim.separable_conv2d(inputs,num_outputs=None,depth_multiplier=1,kernel_size=kernel_size,scope=scope+'_dw')
            net=slim.conv2d(net,num,kernel_size=[1,1],scope=scope)
    return net


# In[ ]:

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
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap[:,:,:,:13],
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



# In[ ]:

def print_current_training_stats(global_step, cur_lr, total_loss,total_loss1, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, 300000,
                                                                                 cur_lr, time_elapsed)
    print(stats)
    print('Training total_loss: {:>7.2f}  Testing total_loss:{:>7.2f}'.format(total_loss,total_loss1))



# In[ ]:

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [32, 32, 14])
    return img,label


# In[ ]:

inpath='TC/tfrecord/blouse1/train/'
files = [f for f in os.listdir(inpath) ]
file_name=[inpath+fs for fs in files]
img, label = read_and_decode(file_name)

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=16, capacity=1000,
                                                min_after_dequeue=500)

img_test,label_test=read_and_decode(["TC/tfrecord/blouse/val/blouse1.tfrecords"])
img_tbatch, label_tbatch = tf.train.shuffle_batch([img_test, label_test],
                                                batch_size=16, capacity=1000,
                                                min_after_dequeue=500)


# In[ ]:

x=tf.placeholder(tf.float32,[None,256,256,3])
y_=tf.placeholder(tf.float32,[None,32,32,14])
phase = tf.placeholder(tf.bool, name='phase')
model=CPM_Model(3,13)


# In[ ]:


model.build_model(x,phase)
model.build_loss(y_,0.001,0.1,20000,optimizer='RMSProp')
merged_summary=model.merged_summary
min_loss=10000
with tf.Session() as sess:
    train_writer=tf.summary.FileWriter('TC/Graph/blouse1/train/',sess.graph)
    test_writer = tf.summary.FileWriter('TC/Graph/blouse1/test/', sess.graph)
    saver=tf.train.Saver(max_to_keep=3)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    threads = tf.train.start_queue_runners(sess=sess)
    for training_itr in range(300000):
        
        batch_x_np, batch_gt_heatmap_np = sess.run([img_batch,label_batch])
        _, summaries, current_lr,         stage_heatmap_np, global_step = sess.run([model.train_step,
                                                      merged_summary,
                                                      model.cur_lr,
                                                      model.stage_heatmap,
                                                      model.global_step
                                                      ],feed_dict={x: batch_x_np,y_: batch_gt_heatmap_np,phase:True})
        
        total_loss_np=sess.run(model.total_loss,feed_dict={x: batch_x_np,y_: batch_gt_heatmap_np,phase:True})
        t1=time.time()
        sess.run(model.stage_heatmap,feed_dict={x: batch_x_np,phase:False})
        t2=time.time()-t1
        train_writer.add_summary(summaries, global_step)
        if training_itr %5==0:
#             saver.save(sess=sess, save_path='model/hand_landmark_v6.1_model/model.ckpt',global_step=(global_step + 1))
            mean_val_loss = 0
            cnt = 0
            while cnt < 30:
                x_test,y_test=sess.run([img_tbatch,label_tbatch])
                total_loss_np1, summaries1 = sess.run([model.total_loss, merged_summary],
                                                        feed_dict={x: x_test,y_:y_test,phase:False})
                mean_val_loss += total_loss_np1
                cnt += 1
            print_current_training_stats(global_step, current_lr, total_loss_np,mean_val_loss / cnt, t2)
            test_writer.add_summary(summaries1, global_step)
            if mean_val_loss / cnt < min_loss:
                min_loss=mean_val_loss/cnt
                saver.save(sess=sess, save_path='TC/model/blouse1/model.ckpt',global_step=(global_step + 1))


