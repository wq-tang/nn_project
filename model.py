# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: alexnet.py
   create time: 2018年09月5日 
   author: wenqi Tang

'''''''''''''''''''''''''''''''''''''''''''''''''''''
# based on Frederik Kratzert's alexNet with tensorflow
import tensorflow as tf
import numpy as np
from functools import reduce

# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def maxPoolLayer(x, ksize,strides=[1,1], name='None', padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize =[1]+ ksize+[1],
                          strides = [1] +strides+[1], padding = padding, name = name)



def fcLayer(x, input_size, output_size, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            w = tf.get_variable("w", shape = [input_size, output_size], dtype = "float")
            b = tf.get_variable("b", [output_size], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, ksize, strides,out_channel, name, padding = "SAME"): 
    """convolution"""
    in_channel = int(x.get_shape()[-1])

    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1] +strides +[ 1], padding = padding)

    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            w = tf.get_variable("w", shape = ksize+[in_channel,out_channel])
            b = tf.get_variable("b", shape = [out_channel])
        out_put = conv(x,w)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(out_put, b)
        return tf.nn.relu(out, name = scope.name)

class alexNet(object):
    """alexNet model"""
    def __init__(self, x, classNum, seed,skip=None, modelPath = "alexnet"):
        self.X = x
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        self.training = True
        tf.set_random_seed(seed)  
        self.seed = seed
        #build CNN
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        with tf.variable_scope('model_%d'%self.seed):
            conv1 = convLayer(self.X, [5, 5], [1, 1], 128, "conv1", "SAME")
            pool1 = maxPoolLayer(conv1,[3, 3],[ 1,1], "pool1", "SAME")
            norm_pool1=tf.layers.batch_normalization(pool1,training=self.training)

            conv2 = convLayer(norm_pool1, [3, 3], [1, 1], 64, "conv2",'SAME')
            pool2 = maxPoolLayer(conv2,[3, 3], [1, 1], "pool2", "SAME")
            norm_pool2=tf.layers.batch_normalization(pool2,training=self.training)

            conv3 = convLayer(norm_pool2, [5, 5], [1, 1], 64, "conv3",'VALID')
            pool3 = maxPoolLayer(conv3, [3, 3], [2, 2], "pool3", "VALID")
            norm_pool3=tf.layers.batch_normalization(pool3,training=self.training)

            conv4 = convLayer(norm_pool3, [3, 3], [1, 1], 64, "conv4",'VALID')
            pool4 = maxPoolLayer(conv4, [3, 3], [2, 2], "pool4", "VALID")


            shapes = pool4.get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            
            reshape = tf.reshape(pool4,[-1,mul])
            dim = reshape.get_shape()[1].value

            norm_reshape=tf.layers.batch_normalization(reshape,training=self.training)
            fc1 = fcLayer(norm_reshape, dim, 512, reluFlag=True, name = "fc4")

            norm_fc1=tf.layers.batch_normalization(fc1,training=self.training)
            fc2 = fcLayer(norm_fc1, 512, 128, reluFlag=True,name =  "fc5")

            norm_fc2=tf.layers.batch_normalization(fc2,training=self.training)
            self.fc3 = fcLayer(norm_fc2, 128, self.CLASSNUM, reluFlag=True,name =  "fc6")


class attention(object):
    """alexNet model"""
    def __init__(self, x, seed):
        self.X = x
        self.training = True
        tf.set_random_seed(seed)  
        self.seed = seed
        #build CNN
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        with tf.variable_scope('attention_%d'%self.seed):
            out_channel = int(self.X.get_shape()[-1])
            conv1 = convLayer(self.X, [5, 5], [1, 1], 80, "conv1", "SAME")
            pool1 = maxPoolLayer(conv1,[3, 3],[ 1,1], "pool1", "SAME")
            norm_pool1=tf.layers.batch_normalization(pool1,training=self.training)

            conv2 = convLayer(norm_pool1, [3, 3], [1, 1], 24, "conv2",'SAME')
            pool2 = maxPoolLayer(conv2,[3, 3], [1, 1], "pool2", "SAME")
            norm_pool2=tf.layers.batch_normalization(pool2,training=self.training)

            conv3 = convLayer(norm_pool2, [3, 3], [1, 1], 3, "conv3",'SAME')
            pool3 = maxPoolLayer(conv3, [3, 3], [1, 1], "pool3", "SAME")
            norm_pool3=tf.layers.batch_normalization(pool3,training=self.training)

            self.attention = tf.nn.sigmoid(norm_pool3)


class angle_net():
    """alexNet model"""
    def __init__(self, x, classNum, seed):
        self.X = x
        self.CLASSNUM = classNum
        self.training = True
        tf.set_random_seed(seed)  
        self.seed = seed
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        with tf.variable_scope('angle_%d'%(self.seed+10)):
            conv1 = convLayer(self.X, [5, 5], [1, 1], 128, "conv1", "SAME")
            pool1 = maxPoolLayer(conv1,[3, 3],[ 1,1], "pool1", "SAME")
            norm_pool1=tf.layers.batch_normalization(pool1,training=self.training)

            conv2 = convLayer(norm_pool1, [3, 3], [1, 1], 64, "conv2",'SAME')
            pool2 = maxPoolLayer(conv2,[3, 3], [1, 1], "pool2", "SAME")
            norm_pool2=tf.layers.batch_normalization(pool2,training=self.training)

            conv3 = convLayer(norm_pool2, [5, 5], [1, 1], 64, "conv3",'VALID')
            pool3 = maxPoolLayer(conv3, [3, 3], [2, 2], "pool3", "VALID")
            norm_pool3=tf.layers.batch_normalization(pool3,training=self.training)

            conv4 = convLayer(norm_pool3, [3, 3], [1, 1], 64, "conv4",'VALID')
            pool4 = maxPoolLayer(conv4, [3, 3], [2, 2], "pool4", "VALID")


            shapes = pool4.get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            
            reshape = tf.reshape(pool4,[-1,mul])
            dim = reshape.get_shape()[1].value

            norm_reshape=tf.layers.batch_normalization(reshape,training=self.training)
            fc1 = fcLayer(norm_reshape, dim, 512, reluFlag=True, name = "fc4")

            norm_fc1=tf.layers.batch_normalization(fc1,training=self.training)
            fc2 = fcLayer(norm_fc1, 512, 128, reluFlag=True,name =  "fc5")

            norm_fc2=tf.layers.batch_normalization(fc2,training=self.training)
            self.fc3 = fcLayer(norm_fc2, 128, self.CLASSNUM, reluFlag=True,name =  "fc6")
