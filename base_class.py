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
from .bn import ComplexBatchNormalization

# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def sign(x):
    return tf.gradients(tf.nn.relu(x),x)[0]
    # e = 0.1**8
    # return tf.nn.relu(x)/(tf.nn.relu(x)+e)
    # x = (x>0)
    # x = tf.cast(x,tf.float32)
    # return x


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



class base_class(object):
    """alexNet model"""
    def __init__(self, x, classNum, seed,is_training,is_complex,modelPath = "alexnet"):
        self.X_com = [x,x]
        self.X = x
        self.is_complex=is_complex
        self.CLASSNUM = classNum
        self.MODELPATH = modelPath
        self.training = is_training
        tf.set_random_seed(seed)  
        self.seed = seed
        #build CNN
        # self.buildCNN()
        # self.build_complex_CNN()

    def Learnable_angle_relu(self,C,name):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('alpha'):
                alpha = tf.get_variable("alpha",shape = [1],dtype=tf.float32)
            tf.summary.histogram('alpha',alpha)
            with tf.variable_scope('beita'):
                beita = tf.get_variable("beita",shape = [1],dtype=tf.float32)
            tf.summary.histogram('beita',beita)

            activations= [C[0]*sign(tf.atan(C[1]/C[0])-alpha)*sign(alpha+beita-tf.atan(C[1]/C[0])),\
            C[1]*sign(tf.atan(C[1]/C[0])-alpha)*sign(alpha+beita-tf.atan(C[1]/C[0]))]

            tf.summary.histogram('activations', activations)
            return activations
    
    def Learnable_angle_relu_per_neural(self,C,name,seed=None):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('alpha'):
                alpha = tf.get_variable("alpha",shape = C[0].get_shape()[1:].as_list(),dtype=tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
            variable_summaries(alpha)
            with  tf.variable_scope('beita'):
                beita = tf.get_variable("beita",shape = C[0].get_shape()[1:].as_list(),dtype=tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
            variable_summaries(beita)
            activations =  [C[0]*sign(tf.atan(C[1]/C[0])-alpha)*sign(alpha+beita-tf.atan(C[1]/C[0])),\
            C[1]*sign(tf.atan(C[1]/C[0])-alpha)*sign(alpha+beita-tf.atan(C[1]/C[0]))]
            tf.summary.histogram('activations', activations)
            return activations

    def Learnable_radius_relu(self,C,name):
        with tf.variable_scope(name) as scope:
            with  tf.variable_scope('radius'):
                radius = tf.get_variable("radius",shape = [1],dtype=tf.float32)
            # tf.summary.histogram('radius',radius)
            activations= [C[0]*sign(tf.sqrt(C[0]**2+C[1]**2)-radius),C[1]*sign(tf.sqrt(C[0]**2+C[1]**2)-radius)]
            tf.summary.histogram('activations', activations)
            return activations

    def Learnable_radius_relu_per_neural(self,C,name,seed=None):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('radius'):
                radius = tf.get_variable("radius",shape = C[0].get_shape().as_list()[1:],dtype=tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
            variable_summaries(radius)
            activations= [C[0]*sign(tf.sqrt(C[0]**2+C[1]**2)-radius),C[1]*sign(tf.sqrt(C[0]**2+C[1]**2)-radius)]
            tf.summary.histogram('activations', activations)
            return activations

    def complex_batch_normalization(self,C):
        R,I = C
        R,I = ComplexBatchNormalization(R,I,is_training=self.training)
        return [R,I]

    def complex_fcLayer(self,x, input_size, output_size, name,seed = None,norm=True, relu_fun =tf.nn.relu):
        """fully-connect"""
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('wightr'):
                wr = tf.get_variable("wr", shape = [input_size, output_size], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
                # variable_summaries(wr)
            with tf.variable_scope('wighti'):
                wi = tf.get_variable("wi", shape = [input_size, output_size], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
                # variable_summaries(wi)
            with tf.variable_scope('biasr'):
                br = tf.get_variable("br", [output_size], dtype = tf.float32,\
                    initializer=tf.zeros_initializer())
                # variable_summaries(br)
            with tf.variable_scope('biasi'):
                bi = tf.get_variable("bi", [output_size], dtype = tf.float32,\
                initializer=tf.zeros_initializer())
                # variable_summaries(bi)
            R = tf.nn.xw_plus_b(x[0], wr, br)- tf.nn.xw_plus_b(x[1], wi, bi)
            I = tf.nn.xw_plus_b(x[0], wi, bi)+ tf.nn.xw_plus_b(x[1], wr, br)
            # tf.summary.histogram('R',R)
            # tf.summary.histogram('I',I)
            if norm :
                R,I = self.complex_batch_normalization([R,I])
            # tf.summary.histogram('normR',R)
            # tf.summary.histogram('normI',I)            
            if relu_fun == tf.nn.relu:
                R,I=relu_fun(R,'reluR'),relu_fun(I,'reluI')
                # tf.summary.histogram('fcR',R)
                # tf.summary.histogram('fcI',I)                 
                return [R,I]
            relu =  relu_fun([R,I],scope)
            tf.summary.histogram('fcR',relu[0])
            tf.summary.histogram('fcI',relu[1])            
            return relu

    def complex_convLayer(self,x, ksize, strides,out_channel, name, padding = "SAME",norm=True,seed=None,relu_fun = tf.nn.relu): 
        """convolution"""
        in_channel = int(x[0].get_shape()[-1])
        conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1] +strides +[ 1], padding = padding)

        with tf.variable_scope(name) as scope:
            with tf.variable_scope('wightr'):
                wr = tf.get_variable("wr", shape = ksize+[in_channel,out_channel],dtype=tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
                # variable_summaries(wr)
            with tf.variable_scope('wighti'):
                wi = tf.get_variable("wi", shape = ksize+[in_channel,out_channel],dtype=tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
                # variable_summaries(wi)
            with tf.variable_scope('biasr'):
                br = tf.get_variable("br", shape = [out_channel],dtype=tf.float32,\
                    initializer=tf.zeros_initializer())
                # variable_summaries(br)
            with tf.variable_scope('biasi'):
                bi = tf.get_variable("bi", shape = [out_channel],dtype=tf.float32,\
                    initializer=tf.zeros_initializer())
                # variable_summaries(bi)

            R = conv(x[0],wr)-conv(x[1],wi) +br
            I= conv(x[1],wr)+conv(x[0],wi) +bi
            # tf.summary.histogram('R',R)
            # tf.summary.histogram('I',I)
            if norm:
                R,I=self.complex_batch_normalization([R,I])
                # tf.summary.histogram('normR',R)
                # tf.summary.histogram('normI',I)  
            if relu_fun == tf.nn.relu:
                R,I = relu_fun(R,'reluR'),relu_fun(I,'reluI')
                # tf.summary.histogram('convR',R)
                # tf.summary.histogram('convI',I)                
                return [R,I]
            relu =  relu_fun([R,I],scope)
            tf.summary.histogram('convR',relu[0])
            tf.summary.histogram('convI',relu[1])            
            return relu



    def complex_maxPoolLayer(self,x, ksize,strides=[1,1], name='None', padding = "SAME"):
        """max-pooling"""
        with tf.variable_scope(name):
            activations= [tf.nn.max_pool(x[0], ksize =[1]+ ksize+[1],
                                  strides = [1] +strides+[1], padding = padding, name = name),tf.nn.max_pool(x[1], ksize =[1]+ ksize+[1],
                                  strides = [1] +strides+[1], padding = padding, name = name)]
            tf.summary.histogram('poolR',activations[0])
            tf.summary.histogram('poolI',activations[1])
            return activations
 

    def fcLayer(self,x, input_size, output_size, name,norm=True,seed=None, relu_fun = tf.nn.relu):
        """fully-connect"""
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('wight'):
                w = tf.get_variable("w", shape = [input_size, output_size], dtype = "float",\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
                # variable_summaries(w)
            with tf.variable_scope('bias'):
                b = tf.get_variable("b", [output_size], dtype = "float",\
                initializer=tf.zeros_initializer())
                # variable_summaries(b)
            out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
            # tf.summary.histogram('xw+b',out)
            if norm:
                out=tf.layers.batch_normalization(out,training=self.training)
                # tf.summary.histogram('norm',out)
            activations= relu_fun(out,'relu')
            tf.summary.histogram('fc',activations)
            return activations

    def convLayer(self,x, ksize, strides,out_channel, name, padding = "SAME",seed=None,relu_fun = tf.nn.relu): 
        """convolution"""
        in_channel = int(x.get_shape()[-1])

        conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1] +strides +[ 1], padding = padding)

        with tf.variable_scope(name) as scope:
            with tf.variable_scope('wight'):
                w = tf.get_variable("w", shape = ksize+[in_channel,out_channel],\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, seed=seed,dtype=tf.float32))
                # variable_summaries(w)
            with tf.variable_scope('bias'):
                b = tf.get_variable("b", shape = [out_channel],\
                    initializer=tf.zeros_initializer())
                # variable_summaries(b)
            out_put = conv(x,w)
            # tf.summary.histogram('convout',out_put)
            # print mergeFeatureMap.shape
            out = tf.nn.bias_add(out_put, b)
            out = tf.layers.batch_normalization(out,training=self.training)
            # tf.summary.histogram('norm',out)
            relu =  relu_fun(out, name = scope.name)
            tf.summary.histogram('conv',relu)
            return relu

    def maxPoolLayer(self,x, ksize,strides=[1,1], name='None', padding = "SAME"):
        with tf.variable_scope(name):
            """max-pooling"""
            activations= tf.nn.max_pool(x, ksize =[1]+ ksize+[1],
                            strides = [1] +strides+[1], padding = padding, name = name)
            tf.summary.histogram('pool',activations)
            return activations


