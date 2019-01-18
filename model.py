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
def sign(x):
    e = 0.1**8
    return tf.nn.relu(x)/(tf.nn.relu(x)+e)

def maxPoolLayer(x, ksize,strides=[1,1], name='None', padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize =[1]+ ksize+[1],
                          strides = [1] +strides+[1], padding = padding, name = name)





def myconvLayer(x, ksize, strides,out_channel, name, padding = "SAME"): 
    """convolution"""
    in_channel = int(x.get_shape()[-1])

    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1] +strides +[ 1], padding = padding)

    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = ksize+[in_channel,out_channel])
        wT = tf.transpose(w,perm= [1,0,2,3],name='transpose_w')

        pi = np.ones([in_channel,out_channel] +ksize)*np.pi
        pi = np.triu(pi,1)
        pi = np.transpose(pi,[2,3,0,1])
        pi = tf.constant(pi,shape = [in_channel,out_channel] +ksize,name = 'pi')
        b = tf.get_variable("b", shape = [out_channel])
        cos = tf.cos(w+wT+pi)
        sin = tf.sin(w+wT+pi)
        out_cos = tf.square(conv(x,cos))
        out_sin = tf.square(conv(x,sin))
        out_put = tf.sqrt(out_sin+out_cos)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(out_put, b)
        return tf.nn.relu(out, name = scope.name)
        



class alexNet(object):
    """alexNet model"""
    def __init__(self, x, classNum, seed,skip=None, modelPath = "alexnet"):
        self.X_com = [x,x*0]
        self.X = x
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        self.training = True
        tf.set_random_seed(seed)  
        self.seed = seed
        #build CNN
        self.buildCNN()
        # self.build_complex_CNN()

    def build_complex_CNN(self):
        """build model"""
        with tf.variable_scope('model_%d'%self.seed):
            conv1 = complex_convLayer(self.X_com, [5, 5], [1, 1], 128, "conv1", "SAME")
            pool1 = complex_maxPoolLayer(conv1,[3, 3],[ 1,1], "pool1", "SAME")

            conv2 = complex_convLayer(pool1, [3, 3], [1, 1], 64, "conv2",'SAME')
            pool2 = complex_maxPoolLayer(conv2,[3, 3], [1, 1], "pool2", "SAME")

            conv3 = complex_convLayer(pool2, [5, 5], [1, 1], 64, "conv3",'VALID')
            pool3 = complex_maxPoolLayer(conv3, [3, 3], [2, 2], "pool3", "VALID")

            conv4 = complex_convLayer(pool3, [3, 3], [1, 1], 64, "conv4",'VALID')
            pool4 = complex_maxPoolLayer(conv4, [3, 3], [2, 2], "pool4", "VALID")


            shapes = pool4[0].get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            
            R = tf.reshape(pool4[0],[-1,mul])
            I = tf.reshape(pool4[1],[-1,mul])
            dim = R.get_shape()[1].value
            self.real_fc(R,I,dim)
            # self.complex_fc(R,I,dim)

    def real_fc(self,R,I,dim):
        reshape = tf.sqrt(tf.square(R)+tf.square(I))
        fc1 = fcLayer(reshape, dim, 512, reluFlag=True, name = "fc4")
        fc2 = fcLayer(fc1, 512, 128, reluFlag=True,name =  "fc5")
        self.fc3 = fcLayer(fc2, 128, self.CLASSNUM, reluFlag=True,name =  "fc6",norm=False)
            
    def complex_fc(self,R,I,dim):
        fc1 = complex_fcLayer([R,I], dim, 512, reluFlag=True, name = "fc4")
        fc2 = complex_fcLayer(fc1, 512, 128, reluFlag=True,name =  "fc5")
        fc3 = complex_fcLayer(fc2, 128, self.CLASSNUM, reluFlag=True,name =  "fc6",norm=False)
        self.fc3 = tf.sqrt(tf.square(fc3[0])+tf.square(fc3[1]))

    def Learnable_relu(self,C,name):
        with tf.variable_scope(name) as scope:
            alpha = tf.get_variable("alpha",shape = [1],dtype=tf.float32)
            beita = tf.get_variable("beita",shape = [1],dtype=tf.float32)
            return [C[0]*sign(tf.atan(C[1]/C[0]-alpha))*sign(alpha+beita-tf.atan(C[1]/C[0]-alpha)),
            C[1]*sign(tf.atan(C[1]/C[0]-alpha))*sign(alpha+beita-tf.atan(C[1]/C[0]-alpha))]





    def complex_fcLayer(self,x, input_size, output_size, reluFlag, name,norm=True):
        """fully-connect"""
        with tf.variable_scope(name) as scope:
            wr = tf.get_variable("wr", shape = [input_size, output_size], dtype = tf.float32)
            wi = tf.get_variable("wi", shape = [input_size, output_size], dtype = tf.float32)
            br = tf.get_variable("br", [output_size], dtype = tf.float32)
            bi = tf.get_variable("bi", [output_size], dtype = tf.float32)
            R = tf.nn.xw_plus_b(x[0], wr, br)- tf.nn.xw_plus_b(x[1], wi, bi)
            I = tf.nn.xw_plus_b(x[0], wi, bi)+ tf.nn.xw_plus_b(x[1], wr, br)
            if norm :
                R=tf.layers.batch_normalization(R,training=self.training)
                I=tf.layers.batch_normalization(I,training=self.training)
            if reluFlag:
            #     f = tf.complex(R,I)
            #     f = tf.tanh(f)
            #     return [tf.real(f),tf.imag(f)]
                return [tf.nn.relu(R*I),tf.nn.relu(R*I)]
            else:
                Z = tf.complex(R,I)
                Z=1/Z
                return [tf.real(Z),tf.imag(Z)]
    def complex_convLayer(self,x, ksize, strides,out_channel, name, padding = "SAME",act_flag=False): 
        """convolution"""
        in_channel = int(x[0].get_shape()[-1])

        conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1] +strides +[ 1], padding = padding)

        with tf.variable_scope(name) as scope:
            wr = tf.get_variable("wr", shape = ksize+[in_channel,out_channel],dtype=tf.float32)
            wi = tf.get_variable("wi", shape = ksize+[in_channel,out_channel],dtype=tf.float32)

            br = tf.get_variable("br", shape = [out_channel],dtype=tf.float32)
            bi = tf.get_variable("bi", shape = [out_channel],dtype=tf.float32)

            R = conv(x[0],wr)-conv(x[1],wi) +br
            I= conv(x[1],wr)+conv(x[0],wi) +bi
            R=tf.layers.batch_normalization(R,training=self.training)
            I=tf.layers.batch_normalization(I,training=self.training)
            if act_flag:
                Z = tf.complex(R,I)
                Z=1/Z
                return [tf.real(Z),tf.imag(Z)]
            else:
                return Learnable_relu([R,I],'relu')

            # print mergeFeatureMap.shape
            # return [tf.nn.relu(R),tf.nn.relu(I)]

    def complex_maxPoolLayer(self,x, ksize,strides=[1,1], name='None', padding = "SAME"):
        """max-pooling"""
        return [tf.nn.max_pool(x[0], ksize =[1]+ ksize+[1],
                              strides = [1] +strides+[1], padding = padding, name = name+'0'),tf.nn.max_pool(x[1], ksize =[1]+ ksize+[1],
                              strides = [1] +strides+[1], padding = padding, name = name)]


    def fcLayer(self,x, input_size, output_size, reluFlag, name,norm=True):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [input_size, output_size], dtype = "float")
        b = tf.get_variable("b", [output_size], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if norm:
            out=tf.layers.batch_normalization(out,training=self.training)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

    def convLayer(self,x, ksize, strides,out_channel, name, padding = "SAME"): 
        """convolution"""
        in_channel = int(x.get_shape()[-1])

        conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1] +strides +[ 1], padding = padding)

        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape = ksize+[in_channel,out_channel])
            b = tf.get_variable("b", shape = [out_channel])
            out_put = conv(x,w)
            # print mergeFeatureMap.shape
            out = tf.nn.bias_add(out_put, b)
            out = tf.layers.batch_normalization(out,training=self.training)
            return tf.nn.relu(out, name = scope.name)








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
        with tf.variable_scope('model_%d'%self.seed):
            out_channel = int(self.X.get_shape()[-1])
            conv1 = convLayer(self.X, [5, 5], [1, 1], 100, "conv1", "SAME")
            pool1 = maxPoolLayer(conv1,[3, 3],[ 1,1], "pool1", "SAME")
            norm_pool1=tf.layers.batch_normalization(pool1,training=self.training)

            conv2 = convLayer(norm_pool1, [3, 3], [1, 1], 64, "conv2",'SAME')
            pool2 = maxPoolLayer(conv2,[3, 3], [1, 1], "pool2", "SAME")
            norm_pool2=tf.layers.batch_normalization(pool2,training=self.training)

            conv3 = convLayer(norm_pool2, [3, 3], [1, 1], 16, "conv3",'VALID')
            pool3 = maxPoolLayer(conv3, [3, 3], [1, 1], "pool3", "VALID")
            norm_pool3=tf.layers.batch_normalization(pool3,training=self.training)

            conv4 = convLayer(norm_pool3, [3, 3], [1, 1], out_channel, "conv4",'VALID')
            self.attention = maxPoolLayer(conv4, [3, 3], [1, 1], "pool4", "VALID")



