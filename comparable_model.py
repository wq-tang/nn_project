import tensorflow as tf
import numpy as np
from model import alexNet
from functools import reduce


def sign(x):
    e = 0.1**8
    return tf.nn.relu(x)/(tf.nn.relu(x)+e)

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


class complex_net(alexNet):
    """docstring for complex_net"""
    def __init__(self, x, classNum, seed,modelPath = "complexnet"):
        super(complex_net,self).__init__(x, classNum, seed,modelPath)
        tf.set_random_seed(seed)
        self.relu_fun = tf.nn.relu
        # self.relu_fun = self.Learnable_angle_relu
        self.build_real_CNN_for_mnist()


    def build_complex_CNN_for_mnist(self):
        with tf.variable_scope('model_%d'%self.seed):
            conv1 = self.complex_convLayer(self.X_com, [5, 5], [1, 1], 16, "conv1", "SAME",seed=1+self.seed,relu_fun = self.relu_fun)
            pool1 = self.complex_maxPoolLayer(conv1,[2, 2],[ 2,2], "pool1", "SAME")

            conv2 = self.complex_convLayer(pool1, [3, 3], [1, 1], 8, "conv2",'SAME',seed=2+self.seed,relu_fun = self.relu_fun)
            pool2 = self.complex_maxPoolLayer(conv2,[2, 2], [2, 2], "pool2", "SAME")

            # conv3 = self.complex_convLayer(pool2, [2, 2], [1, 1], 256, "conv3",'VALID',relu_fun = self.relu_fun)
            # pool3 = self.complex_maxPoolLayer(conv3, [2, 2], [2, 2], "pool3", "VALID")
            cnnout = pool2
            shapes = cnnout[0].get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            R = tf.reshape(cnnout[0],[-1,mul])
            I = tf.reshape(cnnout[1],[-1,mul])
            dim = R.get_shape()[1].value

            fc1 = self.complex_fcLayer([R,I], dim, 30,  name = "fc4",seed=101+self.seed,relu_fun = tf.nn.relu)
            self.fc2 = self.complex_fcLayer(fc1, 30, self.CLASSNUM,name =  "fc5",seed=102+self.seed,relu_fun = tf.nn.relu)
            self.out = self.fc2
            self.out = tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))

    def build_real_CNN_for_mnist(self):
        with tf.variable_scope('model_%d'%self.seed):
            conv1 = self.convLayer(self.X, [5, 5], [1, 1], int(16*1.41)+1, "conv1", "SAME",seed=3+self.seed)
            pool1 = self.maxPoolLayer(conv1,[2, 2],[ 2,2], "pool1", "SAME")

            conv2 = self.convLayer(pool1, [3, 3], [1, 1], int(8*1.41)+1, "conv2",'SAME',seed=4+self.seed)
            pool2 = self.maxPoolLayer(conv2,[2, 2], [2, 2], "pool2", "SAME")

            # conv3 = self.convLayer(pool2, [2, 2], [1, 1], int(256*1.41)+1, "conv3",'VALID')
            # pool3 = self.maxPoolLayer(conv3, [2, 2], [2, 2], "pool3", "VALID")
            cnnout = pool2
            shapes = cnnout.get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            Res = tf.reshape(cnnout,[-1,mul])
            fc1 = self.fcLayer(Res, mul, int(30*1.41)+1,  name = "fc4",seed=103+self.seed)
            self.fc2 = self.fcLayer(fc1, int(30*1.41)+1, self.CLASSNUM, name =  "fc5",seed=104+self.seed)
            self.out = self.fc2



    def build_complex_CNN_for_cifar10(self):
        with tf.variable_scope('model_%d'%self.seed):
            conv1 = self.complex_convLayer(self.X_com, [5, 5], [1, 1], 128, "conv1", "SAME",seed=5+self.seed,relu_fun = self.relu_fun)
            pool1 = self.complex_maxPoolLayer(conv1,[2, 2],[ 2,2], "pool1", "SAME")

            conv2 = self.complex_convLayer(pool1, [3, 3], [1, 1], 64, "conv2",'SAME',seed=6+self.seed,relu_fun = self.relu_fun)
            pool2 = self.complex_maxPoolLayer(conv2,[2, 2], [2, 2], "pool2", "SAME")

            conv3 = self.complex_convLayer(pool2, [3, 3], [1, 1], 64, "conv3",'SAME',seed=7+self.seed,relu_fun = self.relu_fun)
            pool3 = self.complex_maxPoolLayer(conv3, [2, 2], [2, 2], "pool3", "SAME")

            cnnout = pool2
            shapes = cnnout[0].get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            R = tf.reshape(cnnout[0],[-1,mul])
            I = tf.reshape(cnnout[1],[-1,mul])
            dim = R.get_shape()[1].value
            fc1 = self.complex_fcLayer([R,I], dim, 384,  name = "fc4",seed=105+self.seed,relu_fun = tf.nn.relu)
            fc2 = self.complex_fcLayer(fc1, 384, 192, name =  "fc5",seed=106+self.seed,relu_fun = tf.nn.relu)
            fc3 = self.complex_fcLayer(fc2, 192, self.CLASSNUM, name =  "fc6",norm=False,seed=107+self.seed,relu_fun = tf.nn.relu)
            self.out = fc3
            self.out = tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))


    def build_real_CNN_for_cifar10(self):
        with tf.variable_scope('model_%d'%self.seed):
            conv1 = self.convLayer(self.X, [5, 5], [1, 1], int(128*1.41)+1, "conv1", "SAME",seed=8+self.seed)
            pool1 = self.maxPoolLayer(conv1,[2, 2],[ 2,2], "pool1", "SAME")

            conv2 = self.convLayer(pool1, [3, 3], [1, 1], int(64*1.41)+1, "conv2",'SAME',seed=9+self.seed)
            pool2 = self.maxPoolLayer(conv2,[2, 2], [2, 2], "pool2", "SAME")

            conv3 = self.convLayer(pool2, [3, 3], [1, 1], int(64*1.41)+1, "conv3",'SAME',seed=10+self.seed)
            pool3 = self.maxPoolLayer(conv3, [2, 2], [2, 2], "pool3", "SAME")
            cnnout = pool2
            shapes = cnnout.get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            Res = tf.reshape(cnnout,[-1,mul])
            fc1 = self.fcLayer(Res, mul, int(384*1.41)+1, name = "fc4",seed=108+self.seed)
            fc2 = self.fcLayer(fc1, int(384*1.41)+1, int(192*1.41)+1, name =  "fc5",seed=109+self.seed)
            fc3 = self.fcLayer(fc2, int(192*1.41)+1, self.CLASSNUM, name =  "fc6",norm=False,seed=110+self.seed)
            self.out = fc3
