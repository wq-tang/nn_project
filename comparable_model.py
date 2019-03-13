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
        self.build_real_CNN_for_cifar10(2)

    def conv_block(self,name,kernel,channel,stride = 1,is_complex = True):
        if is_complex:
            conv_f = self.complex_convLayer
            pool_f = self.complex_maxPoolLayer
            net = self.X_com
        else:
            conv_f = self.convLayer
            pool_f = self.maxPoolLayer
            net = self.X
        with tf.variable_scope(name):
            for i in range(len(kernel)):
                kernel_size = kernel[i]
                if stride!=1:
                    stride_size = stride[i]
                else:
                    stride_size=1
                if is_complex:
                    channel_num=channel[i]
                else:
                    channel_num=int(channel[i]*1.41)+1

                conv = conv_f(net, [kernel_size, kernel_size], [stride_size, stride_size], channel_num, "conv"+str(i+1), "SAME",relu_fun = self.relu_fun)
                net = pool_f(conv,[2, 2],[ 2,2], "pool"+str(i+1), "SAME")
            if is_complex:
                return np.array(net)
            return net

    def build_complex_CNN_for_mnist(self):
        with tf.variable_scope('complex_mnist'):
            # conv3 = self.complex_convLayer(pool2, [2, 2], [1, 1], 256, "conv3",'VALID',relu_fun = self.relu_fun)
            # pool3 = self.complex_maxPoolLayer(conv3, [2, 2], [2, 2], "pool3", "VALID")
            cnnout = self.conv_block('complex_conv_block',[5,3],[16,8])
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
        with tf.variable_scope('real_mnist'):
            # conv3 = self.convLayer(pool2, [2, 2], [1, 1], int(256*1.41)+1, "conv3",'VALID')
            # pool3 = self.maxPoolLayer(conv3, [2, 2], [2, 2], "pool3", "VALID")
            cnnout = self.conv_block('conv_block',[5,3],[16,8],is_complex=False)
            cnnout = pool2
            shapes = cnnout.get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            Res = tf.reshape(cnnout,[-1,mul])
            fc1 = self.fcLayer(Res, mul, int(30*1.41)+1,  name = "fc4",seed=103+self.seed)
            self.fc2 = self.fcLayer(fc1, int(30*1.41)+1, self.CLASSNUM, name =  "fc5",seed=104+self.seed)
            self.out = self.fc2



    def build_complex_CNN_for_cifar10(self,model_num):
        with tf.variable_scope('complex_cifar10'):
            out = 0
            for i in range(model_num):
                out += self.conv_block('complex_conv_block'+str(i+1),[5,3,3],[128,64,64])
            cnnout = out
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


    def build_real_CNN_for_cifar10(self,model_num):
        with tf.variable_scope('real_cifar10'):
            out = 0
            for i in range(model_num):
                out += self.conv_block('complex_conv_block'+str(i+1),[5,3,3],[128,64,64],is_complex=False)
            cnnout = out
            shapes = cnnout.get_shape().as_list()[1:]
            mul = reduce(lambda x,y:x * y,shapes)
            Res = tf.reshape(cnnout,[-1,mul])
            fc1 = self.fcLayer(Res, mul, int(384*1.41)+1, name = "fc4",seed=108+self.seed)
            fc2 = self.fcLayer(fc1, int(384*1.41)+1, int(192*1.41)+1, name =  "fc5",seed=109+self.seed)
            fc3 = self.fcLayer(fc2, int(192*1.41)+1, self.CLASSNUM, name =  "fc6",norm=False,seed=110+self.seed)
            self.out = fc3
