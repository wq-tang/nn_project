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
    def __init__(self, x, classNum, seed,modelPath = "complexnet",is_complex=True):
        super(complex_net,self).__init__(x, classNum, seed,modelPath)
        tf.set_random_seed(seed)
        self.is_complex=is_complex
        self.relu_fun = tf.nn.relu
        # self.relu_fun = self.Learnable_angle_relu


    def conv_block(self,inputs,name,kernel,channel,pool_size=2,pool_strides=2,strides = 1,same = False):
        if self.is_complex:
            conv_f = self.complex_convLayer
            pool_f = self.complex_maxPoolLayer
        else:
            conv_f = self.convLayer
            pool_f = self.maxPoolLayer
        net = inputs
        with tf.variable_scope(name):
            for i in range(len(kernel)):
                kernel_size = kernel[i]
                if strides!=1:
                    stride_size = strides[i]
                else:
                    stride_size=1
                if self.is_complex:
                    channel_num=channel[i]
                else:
                    channel_num=int(channel[i]*1.41)+1

                conv = conv_f(net, [kernel_size, kernel_size], [stride_size, stride_size], channel_num, "conv"+str(i+1), "SAME",relu_fun = self.relu_fun)
                net = pool_f(conv,[pool_size, pool_size],[ pool_strides,pool_strides], "pool"+str(i+1), "SAME")
            if same == True:
                net = self.tile(net,inputs)
            if self.is_complex:
                return np.array(net)
            return net

    def fc_block(self,inputs,name,layer):
        if self.is_complex:
            fc_connect = self.complex_fcLayer
            cnnout = inputs[0]
        else:
            fc_connect = self.fcLayer
            cnnout=inputs
        shapes = cnnout.get_shape().as_list()[1:]
        mul = reduce(lambda x,y:x * y,shapes)
        pre = mul
        if self.is_complex:
            R = tf.reshape(inputs[0],[-1,mul])
            I = tf.reshape(inputs[1],[-1,mul])
            dim = R.get_shape()[1].value
            net = [R,I]
        else:
            net = tf.reshape(inputs,[-1,mul])

        with tf.variable_scope(name):
            for i in range(len(layer)-1):
                if self.is_complex:
                    now = layer[i]
                else:
                    now=int(layer[i]*1.41)+1

                net = fc_connect(net, pre, now,"fc"+str(i+1),relu_fun = self.relu_fun)
                pre = now
            net = fc_connect(net, pre, layer[-1],"fc"+str(len(layer)),relu_fun = self.relu_fun)
            if self.is_complex:
                return np.array(net)
            return net


    def build_CNN_for_mnist(self,model_num=1):
        with tf.variable_scope('mnist'):
            out = 0
            for i in range(model_num):
                out += self.conv_block('conv_block',[5,3],[16,8])
            self.out=self.fc_block(out,'fc_block',[30,self.CLASSNUM])
            if self.is_complex:
                self.out = tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))


    def build_CNN_for_cifar10(self,model_num=1):
        with tf.variable_scope('cifar10'):
            if self.is_complex:
                inputs = self.X_com
                inputs_data = [inputs,[self.X,self.X*0.0],[0.0*self.X,self.X],inputs]
            else:
                inputs=self.X
            out = 0
            for i in range(model_num):
                out += self.conv_block(inputs,'conv_block'+str(i+1),[5,3,3],[128,64,64])
            self.out=self.fc_block(out,'fc_block',[384,192,self.CLASSNUM])
            if self.is_complex:
                self.out = tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))



    def build_compare_for_cifar10(self,model_num=1):
        with tf.variable_scope('compare_cifar10'):
            if self.is_complex:
                inputs = self.X_com
            else:
                inputs=self.X
            net = inputs

            for i in range(model_num-1):
                net = self.conv_block(net,'conv_block'+str(i+1),[5,3,3],[128,64,64],same=True)
            net = self.conv_block(net,'conv_block'+str(model_num),[5,3,3],[128,64,64])
            self.out=self.fc_block(net,'fc_block',[384,192,self.CLASSNUM])
            if self.is_complex:
                self.out = tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))


    def pad(self,net,inputs):
        if self.is_complex:
            input_size = inputs[0].get_shape().as_list()[1]
            net_size = net[0].get_shape().as_list()[1]
        else:
            input_size = inputs.get_shape().as_list()[1]
            net_size = net.get_shape().as_list()[1]

        pad_total = input_size - net_size#因为这里是valid模式 size = (w-kernel_size+1)/stride 向上取整  所以这也就是原因所在
        pad_beg = pad_total//2
        pad_end = pad_total - pad_beg
        if self.is_complex:
            R = tf.pad(net[0],[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
            I = tf.pad(net[1],[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
            net = [R,I]
        else:
            net = tf.pad(net,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return net

    def tile(self,net,inputs):
        if self.is_complex:
            input_size = inputs[0].get_shape().as_list()[1]
            net_size = net[0].get_shape().as_list()[1]
        else:
            input_size = inputs.get_shape().as_list()[1]
            net_size = net.get_shape().as_list()[1]

        tile_num = input_size//net_size
        if self.is_complex:
            R = tf.tile(net[0],[1,tile_num,tile_num,1])
            I = tf.tile(net[1],[1,tile_num,tile_num,1])
            net = [R,I]
        else:
            net = tf.tile(net,[1,tile_num,tile_num,1])
        if input_size%net_size==0:
            return net
        return self.pad(net,inputs)

