import collections
import tensorflow as tf
from datetime import datetime
import pickle   # 用于序列化和反序列化
import numpy as np  
import os  
# import matplotlib.pyplot as plt
import time
import math
from base_class import base_class
from functools import reduce




class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
	'a nemed tuple describing a resnet block'


class Resnet(base_class):
	"""docstring for Resnet_v2_18"""
	def __init__(self, x, classNum, seed,is_complex = True,modelPath = "Resnet_v2"):
		super(Resnet,self).__init__(x, classNum, seed,modelPath)
		tf.set_random_seed(seed)
		self.is_complex=is_complex
		if is_complex:
			self.conv = self.complex_convLayer
			self.pool = self.complex_maxPoolLayer
			self.connect = self.complex_fcLayer
			self.batch_normalization = self.complex_batch_normalization
			self.inputs = self.X_com
		else:
			self.conv = self.convLayer
			self.pool = self.maxPoolLayer
			self.connect = self.fcLayer
			self.batch_normalization = tf.layers.batch_normalization
			self.inputs = self.X

	def subsample(self,inputs,factors,scope = None):
		if factors == 1:
			return inputs
		return self.pool(inputs,[1,1],strides = [factors,factors],name=scope,padding = "SAME")

	def bottleneck(self,inputs,kernel_size,depth_bottleneck,scope=None):
		with tf.variable_scope(scope):
			if self.is_complex:
				depth_in = inputs[0].get_shape().as_list()[-1]
			else:
				depth_in = inputs.get_shape().as_list()[-1]
			if depth_bottleneck == depth_in:
				shortcut = self.subsample(inputs,2,'shortcut')
			else:
				shortcut = self.conv(inputs,[1,1],[2,2],depth_bottleneck,name = 'shortcut',padding = "SAME")

			residual = self.conv(inputs,[kernel_size,kernel_size],[1,1],depth_bottleneck,padding = 'SAME',name='conv1')
			# residual = self.conv2d_same(residual,depth_bottleneck,3,strides,name='conv2')
			residual = self.conv(residual,[kernel_size,kernel_size],[2,2],depth_bottleneck,padding = 'SAME',name='conv3')

			if self.is_complex:
				output = [shortcut[0]+residual[0],shortcut[1]+residual[1]]
			else:
				output = shortcut+residual

			return output

	def stacks_block_dense(self,inputs,blocks,scope,same = False):
		with tf.variable_scope(scope):
			net = inputs
			for block in blocks:
				with tf.variable_scope(block.scope):
				#variable_scope前三个参数#name_or_scope,default_name=None,values=None. values: 传入该scope的tensor参数
					for i,unit in enumerate(block.args):
						with tf.variable_scope("unit_%d"%(i+1)):
							kernel_size,unit_depth_bottleneck = unit
							net = block.unit_fn(net,kernel_size,depth_bottleneck = unit_depth_bottleneck,scope='bottleneck'+str(i+1))
			if same:
				net = self.pad(net,inputs)
			if self.is_complex:
				return np.array(net)
			return net

	def build_compare_resnet(self,model_num=1):
		channel_list = [64,128,256,512]
		kernel_size_list = [3,3,3,3]
		if not self.is_complex:
			channel_list = [int(chanel*1.41)+1 for chanel in channel_list[:]]
		blocks = [Block('block1',self.bottleneck,[(kernel_size_list[0],channel_list[0])]*2),\
				Block('block2',self.bottleneck,[(kernel_size_list[1],channel_list[1])]*2),\
				Block('block3',self.bottleneck,[(kernel_size_list[2],channel_list[2])]*2),\
				Block('block4',self.bottleneck,[(kernel_size_list[3],channel_list[3])]*2)]
		net = self.inputs
		with tf.variable_scope('resnet'):
			for i in range(model_num-1):
				net = self.stacks_block_dense(net,blocks,scope ='BLOCK'+str(i+1) ,same=True)
			net = self.stacks_block_dense(net,blocks,scope ='BLOCK'+str(model_num))
			if self.is_complex:
				R = tf.reduce_mean(net[0],[1,2],name='reduceR',keep_dims=False)
				I = tf.reduce_mean(net[0],[1,2],name='reduceI',keep_dims=False)
				net = [R,I]
				tensor = net[0]
			else:
				net = tf.reduce_mean(net,[1,2],name='pool5',keep_dims=False)
				tensor=net
			dim = tensor.get_shape().as_list()[-1]
			self.out = self.connect(net,dim,self.classNum,name='fc-100')
			if self.is_complex:
				self.out= tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))

	def build_resnet(self,model_num=1):
		channel_list = [64,128,256,512]
		kernel_size_list = [3,3,3,3]
		if not self.is_complex:
			channel_list = [int(chanel*1.41)+1 for chanel in channel_list[:]]
		blocks = [Block('block1',self.bottleneck,[(kernel_size_list[0],channel_list[0])]*2),\
				Block('block2',self.bottleneck,[(kernel_size_list[1],channel_list[1])]*2),\
				Block('block3',self.bottleneck,[(kernel_size_list[2],channel_list[2])]*2),\
				Block('block4',self.bottleneck,[(kernel_size_list[3],channel_list[3])]*2)]
		net = 0.0
		with tf.variable_scope('resnet'):
			for i in range(model_num):
				net += self.stacks_block_dense(self.inputs,blocks,scope ='BLOCK'+str(i+1))
			if self.is_complex:
				R = tf.reduce_mean(net[0],[1,2],name='reduceR',keep_dims=False)
				I = tf.reduce_mean(net[0],[1,2],name='reduceI',keep_dims=False)
				net = [R,I]
				tensor = net[0]
			else:
				net = tf.reduce_mean(net,[1,2],name='pool5',keep_dims=False)
				tensor=net
			dim = tensor.get_shape().as_list()[-1]
			self.out = self.connect(net,dim,self.CLASSNUM,name='fc-100')
			if self.is_complex:
				self.out= tf.sqrt(tf.square(self.out[0])+tf.square(self.out[1]))
			

	def diff_sigle_model(self,inputs,name,kernel_list,channel_list):
		if not self.is_complex:
			channel_list = [int(chanel*1.41)+1 for chanel in channel_list[:]]
		blocks = [Block('block'+str(i),self.bottleneck,[(kernel_list[i],channel_list[i])]*2)\
		 for i in range(len(channel_list))]

		net = 0.0
		with tf.variable_scope('resnet'):
			net = self.stacks_block_dense(self.inputs,blocks,scope ='BLOCK'+str(i+1) ,same=True)
			if self.is_complex:
				R = tf.reduce_mean(net[0],[1,2],name='reduceR',keep_dims=False)
				I = tf.reduce_mean(net[0],[1,2],name='reduceI',keep_dims=False)
				net = [R,I]
				tensor = net[0]
			else:
				net = tf.reduce_mean(net,[1,2],name='pool5',keep_dims=False)
				tensor=net
			dim = tensor.get_shape().as_list()[-1]
			self.out = self.connect(net,dim,self.classNum,name='fc-100')


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



