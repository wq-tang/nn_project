import collections
import tensorflow as tf
from datetime import datetime
import pickle   # 用于序列化和反序列化
import numpy as np  
import os  
import matplotlib.pyplot as plt
import time
import math
from model import alexNet
from functools import reduce
from cifar100 import read_cifar100




class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
	'a nemed tuple describing a resnet block'


class Resnet_v2_50(alexNet):
	"""docstring for Resnet_v2_18"""
	def __init__(self, x, classNum, seed,is_complex = True,modelPath = "Resnet_v2_50"):
		super(Resnet_v2_50,self).__init__(x, classNum, seed,modelPath)
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
		self.out = self.resnet_v2_50()

	def subsample(self,inputs,factors,scope = None):
		if factors == 1:
			return inputs
		return self.pool(inputs,[1,1],strides = [factors,factors],name=scope,padding = "SAME")

	def bottleneck(self,inputs,depth,depth_bottleneck,strides,scope=None):
		with tf.variable_scope(scope):
			if self.is_complex:
				depth_in = inputs[0].get_shape().as_list()[-1]
			else:
				depth_in = inputs.get_shape().as_list()[-1]
			preact = self.batch_normalization(inputs,training=self.training)
			if depth == depth_in:
				shortcut = self.subsample(inputs,strides,'shortcut')
			else:
				shortcut = self.conv(preact,[1,1],[strides,strides],depth,name = 'shortcut',padding = "SAME")

			residual = self.conv(preact,[1,1],[1,1],depth_bottleneck,padding = 'SAME',name='conv1')
			residual = self.conv2d_same(residual,depth_bottleneck,3,strides,name='conv2')
			residual = self.conv(residual,[1,1],[1,1],depth,padding = 'SAME',name='conv3')

			if self.is_complex:
				output = [shortcut[0]+residual[0],shortcut[1]+residual[1]]
			else:
				output = shortcut+residual

			return output

	def stacks_block_dense(self,net,blocks):
		for block in blocks:
			with tf.variable_scope(block.scope):
			#variable_scope前三个参数#name_or_scope,default_name=None,values=None. values: 传入该scope的tensor参数
				for i,unit in enumerate(block.args):
					with tf.variable_scope("unit_%d"%(i+1)):
						unit_depth,unit_depth_bottleneck,unit_stride = unit
						net = block.unit_fn(net,depth = unit_depth,depth_bottleneck = unit_depth_bottleneck,\
							strides=unit_stride,scope='bottleneck'+str(i+1) )

		return net

	def conv2d_same(self,inputs,num_outputs,kernel_size,strides,name =None):
		if strides == 1:
			return	self.conv(inputs,[kernel_size,kernel_size],[1,1],num_outputs,padding = 'SAME',name=name)

		pad_total = kernel_size-1   #因为这里是valid模式 size = (w-kernel_size+1)/stride 向上取整  所以这也就是原因所在
		pad_beg = pad_total//2
		pad_end = pad_total - pad_beg
		if self.is_complex:
			R = tf.pad(inputs[0],[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
			I = tf.pad(inputs[1],[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
			inputs = [R,I]
		else:
			inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
		return self.conv(inputs,[kernel_size,kernel_size],[strides,strides],num_outputs,padding = 'VALID',name=name)	

	def resnet_v2(self,blocks,num_classes=None,global_pool=True,include_root_block = True,reuse=None,scope=None):
		with tf.variable_scope(scope,reuse=reuse):
			net = self.inputs
			if include_root_block:
				net = self.conv2d_same(net,64,7,strides=2,name='conv1')
				net = self.pool(net,[3,3],[2,2],padding ='SAME',name='pool1')
			net = self.stacks_block_dense(net,blocks)
			net = self.batch_normalization(net,training=self.training)

			if global_pool:
				if self.is_complex:
					R = tf.reduce_mean(net[0],[1,2],name='reduceR',keep_dims=True)
					I = tf.reduce_mean(net[0],[1,2],name='reduceI',keep_dims=True)
					net = [R,I]
				else:
					net = tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)

			if num_classes is not None:
				net = self.conv(net,[1,1],[1,1],num_classes,padding='SAME',name='logits')

			if self.is_complex:
				R = tf.reduce_mean(net[0],[1,2],name='reduceR',keep_dims=False)
				I = tf.reduce_mean(net[0],[1,2],name='reduceI',keep_dims=False)
				net = [R,I]
				tensor = net[0]
			else:
				net = tf.reduce_mean(net,[1,2],name='pool6',keep_dims=False)
				tensor = net

			dim = tensor.get_shape().as_list()[-1]
			net = self.connect(net,dim,100, name='fc-100')
			return net

	def resnet_v2_50(self,global_pool = True,reuse=None,scope='resnet_v2_50'):

		blocks = [Block('block1',self.bottleneck,[(256,64,1)]*2+[(256,64,2)]),\
				Block('block2',self.bottleneck,[(512,128,1)]*3+[(512,128,2)]),\
				Block('block3',self.bottleneck,[(1024,256,1)]*5+[(1024,256,2)]),\
				Block('block4',self.bottleneck,[(2048,512,1)]*3)]
		if self.is_complex:
			R,I=self.resnet_v2(blocks,self.CLASSNUM,global_pool,include_root_block=True,reuse=reuse,scope=scope)
			return tf.sqrt(tf.square(R)+tf.square(I))
		return self.resnet_v2(blocks,self.CLASSNUM,global_pool,include_root_block=True,reuse=reuse,scope=scope)



def resnet(path):

	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits+0.1**8,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')

	def test(test_batch):
		precision=[]
		for i in range(10):
			test_x,test_y = next(test_batch)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)

	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'resnet_board/'+path)
	# data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar-100/cifar-100-python')
	
	cifar100 = Cifar100DataReader(data_dir)


	max_epoch = 50000
	batch_step = 128
	train_batch,test_batch=read_cifar100(batch_step,1000)
	with tf.name_scope("inputs"):
		x = tf.placeholder(tf.float32, [None, 24, 24, 3])
	tf.summary.image('inputs', x, 10)
	y = tf.placeholder(tf.int32, [None])

	model = Resnet_v2_50(x,100,0,is_complex=True)
	net =model.out

	with tf.name_scope('loss'):
		gloss  = loss(net, y)
	tf.summary.scalar('loss', gloss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(gloss)


	top_k_op = tf.nn.in_top_k(net,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()
	# merged = tf.summary.merge_all()
	# train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	# test_writer = tf.summary.FileWriter(log_dir + '/test')

	tf.global_variables_initializer().run()
	# tf.train.start_queue_runners()

	ans=[]
	test_x,test_y = next(test_batch)
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%200 ==0:
			summary,loss_value = sess.run([merged,gloss], feed_dict={x:train_x,y:train_y})
			# train_writer.add_summary(summary, i)
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			model.training = False
			acc = sess.run([ accuracy], feed_dict={x:test_x,y:test_y})
			# test_writer.add_summary(summary, i)
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			# test_accuracy = accuracy.eval(feed_dict={x:test_x, y: test_y})
			model.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
			# print('pre:%g'%pre)
			# print(y_hat)
			# if test_accuracy>0.95:
			# 	break
	train_writer.close()
	test_writer.close()
	

	print('precision @1 = %.5f'%np.mean(ans[-10:]))



if __name__=='__main__':
	resnet('tests')
