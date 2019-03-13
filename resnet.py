import collections
import tensorflow as tf
slim = tf.contrib.slim
from datetime import datetime
import pickle   # 用于序列化和反序列化
import numpy as np  
import os  
import matplotlib.pyplot as plt
import time
import math


class Cifar100DataReader():  
    def __init__(self,cifar_folder,onehot=False):  
        self.cifar_folder=cifar_folder  
        self.onehot=onehot  
        self.data_label_train=None            # 训练集
        self.data_label_test=None             # 测试集
        self.batch_index=0                    # 训练数据的batch块索引
        self.test_batch_index=0                # 测试数据的batch_size
        f=os.path.join(self.cifar_folder,"train")  # 训练集有50000张图片，100个类，每个类500张
        print ('read: %s'%f  )
        fo = open(f, 'rb')
        self.dic_train = pickle.load(fo,encoding='bytes')
        fo.close()
        self.data_label_train=list(zip(self.dic_train[b'data'],self.dic_train[b'fine_labels']) ) #label 0~99  
        np.random.shuffle(self.data_label_train)           
 
    
    def dataInfo(self):
        print (self.data_label_train[0:2] )# 每个元素为二元组，第一个是numpy数组大小为32*32*3，第二是label
        print (self.dic_train.keys())
        print (b"coarse_labels:",len(self.dic_train[b"coarse_labels"]))
        print (b"filenames:",len(self.dic_train[b"filenames"]))
        print (b"batch_label:",len(self.dic_train[b"batch_label"]))
        print (b"fine_labels:",len(self.dic_train[b"fine_labels"]))
        print (b"data_shape:",np.shape((self.dic_train[b"data"])))
        print (b"data0:",type(self.dic_train[b"data"][0]))
 
 
    # 得到下一个batch训练集，块大小为100
    def next_train_data(self,batch_size=100):  
        """ 
        return list of numpy arrays [na,...,na] with specific batch_size 
                na: N dimensional numpy array  
        """            
        if self.batch_index<len(self.data_label_train)/batch_size:  
            print ("batch_index:",self.batch_index  )
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]  
            self.batch_index+=1  
            return self._decode(datum,self.onehot)  
        else:  
            self.batch_index=0  
            np.random.shuffle(self.data_label_train)  
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]  
            self.batch_index+=1  
            return self._decode(datum,self.onehot)  
              
    
    # 把一个batch的训练数据转换为可以放入神经网络训练的数据 
    def _decode(self,datum,onehot):  
        rdata=list()     # batch训练数据
        rlabel=list()  
        if onehot:  
            for d,l in datum:  
                rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))   # 转变形状为：32*32*3
                hot=np.zeros(100)    
                hot[int(l)]=1            # label设为100维的one-hot向量
                rlabel.append(hot)  
        else:  
            for d,l in datum:  
                rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))  
                rlabel.append(int(l))  
        return rdata,np.array(rlabel)  
    
 
    # 得到下一个测试数据 ，供神经网络计算模型误差用         
    def next_test_data(self,batch_size=100):  
        ''''' 
        return list of numpy arrays [na,...,na] with specific batch_size 
                na: N dimensional numpy array  
        '''  
        if self.data_label_test is None:  
            f=os.path.join(self.cifar_folder,"test")  
            print ('read: %s'%f  )
            fo = open(f, 'rb')
            dic_test = pickle.load(fo,encoding='bytes')
            fo.close()           
            data=dic_test[b'data']            
            labels=dic_test[b'fine_labels']   # 0 ~ 99  
            self.data_label_test=list(zip(data,labels) )
            self.batch_index=0
 
        if self.test_batch_index<len(self.data_label_test)/batch_size:  
            print ("test_batch_index:",self.test_batch_index )
            datum=self.data_label_test[self.test_batch_index*batch_size:(self.test_batch_index+1)*batch_size]  
            self.test_batch_index+=1  
            return self._decode(datum,self.onehot)  
        else:  
            self.test_batch_index=0  
            np.random.shuffle(self.data_label_test)  
            datum=self.data_label_test[self.test_batch_index*batch_size:(self.test_batch_index+1)*batch_size]  
            self.test_batch_index+=1  
            return self._decode(datum,self.onehot)    
 
    # 显示 9张图像
    def showImage(self):
        rdata,rlabel = self.next_train_data()
        fig = plt.figure()  
        ax = fig.add_subplot(331)
        ax.imshow(rdata[0])
        ax = fig.add_subplot(332)
        ax.imshow(rdata[1]) 
        ax = fig.add_subplot(333)
        ax.imshow(rdata[2]) 
        ax = fig.add_subplot(334)
        ax.imshow(rdata[3]) 
        ax = fig.add_subplot(335)
        ax.imshow(rdata[4]) 
        ax = fig.add_subplot(336)
        ax.imshow(rdata[5]) 
        ax = fig.add_subplot(337)
        ax.imshow(rdata[6]) 
        ax = fig.add_subplot(338)
        ax.imshow(rdata[7]) 
        ax = fig.add_subplot(339)
        ax.imshow(rdata[8]) 
        plt.show()


class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
	'a nemed tuple describing a resnet block'


class Resnet_v2_50(object):
	"""docstring for Resnet_v2_50"""
	def __init__(self,inputs,num_classes=None,is_training=True,global_pool = True,reuse=None):
		self.inputs = inputs
		self.num_classes=num_classes
		self.is_training=is_training
		self.global_pool = global_pool
		self.reuse=reuse
		with slim.arg_scope(self.resnet_arg_scope(is_training=self.is_training)):
			self.net,self.end_points= self.resnet_v2_50(inputs,num_classes,global_pool,reuse)

	def resnet_v2_50(self,inputs,num_classes=None,global_pool = True,reuse=None,scope='resnet_v2_50'):
		blocks = [Block('block1',self.bottleneck,[(256,64,1)]*2+[(256,64,2)]),\
				Block('block2',self.bottleneck,[(512,128,1)]*3+[(512,128,2)]),\
				Block('block3',self.bottleneck,[(1024,256,1)]*5+[(1024,256,2)]),\
				Block('block4',self.bottleneck,[(2048,512,1)]*3)]
		return self.resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)

	def subsample(self,inputs,factors,scope = None):
		if factors == 1:
			return factors
		return slim.max_pool2d(inputs,[1,1],stride = factors,scope=scope)

	def conv2d_same(self,inputs,num_outputs,kernel_size,stride,scope =None):
		if stride == 1:
			return	slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding = 'SAME',scope=scope)

		pad_total = kernel_size-1   #因为这里是valid模式 size = (w-kernel_size+1)/stride 向上取整  所以这也就是原因所在
		pad_beg = pad_total//2
		pad_end = pad_total - pad_beg
		inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
		return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding = 'VALID',scope=scope)

	@slim.add_arg_scope
	def stacks_block_dense(self,net,blocks,outputs_collections=None):
		for block in blocks:
			with tf.variable_scope(block.scope,'block',[net]) as sc:
			#variable_scope前三个参数#name_or_scope,default_name=None,values=None. values: 传入该scope的tensor参数
				for i,unit in enumerate(block.args):
					with tf.variable_scope("unit_%d"%(i+1),values = [net]):
						unit_depth,unit_depth_bottleneck,unit_stride = unit
						net = block.unit_fn(net,depth = unit_depth,depth_bottleneck = unit_depth_bottleneck,stride=unit_stride)
				net = slim.utils.collect_named_outputs(outputs_collections,sc.name,net)

		return net

	def resnet_arg_scope(self,is_training = True,weight_decay = 0.0001,\
		batch_norm_decay = 0.997,batch_norm_epsilon = 1e-5,batch_norm_scale = True):

		batch_norm_params={"is_training":is_training,"decay":batch_norm_decay,'epsilon':batch_norm_epsilon,\
		'scale':batch_norm_scale,'updates_collections':tf.GraphKeys.UPDATE_OPS}

		with slim.arg_scope([slim.conv2d],weights_regularizer = slim.l2_regularizer(weight_decay),\
			weights_initializer = slim.variance_scaling_initializer(),activation_fn=tf.nn.relu,\
			normalizer_fn = slim.batch_norm,normalizer_params=batch_norm_params):
			with slim.arg_scope([slim.batch_norm],**batch_norm_params):
				with slim.arg_scope([slim.max_pool2d],padding = 'SAME') as arg_sc:
					return arg_sc


	@slim.add_arg_scope
	def bottleneck(self,inputs,depth,depth_bottleneck,stride,outputs_collections = None,scope=None):
		with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
			depth_in = slim.utils.last_dimension(inputs.get_shape(),min_rank = 4)
			preact = slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope = 'preact')
			if depth == depth_in:
				shortcut = self.subsample(inputs,stride,'shortcut')
			else:
				shortcut = slim.conv2d(preact,depth,[1,1],stride=stride,\
					normalizer_fn=None,activation_fn=None,scope = 'shortcut')

			residual = slim.conv2d(preact,depth_bottleneck,[1,1],stride =1,scope='conv1')
			residual = self.conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
			residual = slim.conv2d(residual,depth,[1,1],stride=1,normalizer_fn = None,activation_fn=None,scope='conv3')

			output = shortcut+residual

			return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)


	def resnet_v2(self,inputs,blocks,num_classes=None,global_pool=True,include_root_block = True,reuse=None,scope=None):
		with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
			end_points_collections = sc.original_name_scope + '_end_points'
			with slim.arg_scope([slim.conv2d,self.bottleneck,self.stacks_block_dense],outputs_collections=end_points_collections):
				net = inputs
				if include_root_block:
					with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
						net = self.conv2d_same(net,64,7,stride=2,scope='conv1')
					net = slim.max_pool2d(net,[3,3],stride=2,scope='pool1')
				net = self.stacks_block_dense(net,blocks)
				net = slim.batch_norm(net,activation_fn=tf.nn.relu,scope='postnorm')

				if global_pool:
					net = tf.reduce_mean(net,[1,2],name='pool5',keepdims=True)

				if num_classes is not None:
					net = slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='logits')

				end_points = slim.utils.convert_collection_to_dict(end_points_collections)

				if num_classes is not None:
					end_points['predictions'] = slim.softmax(net,scope = 'predictions')

				net = tf.reduce_mean(net,[1,2],name='pool6',keepdims=False)
				net = slim.fully_connected(net, 100, scope='fc-100')
				return net,end_points


# def resnet_v2_50(inputs,num_classes=None,global_pool = True,reuse=None,scope='resnet_v2_50'):
# 	blocks = [Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),\
# 			Block('block2',bottleneck,[(512,128,1)]*3+[(512,128,2)]),\
# 			Block('block3',bottleneck,[(1024,256,1)]*5+[(1024,256,2)]),\
# 			Block('block4',bottleneck,[(2048,512,1)]*3)]
# 	return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)

# def resnet_v2_101(inputs,num_classes=None,global_pool = True,reuse=None,scope='resnet_v2_101'):
# 	blocks = [Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),\
# 			Block('block2',bottleneck,[(512,128,1)]*3+[(512,128,2)]),\
# 			Block('block3',bottleneck,[(1024,256,1)]*22+[(1024,256,2)]),\
# 			Block('block4',bottleneck,[(2048,512,1)]*3)]
# 	return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope) 

# def resnet_v2_152(inputs,num_classes=None,global_pool = True,reuse=None,scope='resnet_v2_152'):
# 	blocks = [Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),\
# 			Block('block2',bottleneck,[(512,128,1)]*7+[(512,128,2)]),\
# 			Block('block3',bottleneck,[(1024,256,1)]*35+[(1024,256,2)]),\
# 			Block('block4',bottleneck,[(2048,512,1)]*3)]
# 	return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope) 

# def resnet_v2_200(inputs,num_classes=None,global_pool = True,reuse=None,scope='resnet_v2_200'):
# 	blocks = [Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),\
# 			Block('block2',bottleneck,[(512,128,1)]*23+[(512,128,2)]),\
# 			Block('block3',bottleneck,[(1024,256,1)]*35+[(1024,256,2)]),\
# 			Block('block4',bottleneck,[(2048,512,1)]*3)]
# 	return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope) 






def resnet():

	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits+0.1**8,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')

	def test():
		precision=[]
		for i in range(40):
			test_x,test_y = cifar100.next_test_data(1000)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)

	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'resnet/cifar100')
	data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar-100/cifar-100-python')
	
	cifar100 = Cifar100DataReader(data_dir)


	max_epoch = 50000
	batch_step = 128
	with tf.name_scope("inputs"):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	tf.summary.image('inputs', x, 10)
	y = tf.placeholder(tf.int32, [None])

	model = [Resnet_v2_50(x,100)]
	net =model[0].net

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
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	ans=[]
	test_x,test_y = cifar100.next_test_data()
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = cifar100.next_train_data(batch_step)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%200 ==0:
			summary,loss_value = sess.run([merged,gloss], feed_dict={x:train_x,y:train_y})
			train_writer.add_summary(summary, i)
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			for m in model:
				m.training = False
			summary, acc = sess.run([merged, accuracy], feed_dict={x:test_x,y:test_y})
			test_writer.add_summary(summary, i)
			test_accuracy = test()
			ans.append(test_accuracy)
			# test_accuracy = accuracy.eval(feed_dict={x:test_x, y: test_y})
			for m in model:
				m.training = True
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
	resnet()
