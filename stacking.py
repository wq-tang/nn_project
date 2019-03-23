import os
import numpy as np 
import tensorflow as tf 
import time
import math
import sys
import h5py

from alexnet_model import complex_net
from resnet_model import Resnet

from CIFAR100 import read_cifar100
from FashionMNIST import read_fashion
from CIFAR10 import read_cifar10
from MNIST import read_mnist
from READ_DATA import read_data
# def wrrite_file(train_data,test_data,file_name):
# 	with h5py.File(file_name,'w') as f:
# 		f.create_group('Train')
# 		f.create_group('Test')
# 		f.create_dataset('Train/data',data = train_data[0])
# 		f.create_dataset('Train/labelsFine',data = train_data[1])
# 		f.create_dataset('Test/data',data = test_data[0])
# 		f.create_dataset('Test/labelsFine',data = test_data[1])

def wrrite_file(train_data,test_data,file_name):
	with h5py.File(file_name,'w') as f:
		f.create_group('Train')
		f.create_group('Test')
		f.create_dataset('Train/images',data = train_data[0])
		f.create_dataset('Train/labels',data = train_data[1])
		f.create_dataset('Test/images',data = test_data[0])
		f.create_dataset('Test/labels',data = test_data[1])

def stacking(path,kernel_list,channel_list,fc_list,is_complex=True):
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')
	def test(test_batch):
		precision=[]
		for i in range(10):
			test_x,test_y = next(test_batch)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)
	#修改模型以及对应的方法
	#修改读取文件的函数以及文件名称
	#修改输出路径
	#修改模型中的输出参数
	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/cifar10_meta_bagging/'+path)
	max_epoch = 30000
	batch_step = 128 
	file_name = 'CIFAR10_model'+path[-1]+'.h5'
	train_batch,test_batch = read_cifar10('data/'+file_name,batch_step,1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,10,0,is_complex=is_complex)
	model.diff_net(x,name=path,kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)
	out_result= tf.add(model.out,0.0,name = 'out')
	if is_complex:
		models_result = tf.sqrt(tf.square(out_result[0])+tf.square(out_result[1]))
	else:
		models_result =out_result
	tf.add_to_collection("model_out", out_result)
	with tf.name_scope('loss'):
		loss  = loss(models_result,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		tf.add_to_collection("train_op", train_op)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()

	saver=tf.train.Saver(max_to_keep=1)
	max_acc=0

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	test_x,test_y = next(test_batch)


	ans = []
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%500 ==0:
			loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			model.training = False
			acc = sess.run(accuracy, feed_dict={x:test_x,y:test_y})
			test_accuracy = test(test_batch)
			if i>max_epoch*0.8 and test_accuracy>max_acc:
				max_acc=test_accuracy
				saver.save(sess,model_path+'.ckpt')
			ans.append(test_accuracy)
			model.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))

	

	print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()



def generate_Primary_net_cifar(model_shape,model_tag,is_complex):
	#修改文件读取名字，读取函数
	### Using the class ###
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')
	def test(test_batch):
		precision=[]
		for i in range(10):
			test_x,test_y = next(test_batch)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)
	#修改模型以及对应的方法
	#修改读取文件的函数以及文件名称
	#修改输出路径
	#修改模型中的输出参数
	if is_complex:
		local_path = 'complex'+model_shape[0]
	else:
		local_path = 'real'+model_shape[0]

	path,kernel_list,channel_list,fc_list = model_shape
	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\
		'mynet/stacking/cifar10/'+local_path+str(model_tag))
	max_epoch = 50000
	batch_step = 128 
	file_name = 'CIFAR10_st4-'+str(model_tag)+'.h5'
	validation_name =  'CIFAR10_st4'+str(model_tag)+'.h5'
	valid_train,_ = read_cifar10('k_data/'+validation_name,1250,1000)#10次取完
	train_batch,test_batch = read_cifar10('k_data/'+file_name,batch_step,1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,10,0,is_complex=is_complex)
	model.diff_net(x,name=local_path+str(model_tag),kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)
	out_result= tf.add(model.out,0.0,name = 'out')
	if is_complex:
		models_result = tf.sqrt(tf.square(out_result[0])+tf.square(out_result[1]))
	else:
		models_result =out_result
	tf.add_to_collection("model_out", out_result)
	with tf.name_scope('loss'):
		loss  = loss(models_result,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		tf.add_to_collection("train_op", train_op)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()

	saver=tf.train.Saver(max_to_keep=1)

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	

	ans = []
	test_x,test_y = next(test_batch)
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%1000 ==0:
			loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
			model.training=False
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			model.training = False
			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d, test accuracy %g"%(i, test_accuracy))

	
	saver.save(sess,model_path+'.ckpt')

	if is_complex:
		axis = 1
	else:
		axis=0

	valid_data=[]
	valid_label = []
	test_data=[]
	test_label = []
	for i in range(10):
		test_x,test_y = next(test_batch)
		valid_train_data,valid_train_label = next(valid_train)
		valid_data.append((sess.run(out_result,feed_dict={x:valid_train_data})))
		valid_label.append(valid_train_label)
		test_label.append(test_y)
		test_data.append((sess.run(out_result,feed_dict={x:test_x})))

	print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()


	return np.mean(ans[-5:]), np.concatenate(valid_data,axis=axis),np.concatenate(valid_label,axis=0),\
	np.concatenate(test_data,axis=axis),np.concatenate(test_label,axis=0)


def generate_Primary_net_mnist(model_shape,model_tag,is_complex):
	#修改文件读取名字，读取函数
	### Using the class ###
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')
	def test(test_batch):
		precision=[]
		for i in range(10):
			test_x,test_y = next(test_batch)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)
	#修改模型以及对应的方法
	#修改读取文件的函数以及文件名称
	#修改输出路径
	#修改模型中的输出参数
	if is_complex:
		local_path = 'complex'+model_shape[0]
	else:
		local_path = 'real'+model_shape[0]

	path,kernel_list,channel_list,fc_list = model_shape
	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\
		'mynet/stacking/fashion/'+local_path+str(model_tag))   ##修改
	max_epoch = 50000                                        ##修改
	batch_step = 128 
	file_name = 'FashionMNIST_st4-'+str(model_tag)+'.h5'            ##修改
	validation_name =  'FashionMNIST_st4'+str(model_tag)+'.h5'      ##修改
	valid_train,_ = read_fashion('k_data/'+validation_name,1500,1000)#10次取完  ##修改
	train_batch,test_batch = read_fashion('k_data/'+file_name,batch_step,1000) ##修改
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,28,28,1],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,10,0,is_complex=is_complex)       ##修改
	model.diff_net(x,name=local_path+str(model_tag),kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)##修改
	out_result= tf.add(model.out,0.0,name = 'out')
	if is_complex:
		models_result = tf.sqrt(tf.square(out_result[0])+tf.square(out_result[1]))
	else:
		models_result =out_result
	tf.add_to_collection("model_out", out_result)
	with tf.name_scope('loss'):
		loss  = loss(models_result,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		tf.add_to_collection("train_op", train_op)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()

	saver=tf.train.Saver(max_to_keep=1)

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	
	ans=[]
	test_x,test_y = next(test_batch)
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%1000 ==0:
			loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
			model.training = False
			acc = sess.run(accuracy, feed_dict={x:test_x,y:test_y})
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			model.training = True
			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
	
	saver.save(sess,model_path+'.ckpt')

	if is_complex:
		axis = 1
	else:
		axis=0

	valid_data=[]
	valid_label = []
	test_data=[]
	test_label = []
	for i in range(10):
		test_x,test_y = next(test_batch)
		valid_train_data,valid_train_label = next(valid_train)
		valid_data.append((sess.run(out_result,feed_dict={x:valid_train_data})))
		valid_label.append(valid_train_label)
		test_label.append(test_y)
		test_data.append((sess.run(out_result,feed_dict={x:test_x})))

	print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()


	return np.mean(ans[-5:]), np.concatenate(valid_data,axis=axis),np.concatenate(valid_label,axis=0),\
	np.concatenate(test_data,axis=axis),np.concatenate(test_label,axis=0)



class ImportGraph():
	#修改模型读取路径
	"""  Importing and running isolated TF graph """
	def __init__(self, loc):
		model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'resnet/cifar100_meta_bagging/'+loc)
		# Create local graph and use it in the session
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		with self.graph.as_default():
			# Import saved model from location 'loc' into local graph
			# 从指定路径加载模型到局部图中
			saver = tf.train.import_meta_graph(model_path+ '.ckpt.meta',
												clear_devices=True)
			saver.restore(self.sess, model_path+'.ckpt')
			# There are TWO options how to get activation operation:
			# 两种方式来调用运算或者参数
				# FROM SAVED COLLECTION:            
			self.activation = tf.get_collection('model_out')[0]
				# BY NAME:
			# self.activation = self.graph.get_operation_by_name('inputs').outputs[0]

	def run(self, data):
		""" Running the activation operation previously imported """
		# The 'x' corresponds to name of input placeholder
		return self.sess.run(self.activation, feed_dict={"inputs/input_x:0": data})




def restore(model_path_list):
	#修改文件读取名字，读取函数
	### Using the class ###
	file_name = 'CIFAR100.h5'
	_,test_batch = read_cifar100('data/'+file_name,1,1000)
	accuracy=0.0
	for k in range(10):
		data,lable  = next(test_batch)
		result = []
		for i in range(len(model_path_list)):
			model = ImportGraph(model_path_list[i])
			result.append(model.run(data))

		result = np.array(result)
		result= np.sum(result,0)
		if model_path_list[0][:7] == 'complex':
			models_result = tf.sqrt(tf.square(result[0])+tf.square(result[1]))
		else:
			models_result = result

		top_k_op = tf.nn.in_top_k(models_result,lable,1)
		accuracy += tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	sess = tf.InteractiveSession()
	acc = sess.run(accuracy/10)
	print(model_path_list)
	print("%.4f"%(acc))
	return acc



	if is_complex:
		axis = 1
	else:
		axis=0

	valid_data=[]
	valid_label = []
	test_data=[]
	test_label = []
	for i in range(10):
		test_x,test_y = next(test_batch)
		valid_train_data,valid_train_label = next(valid_train)
		valid_data.append((sess.run(out_result,feed_dict={x:valid_train_data})))
		valid_label.append(valid_train_label)
		test_label.append(test_y)
		test_data.append((sess.run(out_result,feed_dict={x:test_x})))

	print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()


	return np.mean(ans[-5:]), np.concatenate(valid_data,axis=axis),np.concatenate(valid_label,axis=0),\
	np.concatenate(test_data,axis=axis),np.concatenate(test_label,axis=0)

def generate_secondary_data(model_shape,is_complex):
	acc = {}
	if is_complex:
		axis =1
	else:
		axis=0
	train_data_set = []
	train_label_set=[]
	test_data_set =[]
	if is_complex:
		file_head = 'CIFAR10_complex'+model_shape[0]  ####修改
	else:
		file_head = 'CIFAR10_real'+model_shape[0]   #修改
	accuracy=0
	for tag in range(1,5):
		graph = tf.Graph()
		with graph.as_default():
			accu,train_data,train_label,test_data,test_label = restore(model_path_list)
		accuracy+=accu
		train_data_set.append(train_data)
		train_label_set.append(train_label)
		test_data_set.append(test_data)
	train_data_set = np.concatenate(train_data_set,axis = axis)
	train_label_set = np.concatenate(train_label_set,axis = 0)

	test_data_set = np.mean(test_data_set,axis=0)
	acc[file_head] = accuracy/4
	print(acc)
		#写操作
	wrrite_file([train_data_set,train_label_set],[test_data_set,test_label],'bu/'+file_head+'.h5')






#cifar100
# kernel_list = [[5,5,3,3],[5,5,5,3],[5,5,3],[5,3,3],[5,5,3,3],[5,5,5,3],[5,5,3],[5,3,3]]
# channel_list = [[128,128,64,64],[128,64,64,64],[128,128,64],[128,64,64],[128,128,64,64],[128,64,64,64],[128,128,64],[128,64,64]]
# fc_list =[[192],[192],[192,81],[192,81],[192],[192],[192,81],[192,81]]

#cifar10
kernel_list = [[5,3,3],[5,5,2],[5,5],[5,3]]
channel_list = [[128,64,64],[128,64,64],[128,128],[128,64]]
fc_list =[[100],[128],[100,50],[100,50]]

#mnist
# kernel_list = [[5,3],[5,3,3],[5],[3]]
# channel_list = [[16,8],[8,16,16],[32],[32]]
# fc_list =[[30],[50],[30],[60,30]]

#cifar100 and 10
# kernel_list = [[3,3,3,3],[5,5,5,5],[5,5,3,3],[5,3,3]]
# channel_list = [[64,128,256,512],[64,128,128,256],[64,64,128,256],[128,256,512]]

path_list = ['1','2','3','4']

shape_list = [path_list,kernel_list,channel_list,fc_list]

if __name__=='__main__':
	model_index = 0
	is_complex = True
	generate_secondary_data([k[model_index] for k in  shape_list],is_complex) #修改

