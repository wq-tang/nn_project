import os
import numpy as np 
import tensorflow as tf 
import time
import math
import sys

from alexnet_model import complex_net
from resnet_model import Resnet

from CIFAR100 import read_cifar100
from FashionMNIST import read_fashion
from CIFAR10 import read_cifar10
from MNIST import read_mnist
##cifar batch =128  epoch = 50000
##mnist epoch=50  batch = 60000
def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def generate_sigle_model(path,kernel_list,channel_list,fc_list,is_complex=True):

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

	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/cifar100_meta/'+path)
	max_epoch = 35000
	batch_step = 128 
	train_batch,test_batch = read_cifar100(batch_step,1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,100,0,is_complex=is_complex)
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


def generate_sigle_summary(path,kernel_list,channel_list,fc_list,is_complex=True):

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

	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/cifar100_sigle_board/'+path)
	max_epoch = 50000
	batch_step = 128 
	train_batch,test_batch = read_cifar100(batch_step,1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,100,0,is_complex=is_complex)
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

	max_acc=0

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	test_x,test_y = next(test_batch)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')

	ans = []
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%500 ==0:
			summary,loss_value = sess.run([merged,loss], feed_dict={x:train_x,y:train_y})
			train_writer.add_summary(summary, i)
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			model.training = False
			acc = sess.run(accuracy, feed_dict={x:test_x,y:test_y})
			test_writer.add_summary(summary, i)
			test_accuracy = test(test_batch)

			ans.append(test_accuracy)
			model.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))

	train_writer.close()
	test_writer.close()
	print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()



class ImportGraph():
	"""  Importing and running isolated TF graph """
	def __init__(self, loc):
		model_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/cifar100_meta/'+loc)
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
	### Using the class ###
	_,test_batch = read_cifar100(1,1000)
	accuracy=0.0
	for i in range(10):
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
	




if __name__=='__main__':
	# path = sys.argv[1]
	# local_path=sys.argv[2]
	# is_complex = bool(int(sys.argv[3]))
	# is_training = bool(int(sys.argv[5]))
	# print(('board_path:%s\nmodel_path:%s')%(path,local_path))
	# print("is_complex:",is_complex)
	# print("is_training:",is_training)
	# cifar10(path,local_path,is_complex=True,is_training=True)



	# model_path_list = ['complex_model1','complex_model2']
	# restore(model_path_list)


	path_list = ['complex_model1','complex_model2','complex_model3','complex_model4',\
				'real_model1','real_model2','real_model3','real_model4']


	# kernel_list = [[5,5,3,3],[5,5,5,3],[5,5,3],[5,3,3],[5,5,3,3],[5,5,5,3],[5,5,3],[5,3,3]]
	# channel_list = [[128,128,64,64],[128,64,64,64],[128,128,64],[128,64,64],[128,128,64,64],[128,64,64,64],[128,128,64],[128,64,64]]
	# fc_list =[[192],[192],[192,81],[192,81],[192],[192],[192,81],[192,81]]
	# i = 0
	# if i>=4:
	# 	is_complex = False
	# else:
	# 	is_complex = True
	# generate_sigle_summary(path_list[i],kernel_list[i],channel_list[i],fc_list[i],is_complex)
	restore(['real_model1','real_model2','real_model3','real_model4'])



