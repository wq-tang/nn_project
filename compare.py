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

def cifar(path,is_complex,model_num):
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

	max_epoch = 50000
	batch_step = 128
	file_name = 'CIFAR100.h5'
	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/cifar100_board/'+path)
	train_batch,test_batch=read_cifar100('data/'+file_name,batch_step,1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3])
	tf.summary.image('inputs', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,100,0,is_complex=is_complex)
	if path[:7] == 'compare':
		model.build_compare_for_cifar10(model_num)
	else:
		model.build_CNN_for_cifar10(model_num)
	models_result =model.out
	with tf.name_scope('loss'):
		loss  = loss(models_result,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')

	tf.global_variables_initializer().run()
	# tf.train.start_queue_runners()


	ans = []
	test_x,test_y = next(test_batch)
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
			summary, acc = sess.run([merged, accuracy], feed_dict={x:test_x,y:test_y})
			test_writer.add_summary(summary, i)
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			model.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))

	train_writer.close()
	test_writer.close()
	

	print('precision @1 = %.5f'%np.mean(ans[-5:]))


def mnist(path,is_complex,model_num):
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

	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/mnist_board/'+path)
	epoch = 50
	batch_step = 128
	file_name = 'MNIST.h5'
	train_batch,test_batch=read_mnist('data/'+file_name,batch_step,1000)
	x  = tf.placeholder(tf.float32,[None,784])
	y = tf.placeholder(tf.int32,[None,10])
	tf.summary.image('input', x, 10)
	model = complex_net(x,10,0,is_complex=is_complex)
	if path[:7] == 'compare':
		model.build_compare_for_mnist(model_num)
	else:
		model.build_CNN_for_mnist(model_num)
	models_result =model.out
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(models_result, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=models_result))
	tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(0.1**3).minimize(cross_entropy) 
	# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	ans = []
	test_x,test_y = next(test_batch)
	for i in range(epoch*600):
		start_time = time.time()
		train_x, train_y =  next(train_batch)
		_ = sess.run( train_step, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%200 ==0:
			examples_per_sec = batch/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			summary,loss_value = sess.run([merged,cross_entropy], feed_dict={x:train_x,y:train_y})
			train_writer.add_summary(summary, i)
			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
			model.training = False
			summary, acc = sess.run([merged, accuracy], feed_dict={x:test_x,y:test_y})
			test_writer.add_summary(summary, i)
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			model.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
	train_writer.close()
	test_writer.close()
	print('precision @1 = %.5f'%np.mean(ans[-10:]))



if __name__=='__main__':
	# path = sys.argv[1]
	# is_complex = bool(int(sys.argv[2]))
	# model_num = int(sys.argv[3])
	# print(path)
	# print(is_complex)
	# print(model_num)
	# cifar(path=path,is_complex=is_complex,model_num=model_num)
	mnist('test',True,1)	