import os
import numpy as np 
import tensorflow as tf 
import time
import math
import sys
import h5py

from alexnet_model import complex_net
from resnet_model import Resnet
from READ_DATA import read_data



def real_model(x,class_num):
	w1 = tf.get_variable("w1", shape = [1, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True,dtype=tf.float32))
	w2 = tf.get_variable("w2", shape = [1, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, dtype=tf.float32))
	w3 = tf.get_variable("w3", shape = [1, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, dtype=tf.float32))
	w4 = tf.get_variable("w4", shape = [1, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, dtype=tf.float32))
	out = tf.multiply(x[0],w1)+tf.multiply(x[1],w2)+tf.multiply(x[2],w3)+tf.multiply(x[3],w4)
	return out

def complex_model(x,class_num):
	w1 = tf.get_variable("w1", shape = [2, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True,dtype=tf.float32))
	w2 = tf.get_variable("w2", shape = [2, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, dtype=tf.float32))
	w3 = tf.get_variable("w3", shape = [2, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, dtype=tf.float32))
	w4 = tf.get_variable("w4", shape = [2, class_num], dtype = tf.float32,\
                    initializer = tf.contrib.layers.xavier_initializer( uniform=True, dtype=tf.float32))
	out = tf.multiply(x[0],w1)+tf.multiply(x[1],w2)+tf.multiply(x[2],w3)+tf.multiply(x[3],w4)
	R = out[:,0,:]
	I = out[:,1,:]
	return tf.sqrt(tf.square(R)+tf.square(I))

def Secondary_net(model_name,is_complex):
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
			test_x1,_ = next(test_data[0])
			test_x2,_ = next(test_data[1])
			test_x3,_ = next(test_data[2])
			test_x4,test_y = next(test_data[3])
			precision.append(accuracy.eval(feed_dict={x1:test_x1,x2:test_x2,x3:test_x3,x4:test_x4, y: test_y}))
		return np.mean(precision)

	#修改读取文件的函数以及文件名称
	#修改输出路径
	if is_complex:
		tail = model_name+'_complex_board'
	else:
		tail = model_name+'_real_board'
	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/stacking/'+tail)
	if is_complex:
		tail = 'complex'
	else:
		tail = 'real'
	max_epoch = 30000  #修改
	batch_step = 128
	class_num = 10   ###修改
	train_data = []
	test_data = []
	for i in range(4):
		batchTrain, batchTest = read_data(model_name+'_'+tail+str(i+1)+'.h5',batch_step,1000)
		train_data.append(batchTrain)
		test_data.append(batchTest)

	y = tf.placeholder(tf.int32,[None])

	if is_complex:
		x1 = tf.placeholder(tf.float32,[None,2,class_num],name = 'input_x1')
		x2 = tf.placeholder(tf.float32,[None,2,class_num],name = 'input_x2')
		x3 = tf.placeholder(tf.float32,[None,2,class_num],name = 'input_x3')
		x4 = tf.placeholder(tf.float32,[None,2,class_num],name = 'input_x4')
		out = complex_model([x1,x2,x3,x4],class_num)
	else:
		x1 = tf.placeholder(tf.float32,[None,class_num],name = 'input_x1')
		x2 = tf.placeholder(tf.float32,[None,class_num],name = 'input_x2')
		x3 = tf.placeholder(tf.float32,[None,class_num],name = 'input_x3')
		x4 = tf.placeholder(tf.float32,[None,class_num],name = 'input_x4')
		out = real_model([x1,x2,x3,x4],class_num)


	with tf.name_scope('loss'):
		loss  = loss(out,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(out,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)


	sess = tf.InteractiveSession()
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	ans = []
	test_x1,_ = next(test_data[0])
	test_x2,_ = next(test_data[1])
	test_x3,_ = next(test_data[2])
	test_x4,test_y = next(test_data[3])

	for i in range(max_epoch):
		start_time = time.time()
		train_x1,_ = next(train_data[0])
		train_x2,_ = next(train_data[1])
		train_x3,_ = next(train_data[2])
		train_x4,train_y = next(train_data[3])
		_ = sess.run( train_op, feed_dict={x1:train_x1,x2:train_x2,x3:train_x3,x4:train_x4,y:train_y})
		duration = time.time() - start_time
		if i%400 ==0:
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch_step)')
			summary,loss_value = sess.run([merged,loss], feed_dict={x1:train_x1,x2:train_x2,x3:train_x3,x4:train_x4,y:train_y})
			train_writer.add_summary(summary, i)
			train_accuracy = accuracy.eval(feed_dict={x1:train_x1,x2:train_x2,x3:train_x3,x4:train_x4, y:train_y})
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
			summary, acc = sess.run([merged, accuracy], feed_dict={x1:test_x1,x2:test_x2,x3:test_x3,x4:test_x4,y:test_y})
			test_writer.add_summary(summary, i)
			test_accuracy = test(test_data)
			ans.append(test_accuracy)
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
	train_writer.close()
	test_writer.close()
	print('precision @1 = %.5f'%np.mean(ans[-10:]))

if __name__=='__main__':
	Secondary_net('MNIST',is_complex=False)