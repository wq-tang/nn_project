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

def build_model(x,class_num,submodel_num,is_complex):
	if is_complex:
		wr = [ 0 for i in range(submodel_num)]
		wi = [ 0 for i in range(submodel_num)]
		out = [0.0,0.0]
		for i in range(submodel_num):
			wr[i] = tf.get_variable("wr"+str(i+1), shape = [1, class_num], dtype = tf.float32,\
							initializer = tf.constant_initializer(1))
			wi[i] = tf.get_variable("wi"+str(i+1), shape = [1, class_num], dtype = tf.float32,\
							initializer = tf.constant_initializer(0))
			out[0] += tf.multiply(x[i][:,0,:],wr[i]) - tf.multiply(x[i][:,1,:],wi[i])
			out[1] += tf.multiply(x[i][:,0,:],wi[i]) + tf.multiply(x[i][:,1,:],wr[i])

		return tf.square(out[0])+tf.square(out[1])
	else:
		w = [ 0 for i in range(submodel_num)]
		out = [0.0]
		for i in range(submodel_num):
			w[i] = tf.get_variable("wr"+str(i+1), shape = [1, class_num], dtype = tf.float32,\
							initializer = tf.constant_initializer(1))
			out += tf.multiply(x[i],w[i])
		return out



def Secondary_net(model_name,combine_list,is_complex):
	#修改文件读取名字，读取函数
	### Using the class ###
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')

	def test(test_data):
		precision=[]
		for i in range(10):
			tests = [0 for i in range(len(combine_list))]
			for i in range(len(combine_list)-1):
				tests[i] ,_ = next(test_data[i])
			tests[-1],test_y = next(test_data[len(combine_list)-1])

			if len(combine_list)==4:
				out  = accuracy.eval( feed_dict={x[0]:tests[0],x[1]:tests[1],x[2]:tests[2],x[3]:tests[3],y:test_y})#
			elif len(combine_list)==3:
				out  = accuracy.eval( feed_dict={x[0]:tests[0],x[1]:tests[1],x[2]:tests[2],y:test_y})#
			else:
				out  = accuracy.eval( feed_dict={x[0]:tests[0],x[1]:tests[1],y:test_y})#
 
			precision.append(out)
		return np.mean(precision)

	#修改读取文件的函数以及文件名称
	#修改输出路径
	if is_complex:
		board_tail = model_name+'/complex'+''.join(combine_list)
	else:
		board_tail = model_name+'/real'+''.join(combine_list)

	log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'st_submodel/alex/'+board_tail)
	if is_complex:
		tail = 'complex'
	else:
		tail = 'real'
	max_epoch = 2000  #修改
	batch_step = 128
	class_num = 10   ###修改
	train_data = []
	test_data = []
	for i in range(len(combine_list)):
		batchTrain, batchTest = read_data(is_complex,'stacking_data/alex/'+model_name+'/'+tail+combine_list[i]+'.h5',batch_step,1000,train_shuffle=False)
		train_data.append(batchTrain)
		test_data.append(batchTest)

	y = tf.placeholder(tf.int32,[None])
	x = [0 for i in range(len(combine_list))]
	
	for i in range(len(combine_list)):
		if is_complex:
			x[i] = tf.placeholder(tf.float32,[None,2,class_num],name = 'input_x'+str(i+1))
		else:
			x[i] = tf.placeholder(tf.float32,[None,class_num],name = 'input_x'+str(i+1))

	#####################################################################	
	out = build_model(x,class_num,len(combine_list),is_complex)
	######################################################################
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
	tests = [0 for i in range(len(combine_list))]
	for i in range(len(combine_list)-1):
		tests[i] ,_ = next(test_data[i])
	tests[-1],test_y = next(test_data[len(combine_list)-1])


	trains = [0 for i in range(len(combine_list))]

	for i in range(max_epoch):
		start_time = time.time()
		for k in range(len(combine_list)-1):
			trains[k] ,_ = next(train_data[k])
		trains[-1 ],train_y = next(train_data[len(combine_list)-1])

		if len(combine_list)==4:
			_ = sess.run( train_op, feed_dict={x[0]:trains[0],x[1]:trains[1],x[2]:trains[2],x[3]:trains[3],y:train_y})#
		elif len(combine_list)==3:
			_ = sess.run( train_op, feed_dict={x[0]:trains[0],x[1]:trains[1],x[2]:trains[2],y:train_y})#
		else:
			_ = sess.run( train_op, feed_dict={x[0]:trains[0],x[1]:trains[1],y:train_y})#

		duration = time.time() - start_time
		if i%200 ==0:
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch_step)')
			
			if len(combine_list)==4:
				summary,loss_value  = sess.run( [merged,loss], feed_dict={x[0]:trains[0],x[1]:trains[1],x[2]:trains[2],x[3]:trains[3],y:train_y})#
			elif len(combine_list)==3:
				summary,loss_value  = sess.run( [merged,loss], feed_dict={x[0]:trains[0],x[1]:trains[1],x[2]:trains[2],y:train_y})#
			else:
				summary,loss_value  = sess.run( [merged,loss], feed_dict={x[0]:trains[0],x[1]:trains[1],y:train_y})#
			
			train_writer.add_summary(summary, i)

			if len(combine_list)==4:
				train_accuracy  = accuracy.eval( feed_dict={x[0]:trains[0],x[1]:trains[1],x[2]:trains[2],x[3]:trains[3],y:train_y})#
			elif len(combine_list)==3:
				train_accuracy  = accuracy.eval( feed_dict={x[0]:trains[0],x[1]:trains[1],x[2]:trains[2],y:train_y})#
			else:
				train_accuracy  = accuracy.eval( feed_dict={x[0]:trains[0],x[1]:trains[1],y:train_y})#
 

			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			if len(combine_list)==4:
				summary, acc = sess.run( [merged,accuracy], feed_dict={x[0]:tests[0],x[1]:tests[1],x[2]:tests[2],x[3]:tests[3],y:test_y})#
			elif len(combine_list)==3:
				summary, acc  = sess.run( [merged,accuracy], feed_dict={x[0]:tests[0],x[1]:tests[1],x[2]:tests[2],y:test_y})#
			else:
				summary, acc  = sess.run( [merged,accuracy], feed_dict={x[0]:tests[0],x[1]:tests[1],y:test_y})#

			test_writer.add_summary(summary, i)
			test_accuracy = test(test_data)
			ans.append(test_accuracy)
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
	train_writer.close()
	test_writer.close()
	print('precision @1 = %.5f'%np.mean(ans[-1:]))

if __name__=='__main__':
	combine_list = ['1','2']#集成模型编号
	model_name='cifar10'
	is_complex = True
	Secondary_net(model_name,combine_list,is_complex)