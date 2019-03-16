import os
from cifar10 import read_cifar10
import numpy as np 
import tensorflow as tf 
import time
import cifar10_input
import math
from comparable_model import complex_net
from tensorflow.examples.tutorials.mnist import input_data
import sys
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

def cifar10(path,local_path,kernel_list,channel_list,fc_list,is_complex=True,is_training=True):
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits+0.1**8,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')
	def test():
		precision=[]
		for i in range(40):
			test_x,test_y = sess.run([test_images,test_labels])
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)


	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),local_path)
	max_epoch = 50000
	batch_step = 128
	log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar10_board/'+path)
	data_dir =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar-10-batches-bin')
	train_images ,train_labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size = batch_step)
	test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3])
	tf.summary.image('inputs', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,10,0,is_complex=is_complex)
	model.diff_net([x,x],name=path,kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)
	if is_complex:
		models_result = tf.sqrt(tf.square(model.out[0])+tf.square(model.out[1]))
	else:
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

	saver=tf.train.Saver(max_to_keep=1)
	max_acc=0

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	test_x,test_y = sess.run([test_images,test_labels])
	if is_training:
		# merged = tf.summary.merge_all()
		# train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
		# test_writer = tf.summary.FileWriter(log_dir + '/test')
		ans = []
		for i in range(max_epoch):
			start_time = time.time()
			train_x,train_y = sess.run([train_images,train_labels])
			_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
			duration = time.time() - start_time
			if i%500 ==0:
				loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
				# train_writer.add_summary(summary, i)
				examples_per_sec = batch_step/duration
				sec_per_batch = float(duration)
				format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
				print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

				train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
				model.training = False
				acc = sess.run(accuracy, feed_dict={x:test_x,y:test_y})
				# test_writer.add_summary(summary, i)
				test_accuracy = test()
				if i>max_epoch*0.9 and test_accuracy>max_acc:
					max_acc=test_accuracy
					saver.save(sess,model_path,global_step=i+1)
				ans.append(test_accuracy)
				# test_accuracy = accuracy.eval(feed_dict={x:test_x, y: test_y})
				model.training = True
				print( "step %d, training accuracy %g"%(i, train_accuracy))
				print( "step %d,test accuracy %g"%(i,test_accuracy))
				# print('pre:%g'%pre)
				# print(y_hat)
				# if test_accuracy>0.95:
				# 	break
		# train_writer.close()
		# test_writer.close()
		

		print('precision @1 = %.5f'%np.mean(ans[-10:]))

		return np.mean(ans[-5:])
	else:
		# model_file=tf.train.latest_checkpoint('ckpt/')
		saver.restore(sess,model_path)
		# test_accuracy=test()
		predict = sess.run(model.out,feed_dict={x:test_x,y:test_y})

		# train_x,train_y = sess.run([train_images,train_labels])
		# train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
		# print('test_accuracy:%f,train_accuracy:%f'%(test_accuracy,train_accuracy))
		return predict

	sess.close()





if __name__=='__main__':
	# path = sys.argv[1]
	# local_path=sys.argv[2]
	# is_complex = bool(int(sys.argv[3]))
	# is_training = bool(int(sys.argv[5]))
	# print(('board_path:%s\nmodel_path:%s')%(path,local_path))
	# print("is_complex:",is_complex)
	# print("is_training:",is_training)
	# cifar10(path,local_path,is_complex=True,is_training=True)
	path_list = ['cifar10_complex_model1','cifar10_complex_model2','cifar10_complex_model3','cifar10_complex_model4',\
				'cifar10_real_model1','cifar10_real_model2','cifar10_real_model3','cifar10_real_model4']

	model_path_list = ['mynet/cifar10_complex_model1','mynet/cifar10_complex_model2','mynet/cifar10_complex_model3','mynet/cifar10_complex_model4',\
						'mynet/cifar10_real_model1','mynet/cifar10_real_model2','mynet/cifar10_real_model3',\
						'mynet/cifar10_real_model4']
	kernel_list = [[5,3,3],[5,5,2],[5,5],[5,3],[5,3,3],[5,5,2],[5,5],[5,3]]
	channel_list = [[128,64,64],[128,64,64],[128,64],[128,64],[128,64,64],[128,64,64],[128,64],[128,64]]
	fc_list =[[100],[256,128],[100,50],[100],[100],[256,128],[100,50],[100]]
	is_complex = True
	i = 0
	ans = cifar10(path_list[i],model_path_list[i],kernel_list[i],channel_list[i],fc_list[i],is_complex,is_training=True)
	print(ans)
	# res1=cifar10(path='rm_test',local_path='rm_test/cifar10_1.ckpt-401' ,is_complex=False,model_num=1,is_training = False)
	# res2=cifar10(path='rm_test',local_path='rm_test/cifar10_2.ckpt-401' ,is_complex=False,model_num=1,is_training = False)
	# train_step,test_step= read_cifar10(10000)
	# test_data = next(test_step)

