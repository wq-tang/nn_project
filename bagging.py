import os
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

def cifar10(path,local_path,is_complex,model_num,is_training=True):
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

	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'/'+local_path +'/cifar10.ckpt')
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

	saver=tf.train.Saver(max_to_keep=1)
	max_acc=0

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	if is_training:
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(log_dir + '/test')
		ans = []
		test_x,test_y = sess.run([test_images,test_labels])
		for i in range(max_epoch):
			start_time = time.time()
			train_x,train_y = sess.run([train_images,train_labels])
			_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
			duration = time.time() - start_time
			if i%200 ==0:
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
				test_accuracy = test()
				if test_accuracy>max_acc:
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
		train_writer.close()
		test_writer.close()
		

		print('precision @1 = %.5f'%np.mean(ans[-10:]))

		return None
	else:
		model_file=tf.train.latest_checkpoint('ckpt/')
		image,lable=cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=10000)
		x_,y_ = sess.run([image,lable])

		saver.restore(sess,model_path)
		test_accuracy=test()
		predict = sess.run(models_result,feed_dict={x:x_,y:y_})

		train_x,train_y = sess.run([train_images,train_labels])
		train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
		print('test_accuracy:%f,train_accuracy:%f'%(test_accuracy,train_accuracy))
		return predict

	sess.close()





if __name__=='__main__':
	path = sys.argv[1]
	is_complex = bool(int(sys.argv[3]))
	model_num = int(sys.argv[4])
	local_path=sys.argv[2]
	is_training = bool(int(sys.argv[5]))
	print(path)
	print(is_complex)
	print(model_num)
	cifar10(path=path,local_path=local_path ,is_complex=is_complex,model_num=model_num,is_training = is_training)

