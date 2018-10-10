# import cifar10
import numpy as np 
import tensorflow as tf 
import time
import cifar10_input
import math
from model import alexNet

def main():
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')

	max_epoch = 3
	batch_step = 128
	data_dir = 'cifar-10-batches-bin'
	# cifar10.maybe_download_and_extract()
	train_images ,train_labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size = batch_step)
	test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=batch_step)
	x  = tf.placeholder(tf.float32,[batch_step,24,24,3])
	y = tf.placeholder(tf.int32,[batch_step])

	model = alexNet(x,10)
	loss  = loss(model.fc3,y)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
	top_k_op = tf.nn.in_top_k(model.fc3,y,1)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = sess.run([train_images,train_labels])
		_,loss_value = sess.run([train_op,loss],feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%100 ==0:
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)

			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

	model.training = False
	num_examples = 10000
	num_iter = int(math.ceil(num_examples/batch_step))
	true_count = 0
	total_sample_count = num_iter*batch_step
	step = 0

	while step<num_iter:
		test_x,test_y = sess.run([test_images,test_labels])
		prediction = sess.run([top_k_op],feed_dict={x:test_x,y:test_y})
		true_count += np.sum(prediction)
		step+=1
	precision = true_count/total_sample_count

	print('precision @1 = %.3f'%precision)

if __name__=='__main__':
	main()
