import os
import numpy as np 
import tensorflow as tf 
import time
import cifar10_input
import math
from comparable_model import complex_net
from tensorflow.examples.tutorials.mnist import input_data
##cifar batch =128  epoch = 50000
##mnist epoch=50  bathch = 60000


def mnist():
	def test(images,labels,accuracy):
		p = 0
		for i in range(10):
			xs = images[i*1000:(i+1)*1000]
			ys = labels[i*1000:(i+1)*1000]
			p+= accuracy.eval(feed_dict={x:xs, y:ys})
		return p/10
	mnist_data_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mnist') 
	mnist=input_data.read_data_sets(mnist_data_folder,one_hot=True)
	epoch = 50
	batch = 100
	x  = tf.placeholder(tf.float32,[None,784])
	y = tf.placeholder(tf.int32,[None,10])

	model = [complex_net(tf.reshape(x,[-1,28,28,1]),10,0)]
	models_result =model[0].out
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(models_result, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=models_result))
	train_step = tf.train.AdamOptimizer(0.1**3).minimize(cross_entropy) 
	# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	ans = []
	for i in range(epoch*600):
		start_time = time.time()
		train_x, train_y = mnist.train.next_batch(batch)
		_,loss_value = sess.run([train_step,cross_entropy],feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%100 ==0:
			examples_per_sec = batch/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			for m in model:
				m.training = False
			test_accuracy = test(mnist.test.images,mnist.test.labels,accuracy)
			ans.append(test_accuracy)
			for m in model:
				m.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))

	print('precision @1 = %.5f'%np.mean(ans[-100:]))
if __name__=='__main__':
	mnist()
