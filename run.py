# import cifar10
import matplotlib.pyplot as plt
import os
import numpy as np 
import tensorflow as tf 
import time 
import cifar10_input
import math
from model import alexNet
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'bagging.ckpt') 
def main():
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')

	max_epoch = 30000
	batch_step = 100
	model_num=8

	data_dir =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar-10-batches-bin')
	# cifar10.maybe_download_and_extract()
	train_images ,train_labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size = batch_step)
	test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=1000)
	x  = tf.placeholder(tf.float32,[None,24,24,3])
	y = tf.placeholder(tf.int32,[None])

	model = []
	for i in range(model_num):
		model.append(alexNet(x,10,i))
	models_result =list(map(lambda x:x.fc3+1e-10,model))
	angle =list(map(lambda x:np.pi*tf.nn.softmax(x),models_result))
	vector = list(zip(models_result,angle))
	vector_x = list(map(lambda x:x[0]*tf.cos(x[1]),vector))
	vector_y = list(map(lambda x:x[0]*tf.sin(x[1]),vector))
	

	vector_x_sum = tf.reduce_sum(vector_x,0)
	vector_y_sum = tf.reduce_sum(vector_y,0)
	result =tf.sqrt(tf.square(vector_x_sum)+tf.square(vector_y_sum))
	
	loss  = loss(result,y)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	optimizer= tf.train.AdamOptimizer(0.1**3)
	########
	gradient_all = optimizer.compute_gradients(loss)
	grads = [g for (g,v) in gradient_all if g is not None]
	########
	with tf.control_dependencies(update_ops):
		train_op =optimizer.minimize(loss)
	top_k_op = tf.nn.in_top_k(result,y,1)
	accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()



	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.ion()
	plt.show()
	train_list = []
	test_list=[]

	test_x,test_y = sess.run([test_images,test_labels])
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = sess.run([train_images,train_labels])
		_,loss_value,gradss = sess.run([train_op,loss,grads],feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%100 ==0:
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			for m in model:
				m.training = False
			test_accuracy = accuracy.eval(feed_dict={x:test_x, y: test_y})
			for m in model:
				m.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
			train_list.append(train_accuracy)
			test_list.append(test_accuracy)

	saver = tf.train.Saver()
	x_axis = list(np.arange(1,max_epoch/100+1)*100)
	save_path = saver.save(sess,model_path)
	ax.plot(x_axis,train_list,'b-','o',lw =5)
	ax.plot(x_axis,train_list,'r-','v',lw =5)
	
	for m in model:
		m.training = False
	precision = []
	for i in range(20):
		test_x,test_y = sess.run([test_images,test_labels])
		precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
	for m in model:
		m.training = True
	print(precision)
	print('precision @1 = %.3f'%np.mean(precision))




if __name__=='__main__':
	main()















