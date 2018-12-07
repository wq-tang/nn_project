# import cifar10
import matplotlib.pyplot as plt
import os
import numpy as np 
import tensorflow as tf 
import time 
import cifar10_input
import math
from model import alexNet
from model import dy_model

class Predict():
	def __init__(self,graph_name,model_name):
		self.graph=tf.Graph()#为每个类(实例)单独创建一个graph
		with self.graph.as_default():
			self.saver=tf.train.import_meta_graph(graph_name)#创建恢复器
			#注意！恢复器必须要在新创建的图里面生成,否则会出错。
			self.sess=tf.Session(graph=self.graph)#创建新的sess
		with self.sess.as_default():
			with self.graph.as_default():
				self.saver.restore(self.sess,model_name)#从恢复点恢复参数
				self.y = tf.get_collection('pred_network'+graph_name[5])[0]
				self.X = self.graph.get_operation_by_name('input_x').outputs[0]


	def predict(self,x):
		return  sess.run(self.y,feed_dict={self.X:x})

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(i):
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')
	modelname = 'model' +str(i)
	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),modelname) 
	max_epoch = 30000
	batch_step = 100
	data_dir =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar-10-batches-bin')
	# cifar10.maybe_download_and_extract()
	train_images ,train_labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size = batch_step)
	test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=1000)
	x  = tf.placeholder(tf.float32,[None,24,24,3])
	y = tf.placeholder(tf.int32,[None])
	cnn1 = [[5,1,256],[5,1,128],[5,1,64],[3,2,64],[3,2,64]]
	pool1 = [[3,2],[3,2],[3,1],[3,2],[3,1]]
	shape_fc1 = [512,256,128,64]

	cnn2 = [[5,2,128],[5,1,64],[3,1,64],[3,2,64]]
	pool2 = [[3,1],[3,2],[3,1],[3,2]]
	shape_fc2 = [512,256,128,64]

	cnn3=[[5,1,256],[3,1,128],[3,1,64],[3,2,64],[3,2,64]]
	pool3 = [[3,2],[3,2],[3,1],[3,2],[3,1]]
	shape_fc3 = [512,256,128,64]

	cnn4 = [[3,1,256],[3,1,128],[5,1,64],[3,1,64],[3,2,64]]
	pool4 = [[3,1],[3,2],[3,2],[3,2],[3,1]]
	shape_fc4 = [512,128,64]

	cnn5 = [[5,1,256],[3,1,128],[5,1,64],[3,2,64]]
	pool5 = [[3,2],[3,1],[3,2],[3,2]]
	shape_fc5 = [512,256,128,64,64]

	cnn6 = [[5,1,256],[3,1,128],[5,1,64],[3,1,64],[3,1,64],[3,1,64],[3,1,32]]
	pool6 = [[3,1],[3,1],[3,1],[3,1],[3,1],[3,1],[3,1]]
	shape_fc6 = [256,128,64,64]
	
	cnn7 = [[5,1,256],[3,1,128],[5,1,64],[3,2,64]]
	pool7 = [[3,2],[3,1],[3,2],[3,2]]
	shape_fc7 = [1024,512,256,128,64,64]

	cnn8 = [[5,2,128],[5,1,64],[3,1,64],[3,2,64]]
	pool8 = [[3,1],[3,2],[3,1],[3,2]]
	shape_fc8 = [1024,512,256,128,64,64]
	
	shape_cnn=[cnn1,cnn2,cnn3,cnn4,cnn5,cnn6,cnn7,cnn8]
	shape_pool=[pool1,pool2,pool3,pool4,pool5,pool6,pool7,pool8]
	fc = [shape_fc1,shape_fc2,shape_fc3,shape_fc4,shape_fc5,shape_fc6,shape_fc7,shape_fc8]
	result = dy_model(x,10,i,shape_cnn[i],shape_pool[i],shape_fc[i]).fc3

	loss  = loss(result,y)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
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
		_,loss_value = sess.run([train_op,loss],feed_dict={x:train_x,y:train_y})
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

	saver = tf.train.Saver(max_to_keep=1)
	tf.add_to_collection('pred_network'+str(i), result)
	x_axis = list(np.arange(1,max_epoch/100+1)*100)
	save_path = saver.save(sess,model_path)
	ax.plot(x_axis,train_list,'b-','o',lw =5)
	ax.plot(x_axis,train_list,'r-','v',lw =5)
	
	for m in model:
		m.training = False
	precision = accuracy.eval(feed_dict={x:test_x, y: test_y})
	for m in model:
		m.training = True
	print('precision @1 = %.3f'%precision)


def get_path(name):
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),name)

def de(array):
	ans = []
	for k in array:
		 k = list(k)
		 ans.append(k.index(max(k)))
	return ans 
if __name__=='__main__':
	graph_name = []
	model_name = []
	result = []
	target = []
	model_num = 2
	for i in range(model_num):
		graph_name.append(get_path('model'+str(i)+'.meta'))
		model_name.append(get_path('model'+str(i)))
	data_dir =get_path('cifar-10-batches-bin')
	test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=10)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()
		test_x,test_y = sess.run([test_images,test_labels])
	for i in range(model_num):
		result.append(Predict(graph_name[i],model_name[i]).predict(test_x))
	print('2')
	# for i in range(model_num):
	# 	target = de(result[i])
	# print(target)
	# print(test_y)

