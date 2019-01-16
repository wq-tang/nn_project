# import cifar10
import os
import numpy as np 
import tensorflow as tf 
import time
import cifar10_input
import math
from model import alexNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'bagging.ckpt') 
def main():
	def loss(logits,y):
		labels =tf.cast(y,tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'),name='total_loss')
	def test():
		precision=[]
		for i in range(20):
			test_x,test_y = sess.run([test_images,test_labels])
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)

	max_epoch = 30000
	batch_step = 100
	model_num=1

	data_dir =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cifar-10-batches-bin')
	# cifar10.maybe_download_and_extract()
	train_images ,train_labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size = batch_step)
	test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=1000)
	x  = tf.placeholder(tf.float32,[None,24,24,3])
	y = tf.placeholder(tf.int32,[None])

	model = [alexNet(x,10,0)]
	models_result =model[0].fc3
	loss  = loss(models_result,y)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

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
			# pre = test()
			y_hat = models_result[:10].eval(feed_dict={x:test_x, y: test_y})
			test_accuracy = accuracy.eval(feed_dict={x:test_x, y: test_y})
			for m in model:
				m.training = True
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
			# print('pre:%g'%pre)
			# print(y_hat)
			train_list.append(train_accuracy)
			test_list.append(test_accuracy)
			# if test_accuracy>0.95:
			# 	break

	saver = tf.train.Saver()
	save_path = saver.save(sess,model_path)
	
	for m in model:
		m.training = False
	pre= test()
	for m in model:
		m.training = True
	print('precision @1 = %.3f'%pre)


def get_path(name):
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),name)

def de(array):
	ans = []
	for k in array:
		 k = list(k)
		 ans.append(k.index(max(k)))
	return ans 
	
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
				self.y = tf.get_collection('pred_network'+modelname[5])[0]
				self.X = graph.get_operation_by_name('input_x').outputs[0]


	def predict(self,x):
		return  sess.run(self.y,feed_dict={self.X:x})

if __name__=='__main__':
	main()



	# graph_name=get_path('bagging.ckpt.meta')
	# model_name=get_path('bagging.ckpt.data-00000-of-00001')
	# data_dir =get_path('cifar-10-batches-bin')
	# test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=1000)
	# sess = tf.InteractiveSession()
	# test_x,test_y = sess.run([test_images,test_labels])
	# result= Predict(graph_name,model_name).predict(test_x)
	# target = de(result)
	# print(target)
	# print(test_y)



	# graph_name = []
	# model_name = []
	# result = []
	# target = []
	# model_num = 2
	# for i in range(model_num):
	# 	graph_name.append(get_path('model'+str(i)+'.meta'))
	# 	model_name.append(get_path('model'+str(i)+'data-00000-of-00001'))
	# data_dir =get_path('cifar-10-batches-bin')
	# test_images,test_labels = cifar10_input.inputs(eval_data = True,data_dir=data_dir,batch_size=10000)
	# with tf.Session() as sess():
	# 	test_x,test_y = sess.run([test_images,test_labels])
	# for i in range(model_num):
	# 	result.append(Predict(graph_name[i],model_name[i]).predict(test_x))
	# for i in range(model_num):
	# 	target = de(result[i])
	# print(target)
	# print(test_y)

