import os
import numpy as np 
import tensorflow as tf 
import time
import math
import sys
import h5py

from alexnet_model import complex_net
from resnet_model import Resnet

from CIFAR100 import read_cifar100
from FashionMNIST import read_fashion
from CIFAR10 import read_cifar10
from MNIST import read_mnist
from READ_DATA import read_data
# def wrrite_file(train_data,test_data,file_name):
# 	with h5py.File(file_name,'w') as f:
# 		f.create_group('Train')
# 		f.create_group('Test')
# 		f.create_dataset('Train/data',data = train_data[0])
# 		f.create_dataset('Train/labelsFine',data = train_data[1])
# 		f.create_dataset('Test/data',data = test_data[0])
# 		f.create_dataset('Test/labelsFine',data = test_data[1])

def wrrite_file(train_data,test_data,file_name):
	with h5py.File(file_name,'w') as f:
		f.create_group('Train')
		f.create_group('Test')
		f.create_dataset('Train/images',data = train_data[0])
		f.create_dataset('Train/labels',data = train_data[1])
		f.create_dataset('Test/images',data = test_data[0])
		f.create_dataset('Test/labels',data = test_data[1])

# def stacking(path,kernel_list,channel_list,fc_list,is_complex=True):
# 	def loss(logits,y):
# 		labels =tf.cast(y,tf.int64)
# 		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = y,name='cross_entropy_per_example')
# 		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
# 		tf.add_to_collection('losses',cross_entropy_mean)
# 		return tf.add_n(tf.get_collection('losses'),name='total_loss')

# 	def test(test_batch):
# 		precision=[]
# 		for i in range(10):
# 			test_x,test_y = next(test_batch)
# 			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
# 		return np.mean(precision)
# 	#修改模型以及对应的方法
# 	#修改读取文件的函数以及文件名称
# 	#修改输出路径
# 	#修改模型中的输出参数
# 	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mynet/cifar10_meta_bagging/'+path)
# 	max_epoch = 30000
# 	batch_step = 128 
# 	file_name = 'CIFAR10_model'+path[-1]+'.h5'
# 	train_batch,test_batch = read_cifar10('data/'+file_name,batch_step,1000)
# 	with tf.name_scope("inputs"):
# 		x  = tf.placeholder(tf.float32,[None,24,24,3],name = 'input_x')
# 	tf.summary.image('input_x', x, 10)
# 	y = tf.placeholder(tf.int32,[None])

# 	model = complex_net(x,10,0,is_complex=is_complex)
# 	model.diff_net(x,name=path,kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)
# 	out_result= tf.add(model.out,0.0,name = 'out')
# 	if is_complex:
# 		models_result = tf.sqrt(tf.square(out_result[0])+tf.square(out_result[1]))
# 	else:
# 		models_result =out_result
# 	tf.add_to_collection("model_out", out_result)
# 	with tf.name_scope('loss'):
# 		loss  = loss(models_result,y)
# 	tf.summary.scalar('loss', loss)
# 	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# 	with tf.control_dependencies(update_ops):
# 		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
# 		tf.add_to_collection("train_op", train_op)
# 		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 	top_k_op = tf.nn.in_top_k(models_result,y,1)
# 	with tf.name_scope('accuracy'):
# 		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
# 	tf.summary.scalar('accuracy', accuracy)
# 	sess = tf.InteractiveSession()

# 	saver=tf.train.Saver(max_to_keep=1)
# 	max_acc=0

# 	tf.global_variables_initializer().run()
# 	tf.train.start_queue_runners()
# 	test_x,test_y = next(test_batch)


# 	ans = []
# 	for i in range(max_epoch):
# 		start_time = time.time()
# 		train_x,train_y = next(train_batch)
# 		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
# 		duration = time.time() - start_time
# 		if i%500 ==0:
# 			loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
# 			examples_per_sec = batch_step/duration
# 			sec_per_batch = float(duration)
# 			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
# 			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))

# 			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
# 			model.training = False
# 			acc = sess.run(accuracy, feed_dict={x:test_x,y:test_y})
# 			test_accuracy = test(test_batch)
# 			if i>max_epoch*0.8 and test_accuracy>max_acc:
# 				max_acc=test_accuracy
# 				saver.save(sess,model_path+'.ckpt')
# 			ans.append(test_accuracy)
# 			model.training = True
# 			print( "step %d, training accuracy %g"%(i, train_accuracy))
# 			print( "step %d,test accuracy %g"%(i,test_accuracy))

	

# 	print('precision @1 = %.5f'%np.mean(ans[-5:]))
# 	sess.close()


#生成初级学习模型
def generate_Primary_net_cifar(model_shape,model_tag,is_complex):
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
			test_x,test_y = next(test_batch)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)
	#修改模型以及对应的方法
	#修改读取文件的函数以及文件名称
	#修改输出路径
	#修改模型中的输出参数
	if is_complex:
		local_path = 'complex'+model_shape[0] #model_shape=【'1'，kernel,chanel,fc】
	else:
		local_path = 'real'+model_shape[0]
	
	max_epoch = 50000											#tag = i 那么就用缺第i份的数据训练，用第i份生成新的数据。
	batch_step = 128 
	#########一些路径#######
	path,kernel_list,channel_list,fc_list = model_shape
	model_path =os.path.join(os.path.dirname(os.path.abspath(__file__)),
		'stacking_submodel/alex/cifar10/'+local_path+'-'+str(model_tag))#model_tag 是当前子模型的第tag个分模型
	file_name = 'split_file/CIFAR10_split_4-'+str(model_tag)+'.h5'
	# validation_name =  'split_file/CIFAR10_split4'+str(model_tag)+'.h5'


	# valid_train,_ = read_cifar10(validation_name,1250,1000)#10次取完
	train_batch,test_batch = read_cifar10(file_name,batch_step,1000)
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,24,24,3],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,10,0,is_complex=is_complex)
	model.diff_net(x,name=local_path+str(model_tag),kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)
	out_result= tf.add(model.out,0.0,name = 'out')
	if is_complex:
		models_result = tf.square(out_result[0])+tf.square(out_result[1])
	else:
		models_result =out_result

	tf.add_to_collection("model_out", out_result)
	with tf.name_scope('loss'):
		loss  = loss(models_result,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		tf.add_to_collection("train_op", train_op)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()

	saver=tf.train.Saver(max_to_keep=1)

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	

	ans = []
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%1000 ==0:
			loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
			model.training=False
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			model.training = False
			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d, test accuracy %g"%(i, test_accuracy))

	
	saver.save(sess,model_path+'.ckpt')

	# if is_complex:
	# 	axis = 1
	# else:
	# 	axis=0

	# valid_data=[]
	# valid_label = []
	# test_data=[]
	# test_label = []
	# for i in range(10):
	# 	test_x,test_y = next(test_batch)
	# 	valid_train_data,valid_train_label = next(valid_train)
	# 	valid_data.append((sess.run(out_result,feed_dict={x:valid_train_data})))
	# 	valid_label.append(valid_train_label)
	# 	test_label.append(test_y)
	# 	test_data.append((sess.run(out_result,feed_dict={x:test_x})))

	# print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()


	# return np.mean(ans[-5:]), np.concatenate(valid_data,axis=axis),np.concatenate(valid_label,axis=0),\
	# np.concatenate(test_data,axis=axis),np.concatenate(test_label,axis=0)


def generate_Primary_net_mnist(model_shape,model_tag,is_complex):
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
			test_x,test_y = next(test_batch)
			precision.append(accuracy.eval(feed_dict={x:test_x, y: test_y}))
		return np.mean(precision)
	#修改模型以及对应的方法
	#修改读取文件的函数以及文件名称
	#修改输出路径
	#修改模型中的输出参数
	if is_complex:
		local_path = 'complex'+model_shape[0]
	else:
		local_path = 'real'+model_shape[0]

	path,kernel_list,channel_list,fc_list = model_shape
	model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\
		'mynet/stacking/fashion/'+local_path+str(model_tag))   ##修改
	max_epoch = 50000                                        ##修改
	batch_step = 128 
	file_name = 'FashionMNIST_st4-'+str(model_tag)+'.h5'            ##修改
	validation_name =  'FashionMNIST_st4'+str(model_tag)+'.h5'      ##修改
	valid_train,_ = read_fashion('k_data/'+validation_name,1500,1000)#10次取完  ##修改
	train_batch,test_batch = read_fashion('k_data/'+file_name,batch_step,1000) ##修改
	with tf.name_scope("inputs"):
		x  = tf.placeholder(tf.float32,[None,28,28,1],name = 'input_x')
	tf.summary.image('input_x', x, 10)
	y = tf.placeholder(tf.int32,[None])

	model = complex_net(x,10,0,is_complex=is_complex)       ##修改
	model.diff_net(x,name=local_path+str(model_tag),kernel_list =kernel_list ,channel_list=channel_list,fc_list=fc_list)##修改
	out_result= tf.add(model.out,0.0,name = 'out')
	if is_complex:
		models_result = tf.sqrt(tf.square(out_result[0])+tf.square(out_result[1]))
	else:
		models_result =out_result
	tf.add_to_collection("model_out", out_result)
	with tf.name_scope('loss'):
		loss  = loss(models_result,y)
	tf.summary.scalar('loss', loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = tf.train.AdamOptimizer(0.1**3).minimize(loss)
		tf.add_to_collection("train_op", train_op)
		# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	top_k_op = tf.nn.in_top_k(models_result,y,1)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	sess = tf.InteractiveSession()

	saver=tf.train.Saver(max_to_keep=1)

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()
	
	ans=[]
	test_x,test_y = next(test_batch)
	for i in range(max_epoch):
		start_time = time.time()
		train_x,train_y = next(train_batch)
		_ = sess.run(train_op, feed_dict={x:train_x,y:train_y})
		duration = time.time() - start_time
		if i%1000 ==0:
			loss_value = sess.run(loss, feed_dict={x:train_x,y:train_y})
			examples_per_sec = batch_step/duration
			sec_per_batch = float(duration)
			format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
			model.training = False
			acc = sess.run(accuracy, feed_dict={x:test_x,y:test_y})
			test_accuracy = test(test_batch)
			ans.append(test_accuracy)
			model.training = True
			train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
			print( "step %d, training accuracy %g"%(i, train_accuracy))
			print( "step %d,test accuracy %g"%(i,test_accuracy))
	
	saver.save(sess,model_path+'.ckpt')

	if is_complex:
		axis = 1
	else:
		axis=0

	valid_data=[]
	valid_label = []
	test_data=[]
	test_label = []
	for i in range(10):
		test_x,test_y = next(test_batch)
		valid_train_data,valid_train_label = next(valid_train)
		valid_data.append((sess.run(out_result,feed_dict={x:valid_train_data})))
		valid_label.append(valid_train_label)
		test_label.append(test_y)
		test_data.append((sess.run(out_result,feed_dict={x:test_x})))

	print('precision @1 = %.5f'%np.mean(ans[-5:]))
	sess.close()


	return np.mean(ans[-5:]), np.concatenate(valid_data,axis=axis),np.concatenate(valid_label,axis=0),\
	np.concatenate(test_data,axis=axis),np.concatenate(test_label,axis=0)


#加载图，run方法是喂入数据得到输出
class ImportGraph():
    #修改模型读取路径
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'stacking_submodel/alex/cifar10/'+loc)
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            # 从指定路径加载模型到局部图中
            saver = tf.train.import_meta_graph(model_path+ '.ckpt.meta',
                                                clear_devices=True)
            saver.restore(self.sess, model_path+'.ckpt')
            # There are TWO options how to get activation operation:
            # 两种方式来调用运算或者参数
                # FROM SAVED COLLECTION:            
            self.activation = tf.get_collection('model_out')[0]
                # BY NAME:
            # self.activation = self.graph.get_operation_by_name('inputs').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"inputs/input_x:0": data})



#加载图恢复数据,一次完成一个子模块的k个小模块的加载,然后算出测试集均值，和本子模块对应的次级学习器的数据集
def restore(submodel_num,k_fold,is_complex):#module_num表示子模型编号
	#model_path_list表示每个子模块的小模块列表
	#修改文件读取名字，读取函数
	### Using the class ###
	if is_complex:
		local_path = 'complex'+str(submodel_num)
		axis = 1
	else:
		local_path = 'real'+str(submodel_num)
		axis=0

	test_file_name = 'CIFAR10.h5'
	_,test_batch = read_cifar10('data/'+test_file_name,1,1000)
	data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'stacking_data/alex/cifar10/'+local_path+'.h5')

	train_result = []
	test_result = []
	trains_lable = []
	tests_label = []

	accuracy=0.0
	for k in range(k_fold):
		train_file_name = 'CIFAR10'+'_split_'+str(k_fold)+str(k+1)+'.h5'
		train_batch,_= read_cifar10("split_file/"+train_file_name,1250,1,train_shuffle=False)
		model = ImportGraph("complex"+str(submodel_num)+'-'+str(k+1))
		test_local_result = []
		for i in range(10):
			test_data,test_lable  = next(test_batch)
			train_data,train_lable = next(train_batch)
			trains_lable.append(train_lable)
			if k ==1:
				tests_label.append(test_lable)
            
			train_result.append(model.run(train_data))
			test_local_result.append(model.run(test_data))
		test_local_result = np.concatenate(test_local_result,axis=1)
		test_result.append(np.array(test_local_result))
	test_results = np.mean(test_result,axis = 0)
	train_result = np.array(train_result)


	train_result = np.concatenate(train_result,axis=axis)
	trains_lable = np.concatenate(trains_lable,axis=0)
	tests_label = np.concatenate(tests_label,axis=0)

	# if is_complex:
	# 	models_result = tf.sqrt(tf.square(train_result[0])+tf.square(train_result[1]))
	# else:
	# 	models_result = test_results

	if is_complex:
		models_result = tf.sqrt(tf.square(test_result[3][0])+tf.square(test_result[3][1]))
	else:
		models_result = test_results

	top_k_op = tf.nn.in_top_k(models_result,tests_label,1)
	accuracy = tf.reduce_mean(tf.cast(top_k_op,tf.float32))

	sess = tf.InteractiveSession()
	acc = sess.run(accuracy)
	print("accuracy is %.4f"%(acc))

	sess.close()

	wrrite_file([train_result,trains_lable],[test_results,tests_label],data_path)






#生成次级模型的输入数据
def generate_secondary_data(is_complex,k_fold):
    for submodel_num in range(1,5):
        graph = tf.Graph()
        with graph.as_default():
            restore(submodel_num,k_fold,is_complex)







#cifar100
# kernel_list = [[5,5,3,3],[5,5,5,3],[5,5,3],[5,3,3],[5,5,3,3],[5,5,5,3],[5,5,3],[5,3,3]]
# channel_list = [[128,128,64,64],[128,64,64,64],[128,128,64],[128,64,64],[128,128,64,64],[128,64,64,64],[128,128,64],[128,64,64]]
# fc_list =[[192],[192],[192,81],[192,81],[192],[192],[192,81],[192,81]]

#cifar10
kernel_list = [[5,3,3],[5,5,2],[5,5],[5,3]]
channel_list = [[128,64,64],[128,64,64],[128,128],[128,64]]
fc_list =[[100],[128],[100,50],[100,50]]

#mnist
# kernel_list = [[5,3],[5,3,3],[5],[3]]
# channel_list = [[16,8],[8,16,16],[32],[32]]
# fc_list =[[30],[50],[30],[60,30]]

#cifar100 and 10
# kernel_list = [[3,3,3,3],[5,5,5,5],[5,5,3,3],[5,3,3]]
# channel_list = [[64,128,256,512],[64,128,128,256],[64,64,128,256],[128,256,512]]

path_list = ['1','2','3','4']

shape_list = [path_list,kernel_list,channel_list,fc_list]

if __name__=='__main__':
	is_complex = True
    #生成子模型
	# model_index = int(sys.argv[1]) #第几个模型
	# model_tag=int(sys.argv[2]) #第几个子模型
	# print("index,tag = ",model_index,model_tag)
    # generate_Primary_net_cifar([k[model_index] for k in  shape_list],model_tag,is_complex) #修改

    #生成次级模型数据,要自己进去修改路径和数据名称等等
	k_fold = 4  #要修改为和之前的训练子模型时的K值相同
	generate_secondary_data(is_complex,k_fold)

