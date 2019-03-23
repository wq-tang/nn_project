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

def wrrite_file(train_data,test_data,file_name):
	with h5py.File(file_name,'w') as f:
		f.create_group('Train')
		f.create_group('Test')
		f.create_dataset('Train/images',data = train_data[0])
		f.create_dataset('Train/labels',data = train_data[1])
		f.create_dataset('Test/images',data = test_data[0])
		f.create_dataset('Test/labels',data = test_data[1])

class ImportGraph():
	#修改模型读取路径
	"""  Importing and running isolated TF graph """
	def __init__(self, model_path):
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




def restore(is_complex,big_tag,tag):
    #修改文件读取名字，读取函数
    ### Using the class ###
    if is_complex:
        local_path = 'complex'+str(big_tag)
    else:
        local_path = 'real'+str(big_tag)
########
    model_path =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\
        'mynet/stacking/cifar10/'+local_path+str(tag))
############
    validation_name =  'CIFAR10_st4'+str(tag)+'.h5'
    valid_train,test_batch = read_cifar10('k_data/'+validation_name,1250,1000)#10次取完  #############
        
    if is_complex:
        axis = 1
    else:
        axis=0

    valid_data=[]
    valid_label = []
    test_data=[]
    test_label = []
    for i in range(10):
        model = ImportGraph(model_path)
        test_x,test_y = next(test_batch)
        valid_train_data,valid_train_label = next(valid_train)
        valid_data.append(model.run(valid_train_data))
        valid_label.append(valid_train_label)
        test_label.append(test_y)
        test_data.append(model.run(test_x))
    return  np.concatenate(valid_data,axis=axis),np.concatenate(valid_label,axis=0),\
    np.concatenate(test_data,axis=axis),np.concatenate(test_label,axis=0)

def generate_secondary_data(file_heads,is_complex,big_tag):
    acc = {}
    if is_complex:
        axis =1
    else:
        axis=0

    train_data_set = []
    train_label_set=[]
    test_data_set =[]

    if is_complex:
        file_head = file_heads+'_complex'+str(big_tag) ####修改
    else:
        file_head =file_heads+ '_real'+str(big_tag)   #修改

    for tag in range(1,5):
        train_data,train_label,test_data,test_label = restore(is_complex,big_tag,tag)
        train_data_set.append(train_data)
        train_label_set.append(train_label)
        test_data_set.append(test_data)
    train_data_set = np.concatenate(train_data_set,axis = axis)
    train_label_set = np.concatenate(train_label_set,axis = 0)

    test_data_set = np.mean(test_data_set,axis=0)
        #写操作
    wrrite_file([train_data_set,train_label_set],[test_data_set,test_label],'bu/'+file_head+'.h5')



if __name__=='__main__':
    file_head = 'CIFAR10'##############
    is_complex= False
    model_tag = 4
    generate_secondary_data(file_head,is_complex,model_tag)
