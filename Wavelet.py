import tensorflow as tf
import numpy as np
from model import alexNet
from functools import reduce
def reshape(tensor):
    shapes = tensor.get_shape().as_list()[1:]
    mul = reduce(lambda x,y:x * y,shapes)
    Res = tf.reshape(tensor,[-1,mul])
    return Res
def sign(x):
    e = 0.1**8
    return tf.nn.relu(x)/(tf.nn.relu(x)+e)

class wavelet(alexNet):
	"""docstring for wavelet"""
	def __init__(self, x, classNum, seed,modelPath = "wavelet"):
		super(wavelet,self).__init__(x, classNum, seed,modelPath)
		tf.set_random_seed(seed) 
		self.build_complex_wavelet_bagging(2)

	def build_complex_wavelet_bagging(self,conv_num):
		conv=0
		for i in range(conv_num):
			conv+=self.complex_wavelet_conv(self.X_com,'conv_model'+str(i))
		fc3 = self.complex_wavelet_fc(conv,'wavelet_fc')
		self.out = tf.sqrt(tf.square(fc3[0])+tf.square(fc3[1]))
	

	def complex_wavelet_conv(self,name):	
		with tf.variable_scope(name):
			layer11 = self.complex_convLayer(self.X_com,[2,2],[1,1],32,"layer11","SAME")
			pool11 = self.complex_maxPoolLayer(layer11,[2, 2],[ 2,2], "pool1", "SAME")

			layer12 = self.complex_convLayer(self.X_com,[3,3],[1,1],32,"layer12","SAME")
			pool12 = self.complex_maxPoolLayer(layer12,[2, 2],[ 2,2], "pool12", "SAME")

			layer13 = self.complex_convLayer(self.X_com,[4,4],[1,1],32,"layer13","SAME")
			pool13 = self.complex_maxPoolLayer(layer13,[2, 2],[ 2,2], "pool13", "SAME")

			layer14 = self.complex_convLayer(self.X_com,[6,6],[1,1],32,"layer14","SAME")
			pool14 = self.complex_maxPoolLayer(layer14,[2, 2],[ 2,2], "pool14", "SAME")


			layer21 = self.complex_convLayer(pool14,[2,2],[1,1],16,"layer21","SAME")
			pool21 = self.complex_maxPoolLayer(layer21,[2, 2],[ 2,2], "pool21", "SAME")

			layer22 = self.complex_convLayer(pool14,[3,3],[1,1],16,"layer22","SAME")
			pool22 = self.complex_maxPoolLayer(layer22,[2, 2],[ 2,2], "pool22", "SAME")

			layer23 = self.complex_convLayer(pool14,[4,4],[1,1],16,"layer23","SAME")
			pool23 = self.complex_maxPoolLayer(layer23,[2, 2],[ 2,2], "pool23", "SAME")

			layer24 = self.complex_convLayer(pool14,[6,6],[1,1],16,"layer24","SAME")
			pool24 = self.complex_maxPoolLayer(layer24,[2, 2],[ 2,2], "pool24", "SAME")


			layer31 = self.complex_convLayer(pool24,[2,2],[1,1],16,"layer31","SAME")
			pool31 = self.complex_maxPoolLayer(layer31,[2, 2],[ 2,2], "pool31", "SAME")

			layer32 = self.complex_convLayer(pool24,[3,3],[1,1],16,"layer32","SAME")
			pool32 = self.complex_maxPoolLayer(layer32,[2, 2],[ 2,2], "pool32", "SAME")

			layer33 = self.complex_convLayer(pool24,[4,4],[1,1],16,"layer33","SAME")
			pool33 = self.complex_maxPoolLayer(layer33,[2, 2],[ 2,2], "pool33", "SAME")

			layer34 = self.complex_convLayer(pool24,[6,6],[1,1],16,"layer34","SAME")
			pool34 = self.complex_maxPoolLayer(layer34,[2, 2],[ 2,2], "pool34", "SAME")

		return np.array([pool11,pool12,pool13,pool21,pool22,pool23,pool31,pool32,pool33,pool34])



	def complex_wavelet_fc(self,C_list,name):
		with tf.variable_scope(name):
			convoutR = tf.concat([reshape(C_list[0][0]),reshape(C_list[1][0]),reshape(C_list[2][0]),reshape(C_list[3][0]),\
				reshape(C_list[4][0]),reshape(C_list[5][0]),reshape(C_list[6][0]),reshape(C_list[7][0]),reshape(C_list[8][0]),\
				reshape(C_list[9][0])],axis = 1)
			convoutI = tf.concat([reshape(C_list[0][1]),reshape(C_list[1][1]),reshape(C_list[2][1]),reshape(C_list[3][1]),\
				reshape(C_list[4][1]),reshape(C_list[5][1]),reshape(C_list[6][1]),reshape(C_list[7][1]),reshape(C_list[8][1]),\
				reshape(C_list[9][1])],axis = 1)
			dim = convoutI.get_shape().as_list()[-1]
			fc1 = self.complex_fcLayer([convoutR,convoutI], dim, 60,  name = "fc1")
			fc2 = self.complex_fcLayer(fc1, 60, 30, name =  "fc2")
			fc3 = self.complex_fcLayer(fc2, 30, self.CLASSNUM, name =  "fc3",norm=False)
			return fc3
		

	def build_real_wavelet(self):

		self.complex_convLayer = self.self.complex_convLayer
		self.complex_maxPoolLayer = self.self.complex_maxPoolLayer
		self.complex_fcLayer = self.self.complex_fcLayer
		self.X_com = self.X
		relu_fun =tf.nn.relu

		with tf.variable_scope("wavelet_net"):
			layer11 = self.complex_convLayer(self.X_com,[2,2],[1,1],int(32*1.41)+1,"layer11","SAME")
			pool11 = self.complex_maxPoolLayer(layer11,[2, 2],[ 2,2], "pool1", "SAME")

			layer12 = self.complex_convLayer(self.X_com,[3,3],[1,1],int(32*1.41)+1,"layer12","SAME")
			pool12 = self.complex_maxPoolLayer(layer12,[2, 2],[ 2,2], "pool12", "SAME")

			layer13 = self.complex_convLayer(self.X_com,[4,4],[1,1],int(32*1.41)+1,"layer13","SAME")
			pool13 = self.complex_maxPoolLayer(layer13,[2, 2],[ 2,2], "pool13", "SAME")

			layer14 = self.complex_convLayer(self.X_com,[6,6],[1,1],int(32*1.41)+1,"layer14","SAME")
			pool14 = self.complex_maxPoolLayer(layer14,[2, 2],[ 2,2], "pool14", "SAME")


			layer21 = self.complex_convLayer(pool14,[2,2],[1,1],int(16*1.41)+1,"layer21","SAME")
			pool21 = self.complex_maxPoolLayer(layer21,[2, 2],[ 2,2], "pool21", "SAME")

			layer22 = self.complex_convLayer(pool14,[3,3],[1,1],int(16*1.41)+1,"layer22","SAME")
			pool22 = self.complex_maxPoolLayer(layer22,[2, 2],[ 2,2], "pool22", "SAME")

			layer23 = self.complex_convLayer(pool14,[4,4],[1,1],int(16*1.41)+1,"layer23","SAME")
			pool23 = self.complex_maxPoolLayer(layer23,[2, 2],[ 2,2], "pool23", "SAME")

			layer24 = self.complex_convLayer(pool14,[6,6],[1,1],int(16*1.41)+1,"layer24","SAME")
			pool24 = self.complex_maxPoolLayer(layer24,[2, 2],[ 2,2], "pool24", "SAME")


			layer31 = self.complex_convLayer(pool24,[2,2],[1,1],int(16*1.41)+1,"layer31","SAME")
			pool31 = self.complex_maxPoolLayer(layer31,[2, 2],[ 2,2], "pool31", "SAME")

			layer32 = self.complex_convLayer(pool24,[3,3],[1,1],int(16*1.41)+1,"layer32","SAME")
			pool32 = self.complex_maxPoolLayer(layer32,[2, 2],[ 2,2], "pool32", "SAME")

			layer33 = self.complex_convLayer(pool24,[4,4],[1,1],int(16*1.41)+1,"layer33","SAME")
			pool33 = self.complex_maxPoolLayer(layer33,[2, 2],[ 2,2], "pool33", "SAME")

			layer34 = self.complex_convLayer(pool24,[6,6],[1,1],int(16*1.41)+1,"layer34","SAME")
			pool34 = self.complex_maxPoolLayer(layer34,[2, 2],[ 2,2], "pool34", "SAME")


			convout = tf.concat([reshape(pool11),reshape(pool12),reshape(pool13),reshape(pool21),reshape(pool22),\
			reshape(pool23),reshape(pool31),reshape(pool32),reshape(pool33),reshape(pool34)],axis = 1)
			mul = convout.get_shape().as_list()[-1]
			fc1 = self.complex_fcLayer(convout, mul, int(60*1.41)+1,  name = "fc1")
			fc2 = self.complex_fcLayer(fc1, int(60*1.41)+1, int(30*1.41)+1,name =  "fc2")
			fc3 = self.complex_fcLayer(fc2, int(30*1.41)+1, self.CLASSNUM, name =  "fc3",norm=False)
			self.out = fc3

	def build_net_share(self,complex = True):
		if not complex:
			self.complex_convLayer = self.self.complex_convLayer
			self.complex_maxPoolLayer = self.self.complex_maxPoolLayer
			self.complex_fcLayer = self.self.complex_fcLayer
			self.X_com = self.X
			relu_fun =tf.nn.relu
		else:
			self.complex_convLayer = self.complex_convLayer
			self.complex_maxPoolLayer = self.complex_maxPoolLayer
			self.complex_fcLayer = self.complex_fcLayer
			self.X_com = self.X_com
			relu_fun =self.Learnable_angle_relu		
		with tf.variable_scope("wavelet_net"):
			with tf.variable_scope("block"):
				layer11 = self.complex_convLayer(self.X_com,[2,2],[1,1],16,"layer1","SAME")
				pool11 = self.complex_maxPoolLayer(layer11,[2, 2],[ 2,2], "pool", "SAME")

				layer12 = self.complex_convLayer(self.X_com,[3,3],[1,1],32,"layer2","SAME")
				pool12 = self.complex_maxPoolLayer(layer12,[2, 2],[ 2,2], "pool2", "SAME")

				layer13 = self.complex_convLayer(self.X_com,[4,4],[1,1],64,"layer3","SAME")
				pool13 = self.complex_maxPoolLayer(layer13,[2, 2],[ 2,2], "pool3", "SAME")

				layer14 = self.complex_convLayer(self.X_com,[6,6],[1,1],3,"layer4","SAME")
				pool14 = self.complex_maxPoolLayer(layer14,[2, 2],[ 2,2], "pool4", "SAME")


			with tf.variable_scope("block",reuse=True):
				layer21 = self.complex_convLayer(pool14,[2,2],[1,1],16,"layer1","SAME")
				pool21 = self.complex_maxPoolLayer(layer21,[2, 2],[ 2,2], "pool1", "SAME")

				layer22 = self.complex_convLayer(pool14,[3,3],[1,1],32,"layer2","SAME")
				pool22 = self.complex_maxPoolLayer(layer22,[2, 2],[ 2,2], "pool2", "SAME")

				layer23 = self.complex_convLayer(pool14,[4,4],[1,1],64,"layer3","SAME")
				pool23 = self.complex_maxPoolLayer(layer23,[2, 2],[ 2,2], "pool3", "SAME")

				layer24 = self.complex_convLayer(pool14,[6,6],[1,1],3,"layer4","SAME")
				pool24 = self.complex_maxPoolLayer(layer24,[2, 2],[ 2,2], "pool4", "SAME")


			with tf.variable_scope("block",reuse=True):
				layer31 = self.complex_convLayer(pool24,[2,2],[1,1],16,"layer1","SAME")
				pool31 = self.complex_maxPoolLayer(layer31,[2, 2],[ 2,2], "pool1", "SAME")

				layer32 = self.complex_convLayer(pool24,[3,3],[1,1],32,"layer2","SAME")
				pool32 = self.complex_maxPoolLayer(layer32,[2, 2],[ 2,2], "pool2", "SAME")

				layer33 = self.complex_convLayer(pool24,[4,4],[1,1],64,"layer3","SAME")
				pool33 = self.complex_maxPoolLayer(layer33,[2, 2],[ 2,2], "pool3", "SAME")

				layer34 = self.complex_convLayer(pool24,[6,6],[1,1],3,"layer4","SAME")
				pool34 = self.complex_maxPoolLayer(layer34,[2, 2],[ 2,2], "pool4", "SAME")



			if not complex:
				convout = tf.concat([reshape(pool11),reshape(pool12),reshape(pool13),reshape(pool21),reshape(pool22),\
				reshape(pool23),reshape(pool31),reshape(pool32),reshape(pool33),reshape(pool34)],axis = 1)
				mul = convout.get_shape().as_list()[-1]
				fc1 = self.complex_fcLayer(convout, mul, int(384*1.41)+1,  name = "fc1")
				fc2 = self.complex_fcLayer(fc1, int(384*1.41)+1, int(192*1.41)+1,name =  "fc2")
				fc3 = self.complex_fcLayer(fc2, int(192*1.41)+1, self.CLASSNUM, name =  "fc3",norm=False)
				self.out = fc3
			else:
				convoutR = tf.concat([reshape(pool11[0]),reshape(pool12[0]),reshape(pool13[0]),reshape(pool21[0]),\
					reshape(pool22[0]),reshape(pool23[0]),reshape(pool31[0]),reshape(pool32[0]),reshape(pool33[0]),\
					reshape(pool34[0])],axis = 1)
				convoutI = tf.concat([reshape(pool11[1]),reshape(pool12[1]),reshape(pool13[1]),reshape(pool21[1]),\
					reshape(pool22[1]),reshape(pool23[1]),reshape(pool31[1]),reshape(pool32[1]),reshape(pool33[1]),\
					reshape(pool34[1])],axis = 1)
				dim = convoutI.get_shape().as_list()[-1]
				fc1 = self.complex_fcLayer([convoutR,convoutI], dim, 384,  name = "fc1")
				fc2 = self.complex_fcLayer(fc1, 384, 192, name =  "fc2")
				fc3 = self.complex_fcLayer(fc2, 192, self.CLASSNUM, name =  "fc3",norm=False)
				self.out = tf.sqrt(tf.square(fc3[0])+tf.square(fc3[1]))