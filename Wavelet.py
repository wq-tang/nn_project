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
		self.build_net(False)
    
	def build_net(self,complex = False):
		if not complex:
			convLayer = self.convLayer
			maxPoolLayer = self.maxPoolLayer
			fcLayer = self.fcLayer
			inputs = self.X
		else:
			convLayer = self.complex_convLayer
			maxPoolLayer = self.complex_maxPoolLayer
			fcLayer = self.complex_fcLayer
			inputs = self.X_com			
		with tf.variable_scope("wavelet_net"):
			layer11 = convLayer(inputs,[2,2],[1,1],16,"layer11","SAME")
			pool11 = maxPoolLayer(layer11,[2, 2],[ 2,2], "pool11", "SAME")

			layer12 = convLayer(inputs,[3,3],[1,1],16,"layer12","SAME")
			pool12 = maxPoolLayer(layer12,[2, 2],[ 2,2], "pool12", "SAME")

			layer13 = convLayer(inputs,[4,4],[1,1],16,"layer13","SAME")
			pool13 = maxPoolLayer(layer13,[2, 2],[ 2,2], "pool13", "SAME")

			layer14 = convLayer(inputs,[6,6],[1,1],16,"layer14","SAME")
			pool14 = maxPoolLayer(layer14,[2, 2],[ 2,2], "pool14", "SAME")



			layer21 = convLayer(pool14,[2,2],[1,1],16,"layer21","SAME")
			pool21 = maxPoolLayer(layer21,[2, 2],[ 2,2], "pool21", "SAME")

			layer22 = convLayer(pool14,[3,3],[1,1],16,"layer22","SAME")
			pool22 = maxPoolLayer(layer22,[2, 2],[ 2,2], "pool22", "SAME")

			layer23 = convLayer(pool14,[4,4],[1,1],16,"layer23","SAME")
			pool23 = maxPoolLayer(layer23,[2, 2],[ 2,2], "pool23", "SAME")

			layer24 = convLayer(pool14,[6,6],[1,1],16,"layer24","SAME")
			pool24 = maxPoolLayer(layer24,[2, 2],[ 2,2], "pool24", "SAME")



			layer31 = convLayer(pool24,[2,2],[1,1],32,"layer31","SAME")
			pool31 = maxPoolLayer(layer31,[2, 2],[ 2,2], "pool31", "SAME")

			layer32 = convLayer(pool24,[3,3],[1,1],32,"layer32","SAME")
			pool32 = maxPoolLayer(layer32,[2, 2],[ 2,2], "pool32", "SAME")

			layer33 = convLayer(pool24,[4,4],[1,1],32,"layer33","SAME")
			pool33 = maxPoolLayer(layer33,[2, 2],[ 2,2], "pool33", "SAME")

			layer34 = convLayer(pool24,[6,6],[1,1],32,"layer34","SAME")
			pool34 = maxPoolLayer(layer34,[2, 2],[ 2,2], "pool34", "SAME")



			if not complex:
				convout = tf.concat([reshape(pool11),reshape(pool12),reshape(pool13),reshape(pool21),reshape(pool22),\
				reshape(pool23),reshape(pool31),reshape(pool32),reshape(pool33),reshape(pool34)],axis = 1)
				mul = convout.get_shape().as_list()[-1]
				fc1 = fcLayer(convout, mul, int(384*1.41)+1,  name = "fc1")
				fc2 = fcLayer(fc1, int(384*1.41)+1, int(192*1.41)+1,name =  "fc2")
				fc3 = fcLayer(fc2, int(192*1.41)+1, self.CLASSNUM, name =  "fc3",norm=False)
			else:
				convoutR = tf.concat([reshape(pool11[0]),reshape(pool12[0]),reshape(pool13[0]),reshape(pool21[0]),\
					reshape(pool22[0]),reshape(pool23[0]),reshape(pool31[0]),reshape(pool32[0]),reshape(pool33[0]),\
					reshape(pool34[0])],axis = 1)
				convoutI = tf.concat([reshape(pool11[1]),reshape(pool12[1]),reshape(pool13[1]),reshape(pool21[1]),\
					reshape(pool22[1]),reshape(pool23[1]),reshape(pool31[1]),reshape(pool32[1]),reshape(pool33[1]),\
					reshape(pool34[1])],axis = 1)
				mul = convoutI.get_shape().as_list()[-1]
				fc1 = fcLayer([convoutR,convoutI], dim, 384,  name = "fc1")
				fc2 = fcLayer(fc1, 384, 192, name =  "fc2")
				fc3 = fcLayer(fc2, 192, self.CLASSNUM, name =  "fc3",norm=False)
				self.out = tf.sqrt(tf.square(fc3[0])+tf.square(fc3[1]))
			self.out = fc3




