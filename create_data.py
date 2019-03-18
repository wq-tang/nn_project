import h5py
import numpy as np
import tensorflow as tf
import Preproc



def loadHDF5(file_name):
    with h5py.File(file_name, 'r') as f:
        dataTrain   = np.expand_dims(np.array(f['Train']['images'])[:, :, :, 0], axis=-1)
        labelsTrain = np.array(f['Train']['labels']).reshape([-1])
        dataTest    = np.expand_dims(np.array(f['Test']['images'])[:, :, :, 0], axis=-1)
        labelsTest  = np.array(f['Test']['labels']).reshape([-1])
        
    return (dataTrain, labelsTrain, dataTest, labelsTest)

def Bootstrap(data,label):
	index = np.random.randint(0,data.shape[0],data.shape[0])
	images = data[index]
	labels = label[index]
	return images,labels

def divide_equally(k,data,label):
	result = []
	step = data.shape[0]//k
	index = list(range(data.shape[0]))
	np.random.shuffle(index)
	for i in range(k):
		if i != k-1:
			images = data[index][i*step:(i+1)*step]
			labels = label[index][i*step:(i+1)*step]
		else:
			images = data[index][i*step:]
			labels = label[index][i*step:]
		result.append((images,labels))


def wrrite_file(train_data,test_data,filename):
	with h5py.File(filename,'w') as f:
		f.create_group('Train')
		f.create_group('Test')
		f.create_dataset('Train/images',data = train_data[0])
		f.create_dataset('Train/labels',data = train_data[1])
		f.create_dataset('Test/images',data = test_data[0])
		f.create_dataset('Test/labels',data = test_data[1])

def Bootstrap_file(filename):
	dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5(filename)
	for i in range(4):
		train_image,train_lable = Bootstrap(dataTrain,labelsTrain)
		wrrite_file([train_image,train_lable],[dataTest,labelsTest],filename[:-3]+'_model'+str(i+1)+'.h5')

if __name__=="__main__":
	Bootstrap_file('MNIST.h5')

