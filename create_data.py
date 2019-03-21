import h5py
import numpy as np
import tensorflow as tf
import Preproc



def loadHDF5Adv(file_name):
    with h5py.File(file_name, 'r') as f:
        dataTrain         = np.array(f['Train']['images'])
        labelsFineTrain   = np.array(f['Train']['labels'])
        dataTest          = np.array(f['Test']['images'])
        labelsFineTest    = np.array(f['Test']['labels'])
    return (dataTrain, labelsFineTrain,dataTest, labelsFineTest)


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
	return result


def Bootstrap_file(file_name):
	dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5(file_name)
	for i in range(4):
		train_image,train_lable = Bootstrap(dataTrain,labelsTrain)
		wrrite_file([train_image,train_lable],[dataTest,labelsTest],file_name[:-3]+'_model'+str(i+1)+'.h5')


# def k_flod(file_name):
# 	dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5Adv(file_name)
# 	for i in range(5):
# 		k = i+2
# 		data_list=divide_equally(k,dataTrain,labelsTrain)
# 		for p,item in enumerate(data_list):
# 			wrrite_file(item,[dataTest, labelsTest],file_name[:-3]+'_st'+str(k)+str(p+1)+'.h5')


def k_flod(file_name):
	dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5Adv(file_name)
	data_list=divide_equally(4,dataTrain,labelsTrain)
	for p,item in enumerate(data_list):
		wrrite_file(item,[dataTest, labelsTest],file_name[:-3]+'_st'+str(4)+str(p+1)+'.h5')



def merge_data(file_head,model_tag):
	data = []
	label = []
	k = int(file_head[-1])
	merge_list = list(range(1,k+1))
	merge_list.remove(model_tag)
	for i in merge_list:
		file_name = file_head+str(i)+'.h5'
		dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5Adv(file_name)
		data.append(dataTrain)
		label.append(labelsTrain)
	data = np.concatenate(data, axis=0)
	label = np.concatenate(label, axis=0)
	return data, label, dataTest, labelsTest

def wrrite_file(train_data,test_data,file_name):
	with h5py.File(file_name,'w') as f:
		f.create_group('Train')
		f.create_group('Test')
		f.create_dataset('Train/images',data = train_data[0])
		f.create_dataset('Train/labels',data = train_data[1])
		f.create_dataset('Test/images',data = test_data[0])
		f.create_dataset('Test/labels',data = test_data[1])
def merge_all(file_heads):
	file_head = file_heads
	file_head+=str(4)
	for tag in range(1,5):
		dataTrain, labelsTrain, dataTest, labelsTest = merge_data(file_head,tag)
		wrrite_file([dataTrain,labelsTrain],[dataTest,labelsTest],file_head+'-'+str(tag)+'.h5')


if __name__=="__main__":
	# merge_all('k_data/MNIST_st')
	loadHDF5Adv('k_data/CIFAR10_st4-4.h5')

	

