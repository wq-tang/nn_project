import pickle
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import Preproc

TrainSize = 50000
TestSize  = 10000
NumFineInOneCoarse = 5

pathTrain = './cifar-100-python/train' # 50000 training data
pathTest = './cifar-100-python/testCIFAR10'   # 10000 testCIFAR10 data

labelsCoarse = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', \
                 'household electrical devices', 'household furniture', 'insects', 'large carnivores', \
                 'large man-made outdoor things', 'large natural outdoor scenes', \
                 'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', \
                 'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

labelsFine = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', \
              'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', \
              'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', \
              'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', \
              'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', \
              'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', \
              'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', \
              'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', \
              'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', \
              'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', \
              'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', \
              'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', \
              'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def getTrainRaw():
    with open(pathTrain, 'rb') as fo:
        tmp = pickle.load(fo, encoding='bytes')
    
    return tmp[b'data'], tmp[b'coarse_labels'], tmp[b'fine_labels']

def getTestRaw():
    with open(pathTest, 'rb') as fo:
        tmp = pickle.load(fo, encoding='bytes')
    
    return tmp[b'data'], tmp[b'coarse_labels'], tmp[b'fine_labels']

def showData():
    dataTrain, labelsCoarseTrain, labelsFineTrain = getTrainRaw()
    dataTest,  labelsCoarseTest,  labelsFineTest  = getTestRaw()
    dataTrain = np.transpose(np.reshape(dataTrain, [-1, 3, 32, 32]), [0, 2, 3, 1])
    dataTest  = np.transpose(np.reshape(dataTest,  [-1, 3, 32, 32]), [0, 2, 3, 1])
    for idx in range(24):
        plt.subplot(6, 6, idx+1)
        tmp = random.randint(0, 50000-1)
        plt.imshow(dataTrain[tmp])
        print('Index: ', idx, ' Type: ', labelsCoarseTrain[tmp], ': ', labelsCoarse[labelsCoarseTrain[tmp]], \
              ', ', labelsFineTrain[tmp], ': ', labelsFine[labelsFineTrain[tmp]])
    for idx in range(24, 36):
        plt.subplot(6, 6, idx+1)
        tmp = random.randint(0, 10000-1)
        plt.imshow(dataTest[tmp])
        print('Index: ', idx, ' Type: ', labelsCoarseTest[tmp], ': ', labelsCoarse[labelsCoarseTest[tmp]],  \
              ', ', labelsFineTest[tmp], ': ', labelsFine[labelsFineTest[tmp]])
    plt.figure()
    count = 0
    for idx in range(50000):
        if labelsFineTrain[idx] == 32 and count < 36:
            count += 1
            plt.subplot(6, 6, count)
            plt.imshow(dataTrain[idx])
    plt.show()

def saveHDF5():
    dataTrain, labelsCoarseTrain, labelsFineTrain = getTrainRaw()
    dataTest,  labelsCoarseTest,  labelsFineTest  = getTestRaw()
    dataTrain = np.transpose(np.reshape(dataTrain, [-1, 3, 32, 32]), [0, 2, 3, 1])
    dataTest  = np.transpose(np.reshape(dataTest,  [-1, 3, 32, 32]), [0, 2, 3, 1])
    with h5py.File('CIFAR100.h5', 'w') as f:
        train = f.create_group('Train')
        train['data']         = dataTrain
        train['labelsCoarse'] = labelsCoarseTrain
        train['labelsFine']   = labelsFineTrain
        print('Training set saved. ')
        test = f.create_group('Test')
        test['data']          = dataTest
        test['labelsCoarse']  = labelsCoarseTest
        test['labelsFine']    = labelsFineTest
        print('Test set saved. ')
        
def loadHDF5():
    with h5py.File('CIFAR100.h5', 'r') as f:
        dataTrain         = np.array(f['Train']['data'])
        labelsCoarseTrain = np.array(f['Train']['labelsCoarse'])
        labelsFineTrain   = np.array(f['Train']['labelsFine'])
        dataTest          = np.array(f['Test']['data'])
        labelsCoarseTest  = np.array(f['Test']['labelsCoarse'])
        labelsFineTest    = np.array(f['Test']['labelsFine'])
        
    return (dataTrain, labelsCoarseTrain, labelsFineTrain, \
            dataTest, labelsCoarseTest, labelsFineTest)
        
def loadHDF5Adv():
    with h5py.File('CIFAR100.h5', 'r') as f:
        dataTrain         = np.array(f['Train']['data'])
        labelsCoarseTrain = np.array(f['Train']['labelsCoarse'])
        labelsFineTrain   = np.array(f['Train']['labelsFine'])
        dataTest          = np.array(f['Test']['data'])
        labelsCoarseTest  = np.array(f['Test']['labelsCoarse'])
        labelsFineTest    = np.array(f['Test']['labelsFine'])
        
    return (dataTrain, labelsCoarseTrain, \
            dataTest, labelsCoarseTest)

def generators(Train_batchSize, Test_batchSize,preprocSize=[32, 32, 3]):
    ''' generators for multi-let
    Args:
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5Adv()
        
    def genTrainDatum():
        index = Preproc.genIndex(dataTrain.shape[0], shuffle=True)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTrain[indexAnchor]
            labelAnchor = labelsTrain[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
        
    def genTestDatum():
        index = Preproc.genIndex(dataTest.shape[0], shuffle=False)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTest[indexAnchor]
            labelAnchor = labelsTest[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
    
    def preprocTrain(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted     = Preproc.randomFlipH(images[idx])
            distorted     = Preproc.randomShift(distorted, rng=4)
            #distorted     = Preproc.randomRotate(distorted, rng=30)
            # distorted     = Preproc.randomRotate(images[idx], rng=30)
            distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted
        
        return results
    
    def preprocTest(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted = images[idx]
            distorted     = Preproc.centerCrop(distorted, size)
            results[idx]  = distorted
        
        return results
    
    def genTrainBatch(BatchSize):
        datum = genTrainDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocTrain(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield batchImages, batchLabels
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocTest(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield batchImages, batchLabels
        
    return genTrainBatch(Train_batchSize), genTestBatch(Test_batchSize)
            
def read_cifar100(Train_batchSize,Test_batchSize):
    batchTrain, batchTest = generators(Train_batchSize=Train_batchSize, Test_batchSize=Test_batchSize,preprocSize=[24, 24, 3])
    return   batchTrain,batchTest      
        
