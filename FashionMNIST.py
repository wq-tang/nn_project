import random
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

def preproc(images, size): 
    results = np.ndarray([images.shape[0]]+size, np.uint8)
    for idx in range(images.shape[0]): 
        distorted     = Preproc.centerCrop(images[idx], size)
        results[idx]  = distorted
    
    return results

def generator(BatchSize, preprocSize=[28, 28, 1]): 
    dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5()
    data = np.concatenate([dataTrain, dataTest], axis=0)
    labels = np.concatenate([labelsTrain, labelsTest], axis=0)
    
    invertedIdx = [[] for _ in range(10)]
    
    for idx in range(len(data)):
        invertedIdx[labels[idx]].append(idx)
    
    def genCIFAR10():
        now = 0
        batchData   = []
        batchLabels = []
        for _ in range(BatchSize):
            classAnchor = labels[now]
            classPos    = classAnchor
            idxAnchor   = now
            idxPos      = random.randint(0, len(invertedIdx[classPos])-1)
            while idxPos == now:
                idxPos  = random.randint(0, len(invertedIdx[classPos])-1)
            idxPos      = invertedIdx[idxPos]
            classNeg    = random.randint(0, 9)
            while classNeg == classPos:
                classNeg = random.randint(0, 9)
            idxNeg      = random.randint(0, len(invertedIdx[classNeg])-1)
            idxNeg      = invertedIdx[idxNeg]
            batchData.extend([data[idxAnchor], data[idxPos], data[idxNeg]])
            batchLabels.extend([classAnchor, classPos, classNeg])
            now += 1
            if now >= 70000: 
                now = 0
        batchData = preproc(np.array(batchData), preprocSize)
        batchLabels = np.array(batchLabels)
        assert batchData.shape[0] == BatchSize*3, "CIFAR10: size is wrong"
        assert len(batchData.shape) == 4, "CIFAR10: size is wrong"
        yield batchData, batchLabels
    
    return genCIFAR10()


def allData(preprocSize=[28, 28, 1]): 
    dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5()
    data = np.concatenate([dataTrain, dataTest], axis=0)
    labels = np.concatenate([labelsTrain, labelsTest], axis=0)
    
    invertedIdx = [[] for _ in range(10)]
    
    for idx in range(len(data)):
        invertedIdx[labels[idx]].append(idx)
    
    return preproc(data, preprocSize), labels, invertedIdx


def generators(TrainBatchSize,TestBatchSize, preprocSize=[28, 28, 1], numSame=1, numDiff=1):
    ''' generators for multi-let
    Args:
        numSame: number of samples in the same coarse class; 
        numDiff: number of sample in different coarse class. 
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5(file_name)
        
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
            # distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted.reshape([28, 28, 1])
        
        return results
    
    def preprocTest(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted = images[idx]
            distorted     = Preproc.centerCrop(distorted, size)
            results[idx]  = distorted
        
        return results
    
    def genTrainBatch(TrainBatchSize):
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
            
    def genTestBatch(TestBatchSize):
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
        
    return genTrainBatch(TrainBatchSize), genTestBatch(TestBatchSize)

def generatorsAdv(BatchSize, preprocSize=[28, 28, 1]):
    ''' generators for multi-let
    Args:
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5(file_name)
        
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
            #distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted.reshape([28, 28, 1])
        
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
            batchTargets = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
                batchTargets.append(random.randint(0, 9))
            batchImages = preprocTrain(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            batchTargets = np.array(batchTargets)
            
            yield batchImages, batchLabels, batchTargets
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages = []
            batchLabels = []
            batchTargets = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
                batchTargets.append(random.randint(0, 9))
            batchImages = preprocTest(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            batchTargets = np.array(batchTargets)
            
            yield batchImages, batchLabels, batchTargets
        
    return genTrainBatch(BatchSize), genTestBatch(BatchSize)


            
def read_fashion(file_name,TrainBatchSize,TestBatchSize):
    batchTrain, batchTest = generators(file_name,TrainBatchSize,TestBatchSize, preprocSize=[28, 28, 1], numSame=0, numDiff=0)
    return batchTrain, batchTest


