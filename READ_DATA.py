import random
import h5py
import numpy as np
import tensorflow as tf
import Preproc

def loadHDF5(file_name):
    with h5py.File(file_name, 'r') as f:
        dataTrain   = np.array(f['Train']['images'])
        labelsTrain = np.array(f['Train']['labels'])
        dataTest    = np.array(f['Test']['images'])
        labelsTest  = np.array(f['Test']['labels'])
        
    return (dataTrain, labelsTrain, dataTest, labelsTest)

def generators(complex_and_second,file_name,TrainBatchSize,TestBatchSize, preprocSize=[28, 28, 3]):
    ''' generators for multi-let
    Args:
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    if complex_and_second:
        axis = 1
    else:
        axis = 0
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5(file_name)
        
    def genTrainDatum():
        index = Preproc.genIndex(dataTrain.shape[axis], shuffle=True)
        while True:
            indexAnchor = next(index)
            if complex_and_second:
                imageAnchor = dataTrain[:,indexAnchor,:]
            else:
                imageAnchor = dataTrain[indexAnchor]
            labelAnchor = labelsTrain[indexAnchor]                
            images      = imageAnchor
            labels      = labelAnchor
            
            yield images, labels
        
    def genTestDatum():
        index = Preproc.genIndex(dataTest.shape[axis], shuffle=False)
        while True:
            indexAnchor = next(index)
            if complex_and_second:
                imageAnchor = dataTest[:,indexAnchor,:]
            else:
                imageAnchor = dataTest[indexAnchor]
            labelAnchor = labelsTest[indexAnchor]
            images      = imageAnchor
            labels      = labelAnchor
            
            yield images, labels
    
    
    def genTrainBatch(BatchSize):
        datum = genTrainDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            # batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield np.array(batchImages), np.array(batchLabels) 
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            # batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield np.array(batchImages), np.array(batchLabels)       
    return genTrainBatch(TrainBatchSize), genTestBatch(TestBatchSize)


            
def read_data(complex_and_second,file_name,TrainBatchSize,TestBatchSize):
    batchTrain, batchTest = generators(complex_and_second,file_name,TrainBatchSize,TestBatchSize, preprocSize=[24, 24, 3])
    return batchTrain, batchTest