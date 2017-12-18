# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:05:27 2017

@author: patrickgavigan

Tool for loading training and testing data. Used for different types of 
tests that I may want to do with my tools (differnt data sets).

"""

from FileFunctions import getRawInputData
from FileFunctions import getImages
from FileFunctions import saveInputFile
from DataSplitter import splitTrainingTestingValidationData
from DataSplitter import splitValidationTest


import sys
import numpy as np
import keras

# dataType specifies the type of data being requested, such as Udiacity 
# Challenge2 data, MNIST, Boston housing, etc.

def getTrainingTestingValidationData(dataType, dataDirectoryPath, 
                                     relativeResultPath):

    if (dataType == 'MNIST_Classification'):
        from keras.datasets import mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        
        (xValidate, yValidate, xTest, yTest) = splitValidationTest(xTest, 
                                                                   yTest)
        
        # convert class vectors to binary class matrices
        numClasses = 10
        yTrain = keras.utils.to_categorical(yTrain, numClasses)
        yValidate = keras.utils.to_categorical(yValidate, numClasses)
        yTest = keras.utils.to_categorical(yTest, numClasses)
        yShape = numClasses
        
        xTrain = xTrain.reshape(xTrain.shape[0], 
                                xTrain.shape[1],
                                xTrain.shape[2], 
                                1)
        xValidate = xValidate.reshape(xValidate.shape[0], 
                                      xValidate.shape[1],
                                      xValidate.shape[2], 
                                      1)
        xTest = xTest.reshape(xTest.shape[0], 
                              xTest.shape[1], 
                              xTest.shape[2], 
                              1)
        xShape = (xTest.shape[1], 
                  xTest.shape[2], 
                  xTest.shape[3])
        newRun = False  # No need to force a new training run, yet
        
    elif (dataType == 'MNIST_Regression'):
        from keras.datasets import mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        
        (xValidate, yValidate, xTest, yTest) = splitValidationTest(xTest, 
                                                                   yTest)
        
        # convert class vectors to binary class matrices
        yShape = 1          # ToDo: I want to parameterize this somehow
        xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1],
                                xTrain.shape[2], 1)
        xValidate = xValidate.reshape(xValidate.shape[0], xValidate.shape[1],
                                      xValidate.shape[2], 1)
        xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 
                              xTest.shape[2], 1)
        xShape = (xTest.shape[1], xTest.shape[2], xTest.shape[3])
        newRun = False  # No need to force a new training run, yet

    elif (dataType == 'UdiacityChallenge2_Classification'):
        numClasses = 10
        relativeRawPath = dataDirectoryPath + '/Datasets/UdacityChallenge2/'
        rawDataFile = relativeRawPath + 'final_example.csv' # The file path for the CSV file
        imagePath = relativeRawPath + 'center/'             # Path to the image files
        imageSuffix = '.jpg'                      # File extention for the image files
        

        trainingFileName = relativeResultPath + 'training.csv'    # File for the training data
        validationFileName = relativeResultPath + 'validation.csv'# File for the training data
        testingFileName = relativeResultPath + 'testing.csv'      # File for the testing  data
        
        (xTrain, yTrain, xValidate, yValidate, xTest, yTest, xShape, yShape, 
         newRun) = loadUdacityChallenge2Data(rawDataFile, 
                                             trainingFileName,
                                             validationFileName,
                                             testingFileName,
                                             imagePath, 
                                             imageSuffix)
        
        # convert class vectors to binary class matrices
        numClasses = 10
        yTrain = keras.utils.to_categorical(yTrain, numClasses)
        yValidate = keras.utils.to_categorical(yValidate, numClasses)
        yTest = keras.utils.to_categorical(yTest, numClasses)
        yShape = numClasses
        

    elif (dataType == 'UdiacityChallenge2_Regression'):
        relativeRawPath = dataDirectoryPath + '/Datasets/UdacityChallenge2/'
        rawDataFile = relativeRawPath + 'final_example.csv' # The file path for the CSV file
        imagePath = relativeRawPath + 'center/'             # Path to the image files
        imageSuffix = '.jpg'                      # File extention for the image files
        

        trainingFileName = relativeResultPath + 'training.csv'    # File for the training data
        validationFileName = relativeResultPath + 'validation.csv'# File for the training data
        testingFileName = relativeResultPath + 'testing.csv'      # File for the testing  data
        
        (xTrain, yTrain, xValidate, yValidate, xTest, yTest, xShape, yShape, 
         newRun) = loadUdacityChallenge2Data(rawDataFile, 
                                             trainingFileName,
                                             validationFileName,
                                             testingFileName,
                                             imagePath, 
                                             imageSuffix)
    else:
        print('Invalid selection, not yet supported')
    
    
    return (xTrain, yTrain, xValidate, yValidate, xTest, yTest, xShape, yShape,
            newRun)


def loadUdacityChallenge2Data(rawDataFile, 
                              trainingFileNameCSV, 
                              validationFileNameCSV,
                              testingFileNameCSV, 
                              imagePath, 
                              imageSuffix):

    newRun = False

    # Check if the training and testing data has already been split into 
    # seperate files. If so, load those files, if not, load raw data file and 
    # then split into training and testing data sets, then save the split 
    # files.

    # Either they both exist or they do not. If neither does, need to start 
    # over to ensure integrety of the training and testing seperation.
    try:
        (trainingFileNames, trainingAngles) = getRawInputData(trainingFileNameCSV)
    except:
        newRun = True
    try:
        (validationFileNames, validationAngles) = getRawInputData(validationFileNameCSV)
    except:
        newRun = True
    try:
        (testingFileNames, testingAngles) = getRawInputData(testingFileNameCSV)
    except:
        newRun = True
    
    if newRun == True:

        # Get the raw data for analysis. Will get file names as well as
        # corresponding angles
        try:
            (fileNames, angles) = getRawInputData(rawDataFile)
        except:
            print('Raw data file is missing, cannot continue.')
            sys.exit(1)
        
        # Get a subset of data for training and another subset for testing
        (trainingFileNames, 
         trainingAngles, 
         validationFileNames, 
         validationAngles, 
         testingFileNames, 
         testingAngles) = splitTrainingTestingValidationData(fileNames, 
                                                             angles)
        
        saveInputFile(trainingFileNameCSV, trainingFileNames, trainingAngles)
        saveInputFile(validationFileNameCSV, validationFileNames, validationAngles)
        saveInputFile(testingFileNameCSV, testingFileNames, testingAngles)
        newRun = True   # Will need to retrain from scratch, regardless if there is already a model or not

    # Load the training and testing images
    (trainingImages, inputShape) = getImages(trainingFileNames, imagePath, imageSuffix)
    (validationImages, inputShape) = getImages(validationFileNames, imagePath, imageSuffix)
    (testingImages, inputShape) = getImages(testingFileNames, imagePath, imageSuffix)
    
    # Load the training and testing angles
    trainingAngles = np.array(trainingAngles)
    validationAngles = np.array(validationAngles)
    testingAngles = np.array(testingAngles)
    outputShape = 1     # ToDo: I want to parameterize this somehow
    
    return (trainingImages, trainingAngles, validationImages, validationAngles, testingImages, testingAngles, inputShape, outputShape, newRun)