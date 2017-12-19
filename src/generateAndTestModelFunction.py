# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:27:13 2017

@author: patrickgavigan
"""
# Load external functions and classes
from DataLoader import getTrainingTestingValidationData
from NeuralNet import setupModel
from NeuralNet import trainModel
from NeuralNet import evaluateModel
from NeuralNet import plotHistory
from FileFunctions import printAndSaveResults
from FileFunctions import generateDirectory

import timeit
import datetime
import platform


def generateAndTestModel(dataSet, 
                         modelVersion, 
                         vanillaGenerator, 
                         workingDirectory = None):
    
    testID = dataSet + '_' + modelVersion
    if vanillaGenerator == True:
        testID = testID + '_vanilla'
    
    if ((dataSet == 'MNIST_Regression') or 
        (dataSet == 'UdiacityChallenge2_Regression')):
        regression = True
    else:
        regression = False
    
    # Start the clock for performance evaluation
    startTime = timeit.default_timer()
    platformName = platform.node()
    
    timeStamp = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
    startDatetimeString = datetime.datetime.today().strftime('%Y-%m-%d at %H:%M:%S')

    if (workingDirectory == None):    
        resultFolderName = timeStamp + '_' + platformName + '_' + testID
    else:
        # Do something about the directory
        resultFolderName = workingDirectory

    dataDirectoryPath = '../../../Machine Learning Data/'
    resultDirectoryPath = dataDirectoryPath + 'Results/Current Test/' + resultFolderName + '/'
    modelFileName = resultDirectoryPath + 'model.h5' # Model storage file
    modelImageFileName = resultDirectoryPath + 'model.png'
    trainingLogFileName = resultDirectoryPath + 'trainingLog.csv'
    resultFileName = resultDirectoryPath + 'results.txt'
    accuracyFileName = resultDirectoryPath + 'accuracyPlot.png'
    lossFileName = resultDirectoryPath + 'lossPlot.png'
    learningRateFileName = resultDirectoryPath + 'learningRatePlot.png'
    generateDirectory(resultDirectoryPath)

#    firstElementIndex = 0   # Value for the first element in the arrays
    step = 1                # First step - used in status outputs

    # Load training and testing data
    print('--- TEST START ---')
    print('Step ' + str(step) + ': Load training and testing data.')
    step = step + 1
    (xTrain, yTrain, xValidate, yValidate, xTest, yTest, xShape, yShape,
     newRun) = getTrainingTestingValidationData(dataSet, dataDirectoryPath, 
                                                resultDirectoryPath)

    # Set up the model 
    print('Step ' + str(step) + ': Setup the model.')
    (model, newRun) = setupModel(modelFileName, xShape, yShape, regression,
                             newRun, modelVersion)
    step = step + 1

    # Train the model if necessary
    if newRun == True:
        print('Step ' + str(step) + ': Train the model.')
        (trainedModel, history) = trainModel(model, 
                                             xTrain, 
                                             yTrain,
                                             xValidate,
                                             yValidate,
                                             vanillaGenerator,
                                             modelFileName, 
                                             modelImageFileName,
                                             trainingLogFileName)
    else:
        print('Step ' + str(step) + 
              ': Model already trained, skipping training step.')
        step = step + 1    

    # Test the model
    print('Step ' + str(step) + ': Test the model')
    step = step + 1
    (lossValue, accuracy, lossFunction) = evaluateModel(model,
                                                        xTest,
                                                        yTest,
                                                        vanillaGenerator,
                                                        regression)

    # Generate plots based on history
    try:
        history
    except:
        print('No history file available, plots cannot be generated.')
    else:
        plotHistory(history, lossFunction, accuracyFileName, lossFileName,
                    learningRateFileName)

    # Stop the timer and format
    elapsedTime = (datetime.timedelta(seconds=(timeit.default_timer() - startTime)))

    # Print and save results
    printAndSaveResults(resultFileName, lossFunction, lossValue, accuracy, 
                        elapsedTime, platformName, startDatetimeString, testID)
    print('--- TEST END ---')
    return