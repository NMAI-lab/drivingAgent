#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:58:03 2017

@author: patrickgavigan

Based on: 
https://elitedatascience.com/keras-tutorial-deep-learning-in-python

and: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Should also read:
https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html

"""



# Import libraries and modules
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
 
# Load a neural net model if available, make one if not
def setupModel(modelFileName, inputShape, outputShape, regression, newRun, 
               modelVersion):
#    np.random.seed(123)  # for reproducibility

    # Check if newRun is False, if it's True, we need a new model anyway, 
    # don't bother opening the old one.
    if (newRun == False):
        try:
            model = load_model(modelFileName)
        except:
            newRun = True

        # Validate that the model has the correct input and output shapes. If 
        # the shapes are not valid, set newRun to True.  
    if (newRun == False):
        validShape = validateModelInputOutputShape(model, inputShape, outputShape)
        if (validShape == False):
            newRun = True

    # Generate new model if needed
    if (newRun == True):
        # Generate a model
        model = defineModel(inputShape, outputShape, regression, modelVersion)

    return (model, newRun)

# Return true if the layers have the shape as defined by inputShape and 
# outputShape, false otherwise
def validateModelInputOutputShape(model, inputShape, outputShape):
    modelInputList = model.inputs
    modelInputTensor = modelInputList[0]
    modelInputDimension = modelInputTensor.get_shape()
    modelInputDimensionAsList = modelInputDimension.as_list()
    modelInputDimensionAsList.pop(0)
    inputsValid = (tuple(modelInputDimensionAsList) == inputShape)
                
    modelOutputList = model.outputs
    modelOutputTensor = modelOutputList[0]
    modelOutputDimension = modelOutputTensor.get_shape()
    modelOutputDimensionAsList = modelOutputDimension.as_list()
    modelOutputDimensionAsList.pop(0)
    outputsValid = (modelOutputDimensionAsList[0] == outputShape)
    
    return (inputsValid and outputsValid)

# Define model architecture
def defineModel(xShape, yShape, regression, modelVersion):
    model = Sequential()
    if modelVersion == 'A':
        # Old model
        model.add(Conv2D(32,(3,3), activation='relu', input_shape=xShape))
        model.add(Conv2D(32,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
    elif modelVersion == 'B':
        # Sample solution model: https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/cg23
        model.add(Conv2D(32,(3,3), activation='relu', input_shape=xShape))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
    else:
        # Keras MNISt tutorial model
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=xShape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))

    # Last layer parameters are common to all models
    if regression == True:          # Regression
        model.add(Dense(yShape))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    else:                       # Classification (use softmax)
        model.add(Dense(yShape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics=['accuracy'])

    return model
 
# Fit model on training data
def trainModel(model, xTrain, yTrain, xValidate, yValidate, vanillaGenerator,
               modelFileName, modelImageFileName, logFileName):
    maxBatchSize = 32
    maxNumEpochs = 500   # Need to generate this on its own
    maxQueue = 1
    stepsPerEpoch = len(xTrain) / maxBatchSize
    verbosity = 1
    validationSteps = len(xValidate) / maxBatchSize

    # Setup the data genearators (so that large data sets can fit in memory)
    trainingGenerator = setupGenerator(xTrain, vanillaGenerator)
    validationGenerator = setupGenerator(xValidate, vanillaGenerator)

    # Set up the callbacks (model file checkpoints, logfile of training, early 
    # stopping, and learning rade reduction)
    callbacksList = configureCallBacks(modelFileName, logFileName)

    # Fits the model on batches with real-time data augmentation:
    history = model.fit_generator(trainingGenerator.flow(xTrain,
                                                         yTrain, 
                                                         batch_size = maxBatchSize),
                                  steps_per_epoch = stepsPerEpoch,
                                  epochs = maxNumEpochs,
                                  verbose = verbosity,
                                  callbacks = callbacksList,
                                  validation_data = validationGenerator.flow(xValidate,
                                                                             yValidate),
                                  validation_steps = validationSteps,
                                  max_queue_size = maxQueue)
    
    # Plot the model - but need to figure out why it doesn't work first :)
    #plot_model(model, to_file = modelImageFileName)
    
    return (model, history)

# Evaluate model on test data
def evaluateModel(model, xTest, yTest, vanillaGenerator, regression):
    
    # Setup a data genearator (so that large data sets can fit in memory)
    dataGenerator = setupGenerator(xTest, vanillaGenerator)
    
    # Set constants
    maxQueue = 1
    percentageBias = 100
    
    (loss, accuracy) = model.evaluate_generator(dataGenerator.flow(xTest,
                                                                   yTest),
                                                steps = len(xTest),
                                                max_queue_size = maxQueue)

#    value = model.evaluate_generator(dataGenerator.flow(xTest,
#                                                        yTest),
#                                     steps = len(xTest),
#                                     max_queue_size = maxQueue)
                                      
    #(loss, accuracy) = model.evaluate(xTest, yTest, verbose = False)
    if regression == True:          # Regression
        lossFunction = 'Mean Squared Error'
    else:                           # Classification
        lossFunction = 'Categorical Crossentropy'
    accuracy = accuracy * percentageBias
    return (loss, accuracy, lossFunction)

def setupGenerator(data, vanillaGenerator):
    if vanillaGenerator == True:
        generator = ImageDataGenerator()
    else:
        generator = ImageDataGenerator(featurewise_center = True,
                                       featurewise_std_normalization = True,
                                       rotation_range = 20,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       horizontal_flip = True)
        generator.fit(data)     # Had a memory error here with ('UdiacityChallenge2', 'A', False) -> Investigate further

    return generator

def configureCallBacks(modelFileName, logFileName):

    verbosity = 1
    
    checkpoint = ModelCheckpoint(modelFileName,
                                 monitor = 'loss',
                                 verbose = verbosity,
                                 save_best_only = True)
    
    logger = CSVLogger(logFileName)
        
    stopper = EarlyStopping(monitor = 'loss',
                            min_delta = 0.01,
                            patience = 10,
                            verbose = verbosity,
                            mode = 'auto')
    
    rateReducer = ReduceLROnPlateau(monitor = 'loss', 
                                    factor = 0.2, 
                                    patience = 5, 
                                    verbose = verbosity, 
                                    min_lr = 0.001)
       
    callbacksList = [checkpoint, logger, stopper, rateReducer]
    return callbacksList

def plotHistory(history, lossFunction, accuracyFileName, lossFileName, 
                learningRateFileName):
    
    plotDualResult(history.history['acc'],
                   history.history['val_acc'],
                   'Epoch',
                   'Accuracy',
                   accuracyFileName)
    
    plotDualResult(history.history['loss'],
                   history.history['val_loss'],
                   'Epoch',
                   lossFunction,
                   lossFileName)
    
    plotResult(history.history['lr'],
               'Epoch',
               'Learning Rate',
               learningRateFileName)
    
def plotDualResult(training, validation, xLabel, yLabel, fileName):
    plt.plot(training)
    plt.plot(validation)
    plt.title(yLabel + ' vs. ' + xLabel)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(fileName, dpi = 500, transparent = False, bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window
    
def plotResult(data, xLabel, yLabel, fileName):
    plt.plot(data)
    plt.title(yLabel + ' vs. ' + xLabel)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.savefig(fileName, dpi = 500, transparent = False, bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window