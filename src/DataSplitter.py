#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:18:12 2017

@author: patrickgavigan


Split provided data into training and testing data. Will need to put more 
thought into how we do this. For now, return the first 75% of the lists as
training and the last 25% as testing.
This function assumes that inputData and outputData are the same length. This
function is not bulletproof

Other ideas for how this could work: take every nth data element as testing,
take a random sample for testing and remove those elements from the training
list, split the data into blocks and take one of those blocks as testing (would
need more input parameters for specifying the block sizes and which block to
use)

"""

from sklearn.cross_validation import train_test_split

def splitTrainingTestingValidationData(x, y):
    testingRatio = 0.3    # Percentage of the data used for testing and validation
    (xTrain, xTest, yTrain, yTest) = splitData(x, y, testingRatio)
    (xValidate, yValidate, xTest, yTest) = splitValidationTest(xTest, yTest)
    return (xTrain, yTrain, xValidate, yValidate, xTest, yTest)    

def splitValidationTest(x, y):
    validationTestingRatio = 0.5
    (xValidate, xTest, yValidate, yTest) = splitData(x, y, 
                                                     validationTestingRatio)
    return (xValidate, yValidate, xTest, yTest)  

def splitData(x, y, ratio):
    seed = 42
    (x1, x2, y1, y2) = train_test_split(x, y, test_size = ratio, 
                                        random_state = seed)
    return (x1, x2, y1, y2)