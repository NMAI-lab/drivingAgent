#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author:     Patrick Gavigan
Date:       15 October 2017

This is a preliminary attempt at performing the Udacity data challenge #2. It
will parse a CSV file that contains file names for jpg images and corresponding
steering angles. It will train a Neural Net using TensorFlow using a portion 
if this data and then test the effectiveness using the remaining data.
"""

# Info on how to send output to a file and the terminal window
# https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file

from generateAndTestModelFunction import generateAndTestModel
from postProcessing import postProcessImages

# function protoype: 
# generateAndTestModel(dataSet, modelVersion, vanillaGenerator)

workingDirectoryName = 'workingDirectory'

#generateAndTestModel('UdiacityChallenge2_Regression', 'B', True)
#generateAndTestModel('UdiacityChallenge2_Regression', 'B', False)
#
#generateAndTestModel('UdiacityChallenge2_Regression', 'C', True)
#generateAndTestModel('UdiacityChallenge2_Regression', 'C', False, workingDirectoryName)
#
#generateAndTestModel('UdiacityChallenge2_Regression', 'A', True)
#generateAndTestModel('UdiacityChallenge2_Regression', 'A', False)
#
generateAndTestModel('UdiacityChallenge2_Regression', 'A', True, workingDirectoryName)
#generateAndTestModel('MNIST', 'A', False)
#
#generateAndTestModel('MNIST', 'B', True)
#generateAndTestModel('MNIST', 'B', False)
#
#generateAndTestModel('MNIST', 'C', True)
#generateAndTestModel('MNIST', 'C', False)

#generateAndTestModel('MNIST_Regression', 'A', False)
#generateAndTestModel('UdiacityChallenge2_Classification', 'B', False)

path = '../../../Machine Learning Data/Results/Current Test/workingDirectory/'
fileType = '.jpg'
postProcessImages(path, 'testImage', fileType, 0.1, 0.15)