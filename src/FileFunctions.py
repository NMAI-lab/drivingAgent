#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:27:17 2017

@author: patrickgavigan

"""

import numpy as np
from PIL import Image
import csv
import os
import errno

# Get the raw input data
def getRawInputData(dataFileName):
    # Initialize constants
    fileNameColumnNumber = 0            # Column containing the file names
    angleColumnNumber = 1               # Column containing the angles

    # Initialize variables
    angle = list()          # Initialize the list of angles
    fileName = list()       # Initialize the list of file names

    # Parse the CSV file, if it's there
    try:
        file = open(dataFileName)
    except:
        raise
        
    with file:
        reader = csv.reader(file, delimiter=',')            # Get the data
        firstRow = True                                     # Set a flag for the first row headers
        for row in reader:                                  # Iterate through the rows
            if (firstRow == False):                         # Make sure that this is not the first row
                fileName.append(row[fileNameColumnNumber])  # Get the file name and add it to the list
                angle.append(float(row[angleColumnNumber])) # get the steering angle and add it to the list as a float
            else:                                           # Deal with the header row (ignore it)
                firstRow = False                            # After the first iteration, won't see headders again
        file.close()
    return fileName, angle

# ToDo: Implement using the python CSV tools
def saveInputFile(fileName, imageFiles, angles):
    with open(fileName, 'a') as csvFile:
        csvFile.write('frame_id,steering_angle\n')
        for i in range(len(imageFiles)):
            rowData = str(imageFiles[i]) + ',' + str(angles[i]) + '\n'
            csvFile.write(rowData)
        csvFile.close()

def getImageShape(image, path, imageSuffix):
    fileName = path + image + imageSuffix  # Generate the file name
    imageData = Image.open(fileName)   # Open the image
    # Documentation: http://pillow.readthedocs.io/en/3.4.x/reference/Image.html#PIL.Image.Image
    height = imageData.height          # Determine the width of the image
    width = imageData.width            # Determine the height of the image
    depth = len(imageData.getbands()) #dimentions[2]
    shape = (height, width, depth)
    return shape

def getImages(fileNames, path, suffix):
    startIndex = 0
    images = []                                         # Initialize list of images
    shape = getImageShape(fileNames[startIndex], path, suffix)
    for i in range(startIndex,len(fileNames)):          # iterate through all the file names
        fileName = path + fileNames[i] + suffix         # Generate the file name
        images.append(np.array(Image.open(fileName)))   # Append the current image to the list as an numpy array
    images = np.array(images)                           # Turn the whole list into a big numpy array
    return images, shape                                # Return the result

def printAndSaveResults(fileName, lossFunction, lossValue, accuracy, 
                        elapsedTime, platformName, startDatetimeString, 
                        testID):
    # Prepare the messages
    headder = '--- TEST REPORT ---'
    testTypeMessage = 'ID: ' + testID
    platformMessage = 'Platform: '+ platformName
    startTimeMessage = 'Start date and time: ' + startDatetimeString
    elapsedTimeResult = 'Time elapsed (HH:MM:SS): ' + str(elapsedTime)
    lossResult = lossFunction + ': ' + str(lossValue)
    accuracyResult = 'Accuracy: ' + str(accuracy) + '%'
    
    # Save to the results file
    f = open(fileName,'w')
    f.write(headder + '\n')
    f.write(testTypeMessage + '\n')
    f.write(platformMessage + '\n')
    f.write(startTimeMessage + '\n')
    f.write(elapsedTimeResult + '\n') 
    f.write(lossResult + '\n')
    f.write(accuracyResult + '\n')
    f.close()
    
    # Print results to the console
    print(headder)
    print(testTypeMessage)
    print(platformMessage)
    print(startTimeMessage)
    print(elapsedTimeResult)
    print(lossResult)
    print(accuracyResult)
    

def generateDirectory(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise