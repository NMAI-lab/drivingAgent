# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:49:53 2017

@author: Patrick
"""

from PIL import Image, ImageDraw, ImageColor
import math


# Calculate the start and end points for a line based on an origin location,
# length, and an angle in RAD.
def calculateLinePoints(originBiasX, originBiasY, length, angleRad):
    x1 = originBiasX
    y1 = originBiasY
    x2 = length * math.sin(angleRad) + x1
    y2 = length * math.cos(angleRad) + y1
    return (x1, y1, x2, y2)


def flipY(value, height):
    newValue = height - value    
    return newValue


def drawLine(angleRad, colour, im):
    width = im.size[0]
    height = im.size[1]
    originBiasX = width/2
    originBiasY = 1
    length = height/2
    (x1, y1, x2, y2) = calculateLinePoints(originBiasX, originBiasY, length, angleRad)
    draw = ImageDraw.Draw(im)
    y1 = flipY(y1, height)
    y2 = flipY(y2, height)
    draw.line((x1, y1, x2, y2), fill = colour)
    del draw
    return im
    
def printAngle(angleRad, colour, im):
    
    return im

def visualizeSteeringAngle(image, truth, predicted):
    truthColour = ImageColor.getrgb('green')
    predictedColour = ImageColor.getrgb('blue')
    image = drawLine(truth, truthColour, image)
    image = drawLine(predicted, predictedColour, image)
    image = printAngle(truth, truthColour, image)
    image = printAngle(predicted, predictedColour, image)
    
    return image


def postProcessImages(path, fileName, fileType, truth, predicted):
    imagePathOpen = path + fileName + fileType
    imagePathSave = path + fileName + '_post' + fileType
    im = Image.open(imagePathOpen)
    im = visualizeSteeringAngle(im, truth, predicted)
    im.save(imagePathSave)