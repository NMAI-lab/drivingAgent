# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:49:53 2017

@author: Patrick
"""

import math

# Calculate the start and end points for a line based on an origin location,
# length, and an angle in RAD.
def calculateLinePoints(originBiasX, originBiasY, length, angleRad):
    x1 = originBiasX
    y1 = originBiasY
    x2 = length * math.sin(angleRad) + x1
    y2 = length * math.cos(angleRad) + y1
    return (x1, y1, x2, y2)