# utils.py

import math
import numpy as np

def getAngle(v):
	if v[1] == 0:
		denom = 0.001
	else:
		denom = v[1]
	theta = np.arctan(v[0]/denom)
	if v[1] < 0:
		theta += math.pi

	return theta

def rotateVector(v, dtheta):
	old_theta = getAngle(v)

	new_theta = old_theta + math.radians(dtheta)
	return (np.sin(new_theta), np.cos(new_theta))


def ftToMm(lengthInFeet):
	return int(lengthInFeet * 304.8)

def downSample(length):
	return int(length/10)

def sameDirection(v1, v2):
	return np.dot(np.array(v1), v2) > 0

def numEq(a, b):
	return a < b + 0.001 and a > b - 0.001

def transformRectangle(loc, v, length, width):
	# The negative sign is used to address the assymatry between angle and sign
	theta = -getAngle(v)
	rect = np.array([(-length/2, -width/2), (length/2, -width/2), (length/2, width/2), (-length/2, width/2), (-length/2, -width/2)])
	R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	y, x = np.array(loc) 
	offset = np.array([x, y])
	transformed_rect = np.dot(rect, R) + offset
	return transformed_rect