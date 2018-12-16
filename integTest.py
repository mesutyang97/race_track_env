# integTest.py
# Developed by Xiaocheng Mesut Yang for CS294-112 Fall 2018


from track import *
from controller import *
import numpy as np
import random
import matplotlib.pyplot as plt


def setup(numCars):
	track = Track(0.6, 0.3)
	controller_lst = []

	for i in range(numCars):
		carState_i = CarState(startRanking = i + 1)
		car_i = Car(carState_i, i + 1)

		track.initializeCar(carState_i)
		controller_i = Controller(track, car_i)
		controller_lst.append(controller_i)

	g = track.getGrid()
	print(g.shape)
	plt.imshow(g, cmap='gray')
	plt.show()
	return track, controller_lst



t, ctrl_lst = setup(1)

maxStep = 203

for i in range(maxStep):
	for ctrl in ctrl_lst:
		rd = random.uniform(-0.8, 0.8)
		ob, rew, done, _ = ctrl.step((rd, 1.0), 202)
		# print ("test side done:", done)
		#if done:
		#	ctrl_lst.remove(ctrl)
	if i % 5 == 0:
		t.rebuildTrack()


