# controller.py
# testing for race_track env by Mesut Xiaocheng Yang



class Controller:
	def __init__(self, track, car):
		self._track = track
		self._car = car

	def getObservation(self):
		return None

	def step(self, ac, i = 0):
		return self._car.getCarState().step(ac[0], ac[1], self._track, i)
