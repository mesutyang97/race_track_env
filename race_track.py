# race_track.py
# This is a wrapper file for track.py
# Developed by Xiaocheng Mesut Yang for CS294-112 Fall 2018

import numpy as np
from gym import spaces
from gym import Env
from track import *
import random

class RaceTrackEnv(Env):
    """
    Race Track Environment
    """
    def __init__(self, num_cars=1, miu = 0.8, dot_miu = 0.3, env_window_w = 500, obs_window_w = 10, sensor_only = 1, max_path_length = 200):
        self.num_cars = num_cars
        self.miu = miu
        self.dot_miu = dot_miu
        self.env_window_w = env_window_w
        self.obs_window_w = obs_window_w
        self.sensor_only = sensor_only
        self.max_path_length = max_path_length

        self.reset()
        if sensor_only == 1:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_window_w * obs_window_w,))
        elif sensor_only == 2:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_window_w * obs_window_w + 1,))
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_window_w * obs_window_w + 4,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def reset_task(self, is_evaluation=False):
        self.enablePrint = False
        return None

    def reset(self, car_i = 0):
        self._track = Track(self.miu, self.dot_miu)

        rec = (random.random() < 0.015)
        if rec == True:
            print("should record now")

        if self.num_cars == 1:
            self._carState_main = CarState(startRanking = 1, env_window_w = self.env_window_w, obs_window_w = self.obs_window_w, 
                sensor_only = self.sensor_only, max_total_T = self.max_path_length, record = rec)
            self._car_main = Car(self._carState_main, 1)
            self._track.initializeCar(self._carState_main)
        else:
            self._car_lst = []
            for i in range(0, self.num_cars):
                carState_i = CarState(startRanking = i + 1, env_window_w = self.env_window_w, obs_window_w = self.obs_window_w, 
                sensor_only = self.sensor_only, max_total_T = self.max_path_length, record = rec)
                car_i = Car(carState_i, i + 1)
                self._track.initializeCar(carState_i)
                self._car_lst.append(carState_i)
        self.count = 0

        return self._get_obs()

    def _get_obs(self, car_i = 0):
        if self.num_cars == 1:
            return self._carState_main.getObservation(self._track)
        else:
            carState_i = self._car_lst[car_i]
            return carState_i.getObservation(self._track)

    def step(self, action, i = 0, car_i = 0, m_done = False):
        self.count += 1
        if self.count % 5 == 0:
            self._track.rebuildTrack()
        action = action.flatten()
        if self.num_cars == 1:
            return self._carState_main.step(action[0], action[1], self._track, i, manual_done = m_done, enablePrint = self.enablePrint)
        else: 
            carState_i = self._car_lst[car_i]
            return carState_i.step(action[0], action[1], self._track, i, manual_done = m_done, enablePrint = self.enablePrint)


    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed