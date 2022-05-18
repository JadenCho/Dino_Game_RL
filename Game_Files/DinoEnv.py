import gym
import numpy as np
from Driver import Game
from collections import deque
from gym import spaces
import time

class DinoEnv(gym.Env):
	def __init__(self, width=120, height=120, chrome_path=None):
		self.screen_width = width
		self.screen_height = height

		self.action_space = spaces.Discrete(3) # Do nothing, jump, crouch
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 4), dtype=np.uint8)

		self.state_queue = deque(maxlen=4)

		self.game = Game(chrome_path)

		self.action_list = [self.game.Nothing_Action(), self.game.Up_Action(), self.game.Down_Action()]		

	def Env_Start(self):
		self.game.Start()

	def step(self):
		'''
		Returns Observation, reward, done, other
		'''
		pass

	def render(self):
		pass

	def reset(self):
		self.game.Restart()


	def close(self):
		pass

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	game = DinoEnv(chrome_path=path)
	game.Env_Start()
	time.sleep(5)
	game.reset()