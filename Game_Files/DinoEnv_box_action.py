import gym
import numpy as np
from Driver import Game
from collections import deque
from gym import spaces
from io import BytesIO
from PIL import Image
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import base64
import cv2
import time
import matplotlib.pyplot as plt

class DinoEnv_box_action(gym.Env):
	def __init__(self, width=120, height=120, chrome_path=None):
		self.screen_width = width
		self.screen_height = height

		self.action_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.float32)
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 4), dtype=np.uint8)

		self.state_queue = deque(maxlen=4)

		self.game = Game(chrome_path)

		self.action_list = [Keys.ARROW_LEFT, Keys.ARROW_UP, Keys.ARROW_DOWN]
		action_ = ActionChains(self.game.driver)
		self.key_press = [action_.key_down(act) for act in self.action_list]
		self.key_unpress = [action_.key_up(act) for act in self.action_list]

	def Env_Start(self):
		'''
		Start the Dino Game Instance
		'''
		self.game.Start()

	def step(self, action):
		'''
		Returns Observation, reward, done, other
		'''
		self.game.Action(self.action_list[np.argmax(action)])

		next_state = self.next_state()

		done = self.done_state()

		reward = 1 if not done else -100

		score = self.game.Get_Score()

		#time.sleep(0.02)

		return next_state, reward, done, {'score': score}

	def reset(self):
		'''
		Reset the Dino Game Instance
		'''
		self.game.Restart()

		return self.next_state()

	def get_state_img(self):
		'''
		Returns an image of the current state of the game
		'''
		LEADING_TEXT = "data:image/png;base64,"
		img = self.game.Img_State()
		img = img[len(LEADING_TEXT):]

		return np.array(Image.open(BytesIO(base64.b64decode(img))))

	def next_state(self):
		'''
		Processes the image of the state
		'''
		img = cv2.cvtColor(self.get_state_img(), cv2.COLOR_BGR2GRAY)
		img = img[:, :480] # Cropping
		img = cv2.resize(img, (self.screen_width, self.screen_height)) # Resize

		self.state_queue.append(img)

		if len(self.state_queue) < 4:
			return np.stack([img] * 4, axis=-1)
		else:
			return np.stack(self.state_queue, axis=-1)
		#return img

	def Score(self):
		'''
		Obtain and return score from the Game Instance
		'''
		score = self.game.Get_Score()
		return score

	def done_state(self):
		'''
		Check and return whether the Dino has crashed or not
		'''
		return self.game.Done_State()

#if __name__ == "__main__":
#	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
#	d = DinoEnv(chrome_path=path)
#	state = d.reset()
#	img = []
#	img.append(state)
#	for i in range(15):
#		next_state, reward, done, _ = d.step(1)
#		img.append(next_state)
#		plt.figure(i)
#		plt.imshow(img[i])
#	plt.show()