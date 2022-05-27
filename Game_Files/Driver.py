import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Game():
	def __init__(self, chrome_path):
		self.chrome_options = webdriver.ChromeOptions()
		self.chrome_options.binary_location = chrome_path # File path where chrome.exe is
		self.chrome_options.add_argument("--mute-audio")
		self.chrome_options.add_argument("--headless")
		self.chrome_options.add_argument('--no-sandbox')
		self.chrome_options.add_argument('--disable-dev-shm-usage')
		self.chrome_options.add_argument('start-maximized')
		self.driver = webdriver.Chrome('chromedriver', options=self.chrome_options)
		self.driver.get('https://tuckercraig.com/dino/')

	def Start(self):
		'''
		Open the Game Instance in Chrome
		'''
		self.driver.get('https://tuckercraig.com/dino/')

	def Action(self, action):
		'''
		Perform action
		'''
		self.driver.find_element(By.TAG_NAME, 'body').send_keys(action)

	def Refresh(self):
		'''
		Refresh the Chrome Tab
		'''
		self.driver.refresh()
		WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))

	def Restart(self):
		'''
		Refresh the Chrome Tab and start the game again
		'''
		WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))
		self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)

	def Get_Score(self):
		'''
		Return the score of the gane
		'''
		score = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
		score = ''.join(score)

		return score

	def Img_State(self):
		'''
		Return the image of the current state
		'''
		img = self.driver.execute_script("return document.querySelector('canvas.runner-canvas').toDataURL()")
		return img

	def Done_State(self):
		'''
		Return whether the dino has crashed or not
		'''
		done = self.driver.execute_script("return Runner.instance_.crashed")
		
		return done