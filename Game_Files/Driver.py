import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Game():
	def __init__(self, chrome_path):
		self.chrome_options = webdriver.ChromeOptions()
		self.chrome_options.binary_location =chrome_path # File path where chrome.exe is
		self.chrome_options.add_argument("--mute-audio")
		self.chrome_options.add_argument('--no-sandbox')
		self.chrome_options.add_argument('--disable-dev-shm-usage')
		self.chrome_options.add_argument("--window-size=900,900")
		self.driver = webdriver.Chrome('chromedriver', options=self.chrome_options)

	def Start(self):
		self.driver.get('https://chromedino.com/')
		WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))
		self.Up_Action()

	def Up_Action(self):
		self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)

	def Down_Action(self):
		self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_DOWN)

	def Nothing_Action(self):
		self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_RIGHT)

	def Refresh(self):
		self.driver.refresh()
		WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))

	def Restart(self):
		self.Refresh()
		self.Up_Action()

	def Get_Score(self):
		score = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
		score = ''.join(score)

		return int(score)