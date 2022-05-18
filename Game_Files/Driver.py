import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

#_chrome_options = webdriver.ChromeOptions()
#_chrome_options.binary_location ='C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe' # File path where chrome.exe is
#_chrome_options.add_argument("--mute-audio")
#_chrome_options.add_argument('--no-sandbox')
#_chrome_options.add_argument('--disable-dev-shm-usage')
#_chrome_options.add_argument("--window-size=900,900")

# Make sure that chromedriver.exe is in the same directory as this file
#_driver = webdriver.Chrome('chromedriver', options=_chrome_options)
#_driver.get('https://chromedino.com/')
#time.sleep(2)
#_driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)
#_driver.refresh()

class Game():
	def __init__(self, chrome_path):
		self.chrome_options = webdriver.ChromeOptions()
		self.chrome_options.binary_location =chrome_path # File path where chrome.exe is
		self.chrome_options.add_argument("--mute-audio")
		self.chrome_options.add_argument('--no-sandbox')
		self.chrome_options.add_argument('--disable-dev-shm-usage')
		self.chrome_options.add_argument("--window-size=900,900")
		self.driver = None

	def Start(self):
		self.driver = webdriver.Chrome('chromedriver', options=self.chrome_options)
		self.driver.get('https://chromedino.com/')

	def Up_Action(self):
		self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	game = Game(chrome_path=path)
	game.Start()
	time.sleep(2)
	game.Up_Action()