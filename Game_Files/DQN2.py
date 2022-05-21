from DinoEnv import DinoEnv
import os
import time
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

class DQN2():
	def __init__(self, width=120, height=120, chrome_path=None):
		self.env_fnc = lambda: DinoEnv(width=120, height=120, chrome_path=chrome_path)
		self.env = DummyVecEnv([self.env_fnc])
		#self.env = DinoEnv(width=120, height=120, chrome_path=chrome_path)
		self.model = DQN(CnnPolicy, self.env, verbose=1, buffer_size=5000)
		self.save = 'chrome_dino'

	def run(self):
		#self.env.Env_Start()
		print('Begin training...')
		self.model.learn(total_timesteps=1000000)
		self.model.save(self.save)

		self.model = DQN.load(self.save, env=self.env)
		state = self.env.reset()
		print('Begin Test Runs...')
		for i in range(1000):
			action, _ = self.model.predict(state)
			state, reward, done, info = self.env.step(action)
			#print(done)
			if done:
				#time.sleep(0.15)
				#print('RESET')
				state = self.env.reset()

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	dqn = DQN2(chrome_path=path)
	dqn.run()