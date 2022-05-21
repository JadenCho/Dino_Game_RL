from DinoEnv import DinoEnv
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

class PPO_():
	def __init__(self, width=120, height=120, chrome_path=None):
		self.env_fnc = lambda: DinoEnv(width=120, height=120, chrome_path=chrome_path)
		self.cpu = 2
		self.env = SubprocVecEnv([self.env_fnc for i in range(self.cpu)])
		#self.env = DinoEnv(width=120, height=120, chrome_path=chrome_path)
		self.model = PPO(CnnPolicy, self.env, verbose=1)
		self.save = 'chrome_dino_ppo'

	def run(self):
		#self.env.Env_Start()
		print('Begin training...')
		self.model.learn(total_timesteps=1000000)
		self.model.save(self.save)

		self.model = PPO.load(self.save, env=self.env)
		state = self.env.reset()
		print('Begin Test Runs...')
		for i in range(1000):
			action, _ = self.model.predict(state, deterministic=True)
			state, reward, done, info = self.env.step(action)
			#print(done)
			if done:
				#time.sleep(0.15)
				#print('RESET')
				state = self.env.reset()

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	ppo = PPO_(chrome_path=path)
	ppo.run()