from DinoEnv import DinoEnv
import os
import time
import math
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

class DQN_():
	def __init__(self, width=120, height=120, chrome_path=None):
		self.env_fnc = lambda: DinoEnv(width=120, height=120, chrome_path=chrome_path)
		self.env = DummyVecEnv([self.env_fnc])
		#self.env = DinoEnv(width=120, height=120, chrome_path=chrome_path)
		self.model = DQN(CnnPolicy, self.env, verbose=1, buffer_size=5000)
		self.save = 'chrome_dino_dqn'

	def run(self):
		#self.env.Env_Start()
		print('Begin training...')
		start_train = time.time()
		self.model.learn(total_timesteps=2000000)
		self.model.save(self.save)

		end_train = time.time()
		self.model = DQN.load(self.save, env=self.env)
		
		print('Begin Test Runs...')
		total_r = []
		max = 0
		for e in range(1000):
			done = False
			time.sleep(0.25)
			state = self.env.reset()
			r = 0
			while not done:
				action, _ = self.model.predict(state, deterministic=True)
				state, reward, done, info = self.env.step(action)
				r +=1
				if done:
					if r > max:
						max = r
					print(f"episode: {e+1}/{1000}, score: {r}, max score: {max}") 
					total_r.append(r)
					state = self.env.reset()

		elapse = np.abs(start-end) 
		hour = elapse/3600 
		minute = np.abs(elapse - (math.floor(hour) * 3600))/60 
		seconds = np.abs(minute - math.floor(minute)) * 60 
		print(f'Total Training Time: {hour} hours, {minute} minutes, {seconds} seconds')

		epi = np.linspace(0, 1000, 1000) 
		plt.plot(epi, total_r) 
		plt.xlabel('Episodes') 
		plt.ylabel('Total Reward') 
		plt.show()

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	dqn = DQN_(chrome_path=path)
	dqn.run()