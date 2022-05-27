from DinoEnv import DinoEnv
import os
import time
import math
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

class PPO_():
	def __init__(self, width=120, height=120, chrome_path=None, cpu_process=1):
		self.path = chrome_path
		self.episodes = 500
		self.env_fnc = lambda: DinoEnv(width=120, height=120, chrome_path=self.path)
		self.env = SubprocVecEnv([self.env_fnc for i in range(cpu_process)])
		self.model = PPO(CnnPolicy, self.env, verbose=1)
		self.save = 'chrome_dino_ppo'
		self.checkpoint = CheckpointCallback(save_freq=25000, save_path='./.checkpoints/', name_prefix=self.save)

	def run(self):
		#self.env.Env_Start()
		print('Begin training...')
		start_train = time.time()
		self.model.learn(total_timesteps=500000, callback=[self.checkpoint])
		self.model.save(self.save)

		end_train = time.time()
		self.test_env = DummyVecEnv([self.env_fnc])
		self.model = PPO.load(self.save, env=self.test_env)
		
		print('Begin Test Runs...')
		total_r = []
		max = 0
		for e in range(self.episodes):
			done = False
			state = self.test_env.reset()
			r = 0
			time.sleep(1)
			while not done:
				action, _ = self.model.predict(state, deterministic=True)
				state, reward, done, info = self.test_env.step(action)
				r +=1
				if done:
					if r > max:
						max = r
					print(f"episode: {e+1}/{self.episodes}, score: {r}, max score: {max}") 
					if r>1:
						total_r.append(r)
					#state = self.env.reset()

		elapse = np.abs(start-end) 
		hour = elapse/3600 
		minute = np.abs(elapse - (math.floor(hour) * 3600))/60 
		seconds = np.abs(minute - math.floor(minute)) * 60 
		print(f'Total Training Time: {hour} hours, {minute} minutes, {seconds} seconds')	

		epi = np.linspace(0, len(total_r), len(total_r)) 
		plt.plot(epi, total_r) 
		plt.xlabel('Episodes') 
		plt.ylabel('Total Reward') 
		plt.show()

if __name__ == "__main__":
	path = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
	ppo = PPO_(chrome_path=path, cpu_process=2)
	ppo.run()