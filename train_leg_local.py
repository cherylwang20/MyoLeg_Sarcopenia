import gym
from gym import spaces
import mujoco_py
import mj_envs
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback

from datetime import datetime
import torch
import time

class TensorboardCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""

	def __init__(self, verbose=0):
	    super(TensorboardCallback, self).__init__(verbose)

	def _on_step(self) -> bool:
	    # Log scalar value (here a random variable)
	    value = self.training_env.get_obs_vec()
	    self.logger.record("obs", value)
	
	    return True

dof_env = ['myoLegReachFixed-v2']

#env = gym.make('mj_envs.robohive.envs.myo:myoLegStairTerrainWalk-v0')
env = gym.make('mj_envs.robohive.envs.myo:myoLegReachFixed-v2')

training_steps = 15000000
for env_name in dof_env:
	print('Begin training')
	print(env_name)
	
	start_time = time.time()
	time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	print(time_now + '\n\n')
	log_path = 'C:/Users/chery/Documents/MyoLeg_Sarcopenia/standingBalance/policy_best_model/'+ env_name + '/' + time_now +'/'

	env = gym.make(env_name)
	print(env.rwd_keys_wt) 
	print(env.obs_dict.keys())
	eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)
	print('max episode steps: ', env._max_episode_steps) 
	obs = env.get_obs_vec()
	print('obs len:', len(obs))
	#policy_kwargs = dict(activation_fn=torch.nn.Sigmoid, net_arch=[dict(pi=[64], vf=[64])])
	policy_kwargs = dict(activation_fn=torch.nn.Sigmoid, net_arch=(dict(pi=[64, 64], vf=[64, 64])))
	model = PPO('MlpPolicy', env, ent_coef= 0.001,verbose=0, policy_kwargs =policy_kwargs, tensorboard_log="C:/Users/chery/Documents/MyoLeg_Sarcopenia/standingBalance/temp_env_tensorboard/"+env_name)
	#model = PPO.load('./policy_best_model/myoLegReachFixed-v1/2023_05_17_18_05_04/best_model.zip', env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./temp_env_tensorboard/")

	obs_callback = TensorboardCallback()

	callback = CallbackList([obs_callback, eval_callback])

	model.learn(total_timesteps= training_steps, tb_log_name=env_name+"_" + time_now, callback=eval_callback)
	model.save('ep_train_results')
	elapsed_time = time.time() - start_time

	hours = int(elapsed_time // 3600)
	minutes = int((elapsed_time % 3600) // 60)
	seconds = int(elapsed_time % 60)

	print(time_now)
	print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.")