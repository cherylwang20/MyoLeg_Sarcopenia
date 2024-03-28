import gym
from gym import spaces
import mujoco_py
import mj_envs
import neptune
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
#import helper_callback

from datetime import datetime
import torch
import time

step = False
sarco = False

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
	
class CustomNeptuneCallback(BaseCallback):
    def __init__(self, run):
        super(CustomNeptuneCallback, self).__init__(verbose=1)
        self.run = run
        # You might want to add more parameters here if needed

    def _on_step(self) -> bool:
        # Check if an episode has ended
        if 'episode' in self.locals["infos"][0]:
            episode_info = self.locals["infos"][0]['episode']
            # Log episodic information to Neptune
            self.run["metrics/episode_reward"].append(episode_info['r'])
            self.run["metrics/episode_length"].append(episode_info['l'])
        return True

dof_env = ['myoLegReachFixed-v5']

training_steps = 5000000
for env_name in dof_env:
	print('Begin training')

	start_time = time.time()
	time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	print(time_now + '\n\n')
	run = neptune.init_run(
    project="cherylwang20/MyoLeg-Stepping",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMGZmM2VhZC04ZDBjLTQxZGQtYjlkOS1hMzEyMGVkOTA3NzMifQ==",
    name =  time_now
	)  # your credentials



	env_name = 'myoLegReachFixed-v5'
	log_path = './standingBalance/policy_best_model/'+ env_name + '/' + time_now +'/'
	env = gym.make(f'mj_envs.robohive.envs.myo:{env_name}')
	print(env_name)
	print(env.rwd_keys_wt) 
	print(env.obs_dict.keys())
	eval_callback = EvalCallback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=10000, deterministic=True, render=False)
	print('max episode steps: ', env._max_episode_steps) 

	loaded_model = "2024_03_11_22_12_16"

	parameter = {
    "dense_units": 256,
    "activation": "relu",
    "max_episode_steps": env._max_episode_steps,
	"training_steps": training_steps,
	"loaded_model": loaded_model,
	"perturb_force_low": env.force_range[0],
    "perturb_force_high": env.force_range[1]
	}

	parameters = {**parameter, **env.rwd_keys_wt}
	run["model/parameters"] = parameters
	
	
	policy_kwargs = {
    	'activation_fn': torch.nn.modules.activation.ReLU,
    	'net_arch': {'pi': [256, 256], 'vf': [256, 256]}
		}
	#policy_kwargs = dict(activation_fn=torch.nn.Sigmoid, net_arch=(dict(pi=[64, 64], vf=[64, 64])))
	#model = PPO.load('standingBalance/policy_best_model/myoLegReachFixed-v2/2023_11_16_16_11_00/best_model',  env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)

	##model = PPO('MlpPolicy', env, verbose=0, policy_kwargs =policy_kwargs, tensorboard_log="C:/Users/chery/Documents/MyoLeg_Sarcopenia/StandingBalance/temp_env_tensorboard/"+env_name)
		#model = PPO.load(f'standingBalance/policy_best_model/myoLegReachFixed-v2/2024_01_03_10_31_35/best_model',  env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)
	model = PPO.load(f'standingBalance/policy_best_model/myoLegReachFixed-v5/' + loaded_model +'/best_model',  env, verbose=0, policy_kwargs=policy_kwargs, 
                  device= 'auto',tensorboard_log="./standingBalance/temp_env_tensorboard/"+env_name)
	
	obs_callback = TensorboardCallback()
	nep_callback = CustomNeptuneCallback(run=run)
	callback = CallbackList([eval_callback, nep_callback])

	model.learn(total_timesteps= training_steps, tb_log_name=env_name+"_" + time_now, callback=callback)
	model.save('ep_train_results')
	elapsed_time = time.time() - start_time

	hours = int(elapsed_time // 3600)
	minutes = int((elapsed_time % 3600) // 60)
	seconds = int(elapsed_time % 60)

	run.stop()
	print(time_now)
	print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.")