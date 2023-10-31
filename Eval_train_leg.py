import gym
from gym import spaces
import mujoco_py
import mj_envs
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch
import matplotlib.pyplot as plt
import skvideo
skvideo.setFFmpegPath(r"C:\ffmpeg\bin")
#skvideo.setFFmpegPath(r'C:\users\chery\appdata\local\packages\pythonsoftwarefoundation.python.3.8_qbz5n2kfra8p0\localcache\local-packages\python38\site-packages\ffmpeg')
import skvideo.io
import os
import random

nb_seed = 1

movie = True
path = r'C:\Users\chery\Documents\MyoLeg_Sarcopenia'
model = PPO.load(r'C:\Users\chery\Documents\MyoLeg_Sarcopenia\standingBalance\policy_best_model\myoLegReachFixed-v1\2023_10_30_16_32_57\best_model.zip')
s, m, t = [], [], []

env = gym.make('mj_envs.robohive.envs.myo:myoLegReachFixed-v1')

env.reset()

random.seed() 

frames = []

for _ in range(1000):
	
	obs = env.get_obs_vec()
	
	action, _ = model.predict(obs, deterministic=True)

	env.sim.data.ctrl[:] = action
	env.step(action)



	t.append(env.obs_dict['reach_err'])
	#s.append(env.sim.data.qpos[joint_interest_id])
	m.append(action)
	if movie:
		#env.sim.render(mode='window') # GUI
		frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=1) # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
	#print(_)
		frames.append(frame[::-1,:,:])
	
env.close()

if movie:
	os.makedirs(path+'/videos/myoLegReachFixed/', exist_ok=True)
# make a local copy
	skvideo.io.vwrite(path+'/videos/myoLegReachFixed/video.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
	
fig = plt.figure('myoLegReachFixed')
plt.subplot(211)
plt.plot(np.array(t),'--',label='traj')
plt.plot(np.array(s),label='policy')
plt.legend((['x', 'y', 'z']))
plt.subplot(212)
plt.imshow(np.array(m).T, aspect='auto');plt.colorbar()
plt.title('Muscle Activations')
os.makedirs(path+'/image/myoLegReachFixed/', exist_ok=True)
fig.savefig(path+'/image/myoLegReachFixed/image.png', dpi=fig.dpi)

plt.close(fig)