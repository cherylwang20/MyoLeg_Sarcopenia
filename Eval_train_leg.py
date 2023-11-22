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
from stable_baselines3.common.callbacks import EvalCallback
import torch
import matplotlib.pyplot as plt
import skvideo
#skvideo.setFFmpegPath(r"C:\ffmpeg\bin")
#skvideo.setFFmpegPath(r'C:\users\chery\appdata\local\packages\pythonsoftwarefoundation.python.3.8_qbz5n2kfra8p0\localcache\local-packages\python38\site-packages\ffmpeg')
import skvideo.io
import os
import random
from tqdm.auto import tqdm

nb_seed = 1

sarco = False

movie = True
path = 'C:/Users/chery/Documents/MyoLeg_Sarcopenia'

model_num = '2023_11_16_16_11_00'
if sarco:
  env_name = 'myoSarcLegReachFixed-v3'
  #model = PPO.load(r"C:/Users/chery\Documents/MyoLeg_Sarcopenia/standingBalance-sarco/policy_best_model/myoSarcLegReachFixed-v2/2023_11_09_11_44_06/best_model")
  model = PPO.load(path+'/standingBalance-sarco/policy_best_model'+ '/'+ env_name + '/' + model_num +
                  r'/best_model')
  env = gym.make('mj_envs.robohive.envs.myo:myoSarcLegReachFixed-v3')

else:
  env_name = 'myoLegReachFixed-v2'
  #model = SAC.load(r"C:/Users/chery\Documents/MyoLeg_Sarcopenia/standingBalance/SAR_RL_Leg_Stability_model_myoLegReachFixed-v2_0")
  #model = PPO.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num +
                  #r'/best_model')
  model=PPO.load('ep_train_results')
  env = gym.make('mj_envs.robohive.envs.myo:myoLegReachFixed-v2')

s, m, t = [], [], []

env.reset()

random.seed() 

frames = []
for _ in tqdm(range(3)):
    ep_rewards = []
    done = False
    obs = env.reset()
    for _ in range(250):
          obs = env.get_obs_vec()
          
          action, _ = model.predict(obs, deterministic=True)
          env.sim.data.ctrl[:] = action
          obs, reward, done, info = env.step(action)
          t.append(env.obs_dict['reach_err']) #s.append(env.sim.data.qpos[joint_interest_id])
          m.append(action)
          if movie:
                  frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=1) 
                  frame = np.rot90(np.rot90(frame))
            # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                  frames.append(frame[::-1,:,:])
                  #env.sim.mj_render(mode='window') # GUI


# evaluate policy
all_rewards = []
for _ in tqdm(range(20)): # 20 random targets
  ep_rewards = []
  done = False
  obs = env.reset()
  step = 0
  while (not done) and (step < 700):
      # get the next action from the policy
      #env.mj_render()
      action, _ = model.predict(obs)
      # take an action based on the current observation
      obs, reward, done, info = env.step(action)
      ep_rewards.append(reward)
      step += 1
  all_rewards.append(np.sum(ep_rewards))
env.close()
print(f"Average reward: {np.mean(all_rewards)} over 20 episodes")


env.close()

if movie:
    if sarco:
      os.makedirs(path+'/videos/sarco' +'/' + env_name, exist_ok=True)
      skvideo.io.vwrite(path+'/videos/sarco'  +'/' + env_name + '/' + model_num + 'video.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
    else:
      os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
      skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + 'video.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
	
fig = plt.figure('myoLegReachFixed')
plt.subplot(211)
plt.plot(np.array(t),'--',label='traj')
plt.plot(np.array(s),label='policy')
plt.legend((['x', 'y', 'z']))
plt.subplot(212)
plt.imshow(np.array(m).T, aspect='auto');plt.colorbar()
plt.title('Muscle Activations')
os.makedirs(path+'/image' +'/' + env_name, exist_ok=True)
fig.savefig(path+'/image'  +'/' + env_name + '/' + model_num + 'image.png', dpi=fig.dpi)

plt.close(fig)