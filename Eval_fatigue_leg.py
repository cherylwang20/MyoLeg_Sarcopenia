import gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
import os
import random
from tqdm.auto import tqdm

nb_seed = 1

movie = True
path = './'

model_num = '2024_02_20_15_03_01'


env_name = 'myoFatiLegReachFixed-v4'

model = PPO.load(path+'/standingBalance-Fatigue/policy_best_model'+ '/'+ env_name + '/' + model_num +
                 r'/best_model')
env = gym.make(f'mj_envs.robohive.envs.myo:{env_name}')

s, m, t = [], [], []

env.reset()

random.seed() 

frames = []
view = 'front'
for _ in tqdm(range(2)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    for _ in tqdm(range(1400)):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()
          
          action, _ = model.predict(obs, deterministic=True)
          #env.sim.data.ctrl[:] = action
          obs, reward, done, info = env.step(action)
          acti = env.sim.data.act[env.sim.model.actuator_name2id('tibant_l')].copy()
          #t.append(env.obs_dict['reach_err']) #s.append(env.sim.data.qpos[joint_interest_id])
          muscle_act.append(acti)
          m.append(action)
          if movie:
                  geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.sim.renderer.render_offscreen(width=640, height=480,camera_id=f'{view}_view')
                  frame = np.rot90(np.rot90(frame))
            # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                  frames.append(frame[::-1,:,:])
                  #env.sim.mj_render(mode='window') # GUI
          step += 1
    plt.plot(range(step),muscle_act)
    plt.show()


# evaluate policy
all_rewards = []
for _ in tqdm(range(10)): # 20 random targets
  ep_rewards = []
  done = False
  obs = env.reset()
  step = 0
  while (not done) and (step < 2000):
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
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	