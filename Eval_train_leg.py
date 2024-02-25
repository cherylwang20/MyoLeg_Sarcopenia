import gym
import numpy as np
from stable_baselines3 import PPO
import skvideo
import skvideo.io
import os
import random
from tqdm.auto import tqdm

nb_seed = 1

sarco = False
step = False
movie = True
path = './'

model_num = '2024_02_24_17_12_44'
env_name = 'myoLegReachFixed-v5'
#model = PPO.load(r"C:/Users/chery/Documents/MyoLeg_Sarcopenia/StepBalance/policy_best_model/SAR/myoLegReachFixed-v2/" 
                   #+ model_num + '/best_model')
model = PPO.load(path+'/standingBalance/policy_best_model'+ '/'+ env_name + '/' + model_num +
                 r'/best_model')
#model = PPO.load('ep_train_results')
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
    for _ in tqdm(range(700)):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          #obs = env.get_obs_dict()
          
          action, _ = model.predict(obs, deterministic=True)
          #env.sim.data.ctrl[:] = action
          obs, reward, done, info = env.step(action)
          #t.append(env.obs_dict['reach_err']) #s.append(env.sim.data.qpos[joint_interest_id])
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


# evaluate policy
all_rewards = []
for _ in tqdm(range(10)): # 20 random targets
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
      skvideo.io.vwrite(path+'/videos/sarco'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
    else:
      os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
      skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	
