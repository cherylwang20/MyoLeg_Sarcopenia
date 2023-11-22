import gym
from gym import spaces
from tqdm import tqdm
import mujoco_py
import mj_envs
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.reservoir import Reservoir
import matplotlib.pyplot as plt
import skvideo.io
import os
import pickle
import random
from matplotlib.backends.backend_pdf import PdfPages 

from shapely.geometry import Polygon
import matplotlib.pyplot as plt

x1 = [1, 1, -1, -1, 1]
y1 = [1, -1, -1, 1, 1]
areaofbase1 = Polygon(zip(x1, y1)).area

x2 = [2, 2, -1, -1, 2]
y2 = [-1, -3, 1, 3, -1]
areaofbase2= Polygon(zip(x2, y2)).area


print(areaofbase1, areaofbase2)

plt.figure()
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.show()