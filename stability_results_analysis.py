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
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

x = [1, 1, -1, -1]
y = [1, -1, -1, 1]

print(PolyArea(x, y))
areaofbase = Polygon(zip(x, y)).area
print(areaofbase)