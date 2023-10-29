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
from stable_baselines import results_plotter
import matplotlib.pyplot as plt

from datetime import datetime
import torch
import time


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

time_steps = 1000000
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "MyoLeg")
plt.show()