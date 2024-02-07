import pickle
import numpy as np
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import scienceplots
import os
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

plt.style.use(['science'])

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":15}) 

sarco =  True
if sarco:
    env_name = 'myoSarcLegReachFixed-v3'
else:
    env_name = 'myoLegReachFixed-v2'
steps = 1000
pkl_path = './output/PKL/' + env_name + '/'
ep = 200

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Assuming your files are named in a sequential pattern


name = '2023_12_04_18_29_11'

image_path = './image/' + env_name + '/' + name
os.makedirs(image_path, exist_ok=True)


for i in range(ep):
    file_path = pkl_path + name+f'_{i}.pkl'  # Replace with actual file paths
    data = load_pickle(file_path)
    if 30 <= data["modelInfo"]["perturbationMagnitude"] <= 40 and data['modelInfo']['perturbationDirection'] == 0:
        print(file_path, data["modelInfo"]["perturbationMagnitude"])