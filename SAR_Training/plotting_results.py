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
'''
for i in range(151):
    name = f'2023_11_27_13_21_48_{i}'
    with open(pkl_path+name+'.pkl', 'rb') as f:
        data = pickle.load(f)
'''

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Assuming your files are named in a sequential pattern


name = '2023_12_04_18_29_11'

image_path = './image/' + env_name + '/' + name
os.makedirs(image_path, exist_ok=True)

AP_pert = []
AP_pert_1, ML_pert_1= [], []#this is only used for the scatter stand/fall plot
ML_pert = []
stand = []
fall = []
pert_1_AP, pert_2_AP, pert_3_AP, pert_4_AP, pert_5_AP = [], [], [], [], []
pert_1_ML, pert_2_ML, pert_3_ML, pert_4_ML, pert_5_ML = [], [], [], [], []
for i in range(ep):
    file_path = pkl_path + name+f'_{i}.pkl'  # Replace with actual file paths
    data = load_pickle(file_path)
    bos_final = data['bodyInfo']['bos'][-1].reshape(2, 4)
    bos_final = mplPath.Path(bos_final.T)
    within = bos_final.contains_point(data['bodyInfo']['com'][0])
    if within:
        stand.append(data.copy())
    else:
        fall.append(data.copy())
    
    if data['modelInfo']['perturbationDirection'] == 1:
        AP_pert_1.append(data)
    else:
        ML_pert_1.append(data)

for data in stand:
    if data['modelInfo']['perturbationDirection'] == 1:
        AP_pert.append(data)
        if 0 <= data['modelInfo']['perturbationMagnitude'] <= 10:
            pert_1_AP.append(data)
        elif 10 <= data['modelInfo']['perturbationMagnitude'] <= 20:
            pert_2_AP.append(data)
        elif 20 <= data['modelInfo']['perturbationMagnitude'] <= 30:
            pert_3_AP.append(data)
        elif 30 <= data['modelInfo']['perturbationMagnitude'] <= 40:
            pert_4_AP.append(data)
        else:
            pert_5_AP.append(data)
    else:
        ML_pert.append(data)
        if 0 <= data['modelInfo']['perturbationMagnitude'] <= 10:
            pert_1_ML.append(data)
        elif 10 <= data['modelInfo']['perturbationMagnitude'] <= 20:
            pert_2_ML.append(data)
        elif 20 <= data['modelInfo']['perturbationMagnitude'] <= 30:
            pert_3_ML.append(data)
        elif 30 <= data['modelInfo']['perturbationMagnitude'] <= 40:
            pert_4_ML.append(data)
        else:
            pert_5_ML.append(data)

    
pert_AP = [pert_1_AP, pert_2_AP, pert_3_AP, pert_4_AP, pert_5_AP]
pert_ML = [pert_1_ML, pert_2_ML, pert_3_ML, pert_4_ML, pert_5_ML]