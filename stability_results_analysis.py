import gym
from gym import spaces
import matplotlib.path as mplPath
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


def storeData(env, model, steps, env_name, file_name):
    body_names = ['calcn_l', 'calcn_r', 'femur_l', 'femur_r', 'patella_l', 'patella_r', 
                                            'pelvis', 'root', 'talus_l', 'talus_r', 'tibia_l', 'tibia_r', 'toes_l', 'toes_r', 'world']
    joint_names = ['ankle_angle_l', 'ankle_angle_r', 'hip_adduction_l', 'hip_adduction_r', 'hip_flexion_l', 
                                            'hip_flexion_r', 'hip_rotation_l', 'hip_rotation_r', 'knee_angle_l', 'knee_angle_r',  
                                            'hip_flexion_r', 'hip_rotation_l', 'hip_rotation_r', 'knee_angle_l', 'knee_angle_r',  
                                            'mtp_angle_l', 'mtp_angle_r', 'subtalar_angle_l', 'subtalar_angle_r']
    '''
    tendon_names = ['addbrev_l_tendon', 'addbrev_r_tendon', 'addlong_l_tendon', 'addlong_r_tendon', 'addmagDist_l_tendon', 
                    'addmagDist_r_tendon', 'addmagIsch_l_tendon', 'addmagIsch_r_tendon', 'addmagMid_l_tendon', 'addmagMid_r_tendon', 
                    'addmagProx_l_tendon', 'addmagProx_r_tendon', 'bflh_l_tendon', 'bflh_r_tendon', 'bfsh_l_tendon', 'bfsh_r_tendon', 
                    'edl_l_tendon', 'edl_r_tendon', 'ehl_l_tendon', 'ehl_r_tendon', 'fdl_l_tendon', 'fdl_r_tendon', 'fhl_l_tendon', 
                    'fhl_r_tendon', 'gaslat_l_tendon', 'gaslat_r_tendon', 'gasmed_l_tendon', 'gasmed_r_tendon', 'glmax1_l_tendon', 
                    'glmax1_r_tendon', 'glmax2_l_tendon', 'glmax2_r_tendon', 'glmax3_l_tendon', 'glmax3_r_tendon', 'glmed1_l_tendon', 
                    'glmed1_r_tendon', 'glmed2_l_tendon', 'glmed2_r_tendon', 'glmed3_l_tendon', 'glmed3_r_tendon', 'glmin1_l_tendon', 
                    'glmin1_r_tendon', 'glmin2_l_tendon', 'glmin2_r_tendon', 'glmin3_l_tendon', 'glmin3_r_tendon', 'grac_l_tendon', 
                    'grac_r_tendon', 'iliacus_l_tendon', 'iliacus_r_tendon', 'perbrev_l_tendon', 'perbrev_r_tendon', 'perlong_l_tendon', 
                    'perlong_r_tendon', 'piri_l_tendon', 'piri_r_tendon', 'psoas_l_tendon', 'psoas_r_tendon', 'recfem_l_tendon', 'recfem_r_tendon', 'sart_l_tendon', 'sart_r_tendon', 'semimem_l_tendon', 'semimem_r_tendon', 'semiten_l_tendon', 'semiten_r_tendon', 'soleus_l_tendon', 'soleus_r_tendon', 'tfl_l_tendon', 'tfl_r_tendon', 'tibant_l_tendon', 'tibant_r_tendon', 'tibpost_l_tendon', 'tibpost_r_tendon', 'vasint_l_tendon', 'vasint_r_tendon', 'vaslat_l_tendon', 'vaslat_r_tendon', 'vasmed_l_tendon', 'vasmed_r_tendon']
    '''
    actuator_names =  ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 'addmagIsch_r', 
                       'addmagMid_l', 'addmagMid_r', 'addmagProx_l', 'addmagProx_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 
                       'glmax3_r', 'glmed1_l', 'glmed1_r', 'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 
                       'glmin3_l', 'glmin3_r', 'iliacus_l', 'iliacus_r', 'psoas_l', 'psoas_r', 'piri_l', 'piri_r', 'tfl_l', 'tfl_r', 'sart_l', 'sart_r', 
                       'recfem_l', 'recfem_r', 'edl_l', 'edl_r', 'ehl_l', 'ehl_r', 'fdl_l', 'fdl_r', 'fhl_l', 'fhl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 
                        'tibpost_r', 'perbrev_l', 'perbrev_r', 'perlong_l', 'perlong_r', 'gaslat_l', 'gaslat_r', 'gasmed_l', 'gasmed_r', 'soleus_l', 'soleus_r']


    '''
    ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 
    '''
    tendon_names = ['addbrev_l_tendon', 'addbrev_r_tendon', 'addlong_l_tendon', 'addlong_r_tendon', 'addmagDist_l_tendon', 
                    'addmagDist_r_tendon', 'addmagIsch_l_tendon', 'addmagIsch_r_tendon', 'addmagMid_l_tendon', 'addmagMid_r_tendon', 
                    'addmagProx_l_tendon', 'addmagProx_r_tendon', 'bflh_l_tendon', 'bflh_r_tendon', 'bfsh_l_tendon', 'bfsh_r_tendon', 
                    'edl_l_tendon', 'edl_r_tendon', 'ehl_l_tendon', 'ehl_r_tendon', 'fdl_l_tendon', 'fdl_r_tendon', 'fhl_l_tendon', 
                    'fhl_r_tendon', 'gaslat_l_tendon', 'gaslat_r_tendon', 'gasmed_l_tendon', 'gasmed_r_tendon', 'glmax1_l_tendon', 
                    'glmax1_r_tendon', 'glmax2_l_tendon', 'glmax2_r_tendon', 'glmax3_l_tendon', 'glmax3_r_tendon', 'glmed1_l_tendon', 
                    'glmed1_r_tendon', 'glmed2_l_tendon', 'glmed2_r_tendon', 'glmed3_l_tendon', 'glmed3_r_tendon', 'glmin1_l_tendon', 
                    'glmin1_r_tendon', 'glmin2_l_tendon', 'glmin2_r_tendon', 'glmin3_l_tendon', 'glmin3_r_tendon', 'grac_l_tendon', 
                    'grac_r_tendon', 'iliacus_l_tendon', 'iliacus_r_tendon', 'perbrev_l_tendon', 'perbrev_r_tendon', 'perlong_l_tendon', 
                    'perlong_r_tendon', 'piri_l_tendon', 'piri_r_tendon', 'psoas_l_tendon', 'psoas_r_tendon', 'recfem_l_tendon', 'recfem_r_tendon', 'sart_l_tendon', 'sart_r_tendon', 'semimem_l_tendon', 'semimem_r_tendon', 'semiten_l_tendon', 'semiten_r_tendon', 'soleus_l_tendon', 'soleus_r_tendon', 'tfl_l_tendon', 'tfl_r_tendon', 'tibant_l_tendon', 'tibant_r_tendon', 'tibpost_l_tendon', 'tibpost_r_tendon', 'vasint_l_tendon', 'vasint_r_tendon', 'vaslat_l_tendon', 'vaslat_r_tendon', 'vasmed_l_tendon', 'vasmed_r_tendon']
    '''
    actuator_names =  ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 'addmagIsch_r', 
                       'addmagMid_l', 'addmagMid_r', 'addmagProx_l', 'addmagProx_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 
                       'glmax3_r', 'glmed1_l', 'glmed1_r', 'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 
                       'glmin3_l', 'glmin3_r', 'iliacus_l', 'iliacus_r', 'psoas_l', 'psoas_r', 'piri_l', 'piri_r', 'tfl_l', 'tfl_r', 'sart_l', 'sart_r', 
                       'recfem_l', 'recfem_r', 'edl_l', 'edl_r', 'ehl_l', 'ehl_r', 'fdl_l', 'fdl_r', 'fhl_l', 'fhl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 
                        'tibpost_r', 'perbrev_l', 'perbrev_r', 'perlong_l', 'perlong_r', 'gaslat_l', 'gaslat_r', 'gasmed_l', 'gasmed_r', 'soleus_l', 'soleus_r']


    '''
    ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 
                       'addmagIsch_r', 'addmagMid_l', 'addmagMid_r', 'addmagProx_l', 'addmagProx_r', 'bflh_l', 'bflh_r', 'bfsh_l', 
                       'bfsh_r', 'edl_l', 'edl_r', 'ehl_l', 'ehl_r', 'fdl_l', 'fdl_r', 'fhl_l', 'fhl_r', 'gaslat_l', 'gaslat_r', 
                       'gasmed_l', 'gasmed_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 
                       'glmed1_r', 'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 
                       'glmin3_l', 'glmin3_r', 'grac_l', 'grac_r', 'iliacus_l', 'iliacus_r', 'perbrev_l', 'perbrev_r', 'perlong_l', 
                       'perlong_r', 'piri_l', 'piri_r', 'psoas_l', 'psoas_r', 'recfem_l', 'recfem_r', 'sart_l', 'sart_r', 'semimem_l', 
                       'semimem_r', 'semiten_l', 'semiten_r', 'soleus_l', 'soleus_r', 'tfl_l', 'tfl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 
                       'tibpost_r', 'vasint_l', 'vasint_r', 'vaslat_l', 'vaslat_r', 'vasmed_l', 'vasmed_r']
    '''
    '''
    dataStore = {}
    dataStore['modelInfo'] = {}
    dataStore['modelInfo']['bodyNames'] =  body_names
    dataStore['modelInfo']['jointNames'] = joint_names
    #dataStore['modelInfo']['tendonNames'] = tendon_names
    #dataStore['modelInfo']['tendonNames'] = tendon_names
    dataStore['modelInfo']['rewardWeights'] = env.rwd_keys_wt
    dataStore['modelInfo']['trainingSteps'] = 15000000
    dataStore['modelInfo']['testSteps'] = steps
    dataStore['modelInfo']['perturbationMagnitude'] = next((x for x in env.perturbation_magnitude if x != 0), None)
    dataStore['modelInfo']['perturbationDirection'] = next((index for index, value in enumerate(env.perturbation_magnitude) if value != 0))
    dataStore['modelInfo']['perturbationTime'] = env.perturbation_time

    dataStore['modelInfo']['targetPosition'] = {}
    dataStore['modelInfo']['reachError'] = {}
    dataStore['modelInfo']['tipPosition'] = {}
    dataStore['modelInfo']['rewardDict'] = {}

    dataStore['stateInfo'] = {}

    dataStore['jointInfo'] = {}
    dataStore['jointInfo']['qpos'] = {}
    dataStore['jointInfo']['qvel'] = {}
    dataStore['jointInfo']['qtau'] = {}
    dataStore['jointInfo']['qacc'] = {}

    dataStore['jointInfo']['ROM'] = {}

    data_joint = {}
    for i, joint in enumerate(joint_names):
        data_joint[joint] = {}
        data_joint[joint]['MinRom'] = env.sim.model.jnt_range[env.sim.model.joint_name2id(
            joint), 0].copy()
        data_joint[joint]['MaxRom'] = env.sim.model.jnt_range[env.sim.model.joint_name2id(
            joint), 1].copy()
    dataStore['jointInfo']['ROM'] = data_joint

    dataStore['bodyInfo'] = {}
    dataStore['bodyInfo']['com'] = {}
    dataStore['bodyInfo']['com_v'] = {}
    dataStore['bodyInfo']['bos'] = {}
    dataStore['bodyInfo']['xipos'] = {}
    dataStore['bodyInfo']['xpos'] = {}
    dataStore['bodyInfo']['grf'] = {}
    dataStore['bodyInfo']['grf']['rToes'] = {}
    dataStore['bodyInfo']['grf']['lToes'] = {}
    dataStore['bodyInfo']['grf']['rCal'] = {}
    dataStore['bodyInfo']['grf']['lCal'] = {}

    dataStore['muscleInfo'] = {}
    dataStore['muscleInfo']['action'] = {}
    dataStore['muscleInfo']['muscleForce'] = {}
    dataStore['muscleInfo']['muscleActivation'] = {}
    dataStore['muscleInfo']['muscleLength'] = {}
    dataStore['muscleInfo']['muscleMoment'] = {}
    dataStore['muscleInfo']['muscleVelocity'] = {}

    dataStore['tensorBoard'] = {}
    dataStore['tensorBoard']['meanReward'] = {}
    dataStore['tensorBoard']['epMeanReward'] = {}

    dataStore['videoInfo'] = {}
    dataStore['videoInfo']['framesSide'] = {}
    dataStore['videoInfo']['framesFront'] = {}
    
    activation_dict = {}

    qpos_dict, qvel_dict, qtorque_dict = {}, {}, {}
    state = []
    targetPosition, reachError, tipPosition, rewardDict = [], [], [], []
    qpos, qvel, qacc, torque = [], [], [], []
    com, com_v, bos, xpos, xipos, grf_rToes, grf_lToes, grf_rCal, grf_lCal = [], [], [], [], [], [], [], [], []
    muscleAction, muscleForce, muscleActivation, muscleLength, muscleMoment, muscleVelocity = [], [], [], [], [], []
    frames_side, frames_front = [], []

    for _ in tqdm(range(dataStore['modelInfo']['testSteps'])):
        obs = env.get_obs_vec()
        action, __ = model.predict(obs, deterministic=True)

        env.sim.data.ctrl[:] = action
        obs, reward, done, info = env.step(action)

        state.append(env.env.get_env_state())
        for isite in range(len(env.tip_sids)):
            targetPosition.append(
                env.sim.data.site_xpos[env.target_sids[isite]].copy())
            tipPosition.append(env.sim.data.site_xpos[env.tip_sids[isite]].copy())
            reachError.append(np.array(env.sim.data.site_xpos[env.target_sids[isite]].copy(
            )) - np.array(env.sim.data.site_xpos[env.tip_sids[isite]].copy()))

        rewardDict.append(env.get_reward_dict(env.get_obs_dict(env.sim)).copy())
        # Joint Info
        for joint in joint_names:
            if _ == 0:
                qpos_dict[joint], qvel_dict[joint], qtorque_dict[joint] = {}, {}, {}
            
            qpos_dict[joint][_] = env.sim.data.joint(joint).qpos.copy()#env.sim.data.get_jnt_qpos(joint).copy()
            qvel_dict[joint][_] = env.sim.data.joint(joint).qvel.copy()#env.sim.data.get_joint_qvel(joint).copy()
            qtorque_dict[joint][_] = env.sim.data.joint(joint).qfrc_actuator.copy()
            #env.sim.data.joint(joint).qfrc_smooth.copy() + env.sim.data.joint(joint).qfrc_constraint.copy()
        #print(qtorque_dict['hip_flexion_l'][_])
        #print('actuator',env.sim.data.joint('hip_flexion_l').qfrc_actuator.copy())
        qpos_dict['hip_flexion_l'][_] =-env.obs_dict['pelvis_rot'][0]+env.sim.data.joint('hip_flexion_l').qpos.copy()
        qpos_dict['hip_flexion_r'][_]= -env.obs_dict['pelvis_rot'][0]+env.sim.data.joint('hip_flexion_r').qpos.copy()   
        qacc.append(env.sim.data.qacc.copy())

        # Body Info
        x, y = np.array([]), np.array([])
        for label in ['calcn_r', 'calcn_l', 'toes_r', 'toes_l']:
            x_and_y = np.array(env.sim.data.xipos[env.sim.model.body_name2id(label)].copy())[
                :2]  # select x and y position of the current body
            x = np.append(x, x_and_y[0])
            y = np.append(y, x_and_y[1])

        # CoM is considered to be the center of mass of the pelvis (for now)
        pos = env.sim.data.xipos.copy()
        mass = env.sim.model.body_mass
        com1 = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        vel = env.sim.data.cvel.copy()
        com_v_int = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        com_v.append(com_v_int[-3:].copy())
        com.append(com1[:2].copy())
        bos.append(np.append(x, y))
        xpos.append(env.sim.data.body_xpos.copy())
        xipos.append(env.sim.data.xipos.copy())

        grf_rToes.append(env.sim.data.sensordata[env.sim.model.sensor_name2id('r_toes')].copy())
        grf_lToes.append(env.sim.data.sensordata[env.sim.model.sensor_name2id('l_toes')].copy())
        grf_rCal.append(env.sim.data.sensordata[env.sim.model.sensor_name2id('r_foot')].copy())
        grf_lCal.append(env.sim.data.sensordata[env.sim.model.sensor_name2id('l_foot')].copy())


        #print(len(env.sim.data.act[0]))
        # Muscle Info
        for actuator in actuator_names:
            if _ == 0:
                activation_dict[actuator]= {}
            #print(env.sim.data.act[env.sim.model.actuator_name2id(actuator)].copy())
            activation_dict[actuator][_] = env.sim.data.act[env.sim.model.actuator_name2id(actuator)].copy()#env.sim.data.get_jnt_qpos(joint).copy()


        
        muscleAction.append(action.copy())
        muscleForce.append(env.sim.data.actuator_force.copy())
        muscleLength.append(env.sim.data.actuator_length.copy())
        muscleMoment.append(env.sim.data.actuator_moment.copy())
        muscleVelocity.append(env.sim.data.actuator_velocity.copy())

        geom_1_indices = np.where(env.sim.model.geom_group == 1)
        env.sim.model.geom_rgba[geom_1_indices, 3] = 0
        frame_front = env.sim.renderer.render_offscreen(width=640, height=480,camera_id='front_view')
        frame_front = np.rot90(np.rot90(frame_front))
        #frames_front.append(frame_front[::-1, :, :])
        '''

    tb_logdir = f"./standingBalance/temp_env_tensorboard/" + env_name + '/' + env_name + "_" + file_name + "_1"
        #frames_front.append(frame_front[::-1, :, :])
        '''

    tb_logdir = f"./standingBalance/temp_env_tensorboard/" + env_name + '/' + env_name + "_" + file_name + "_1"
    event_accumulator = EventAccumulator(tb_logdir)
    event_accumulator.Reload()
    events = event_accumulator.Scalars('rollout/ep_rew_mean')
    ep_rew_mean = [x.value for x in events]
    events = event_accumulator.Scalars('eval/mean_reward')
    mean_reward = [x.value for x in events]

    dataStore['modelInfo']['targetPosition'] = targetPosition
    dataStore['modelInfo']['rewardDict'] = rewardDict
    dataStore['modelInfo']['tipPosition'] = tipPosition
    dataStore['modelInfo']['reachError'] = reachError

    dataStore['stateInfo'] = state

    for joint in joint_names:
        qpos, qvel, qtau = [], [], []
        hip_fle_l, hip_fle_r = [], []
        hip_fle_l, hip_fle_r = [], []
        for _ in range(steps):
            qpos.append(qpos_dict[joint][_])
            qvel.append(qvel_dict[joint][_])
            qtau.append(qtorque_dict[joint][_])
        dataStore['jointInfo']['qpos'][joint] = qpos
        dataStore['jointInfo']['qvel'][joint] = qvel
        dataStore['jointInfo']['qtau'][joint] = qtau
    #print('knee',dataStore['jointInfo']['qpos']['knee_angle_l'])
    #print(dataStore['jointInfo']['qpos']['hip_flexion_l'])
    # dataStore['jointInfo']['qpos'] = qpos
    # dataStore['jointInfo']['qvel'] = qvel
    dataStore['jointInfo']['qacc'] = qacc
    dataStore['jointInfo']['torque'] = torque

    for actuator in actuator_names:
        muscleActivation = []
        for _ in range(steps):
            muscleActivation.append(activation_dict[actuator][_])
        dataStore['muscleInfo']['muscleActivation'][actuator] = muscleActivation

    dataStore['bodyInfo']['com'] = com
    dataStore['bodyInfo']['com_v'] = com_v
    dataStore['bodyInfo']['height'] = com1[-1]
    dataStore['bodyInfo']['bos'] = bos
    dataStore['bodyInfo']['xpos'] = xpos
    dataStore['bodyInfo']['xipos'] = xipos
    dataStore['bodyInfo']['grf']['rToes'] = grf_rToes
    dataStore['bodyInfo']['grf']['lToes'] = grf_lToes
    dataStore['bodyInfo']['grf']['rCal'] = grf_rCal
    dataStore['bodyInfo']['grf']['lCal'] = grf_lCal

    dataStore['muscleInfo']['action'] = muscleAction
    dataStore['muscleInfo']['muscleForce'] = muscleForce
    #dataStore['muscleInfo']['muscleActivation'] = muscleActivation
    dataStore['muscleInfo']['muscleLength'] = muscleLength
    dataStore['muscleInfo']['muscleMoment'] = muscleMoment
    dataStore['muscleInfo']['muscleVelocity'] = muscleVelocity

    dataStore['tensorBoard']['epMeanReward'] = ep_rew_mean
    dataStore['tensorBoard']['meanReward'] = mean_reward

    dataStore['videos'] = frames_front
    
    bos_final = dataStore['bodyInfo']['bos'][-1].reshape(2, 4)
    bos_final = mplPath.Path(bos_final.T)
    within = bos_final.contains_point(dataStore['bodyInfo']['com'][0])
    if within:
        dataStore['modelInfo']['stability_state'] = True
    else:
        dataStore['modelInfo']['stability_state'] = False

    return dataStore

nb_seed = 1
# Making automatic True will run the code for the last policy that was generated
# If automatic is True, it will loop through all files in the policy_best_model folder. 
automatic = False
selected = True # If selected = true, make sure you write the file you want :)
#selected_file = ['2024_02_17_18_56_59'] 
# Which output do you want? 
movie = False
img = False
pdf = False
sarco = False
fatigue = False

if sarco:
    selected_file = ['2024_02_17_18_56_59'] 
    env_name = 'myoSarcLegReachFixed-v3'
    dir_path = './standingBalance-sarco/policy_best_model/'+ 'myoSarcLegReachFixed-v3' +'/'
    all_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
elif fatigue:
    selected_file = ['2024_02_20_15_03_01'] 
    env_name = 'myoFatiLegReachFixed-v4'
    dir_path = './standingBalance-Fatigue/policy_best_model/'+ 'myoFatiLegReachFixed-v4' +'/'
    all_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
else:
    selected_file = ['2024_02_17_20_19_05'] 
    env_name = 'myoLegReachFixed-v2'
    dir_path = './standingBalance/policy_best_model/'+ env_name +'/'
    all_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

if automatic:
    # sort the directories by creation time, newest first
    sorted_dirs = sorted(all_dirs, key=lambda d: os.path.getctime(os.path.join(dir_path, d)), reverse=True)
    # get the name of the latest created directory
    latest_dir_name = sorted_dirs[0]
    # create the file name using the format "year_month_day_hour_minute_second"
    last_file = "{}".format(latest_dir_name)
    name_list = [last_file]
else: 
    if selected: name_list = selected_file
    else: name_list = all_dirs 
policy_name = name_list[0]
print(policy_name)

policy_path = dir_path + '/' + policy_name + '/best_model'

model = PPO.load(policy_path)
env = gym.make(f'mj_envs.robohive.envs.myo:{env_name}') 
env.reset()
random.seed() 
steps = 1000
obs = env.get_obs_vec()
obs_dict = env.get_obs_dict(env.sim)

for key in obs_dict.keys():print(f'{key} {len(obs_dict[key])}')

pkl_path = './output/PKL/' + env_name + '/'

os.makedirs(pkl_path, exist_ok=True)

for ep in range(100):
    print(f'### EPISODE {ep} ###')
    env = gym.make(env_name) 
    env.reset()
    random.seed()
    data = {}
    data = storeData(env, model, steps, env_name, policy_name)
    
    #frames_front = data['videos']
    #data['videos'] = {}
    #frames_front = data['videos']
    #data['videos'] = {}
    #frames_front = data['videos']
    #data['videos'] = {}

    #print('Making movie')
    #video_path = './output/videos/' + env_name  + '/' +  policy_name + '/'
    #os.makedirs(video_path, exist_ok=True)
    #video_path = './output/videos/' + env_name  + '/' +  policy_name + '/'
    #os.makedirs(video_path, exist_ok=True)

    # make a local copy skvideo.io.vwrite(video_path + file_name + '_side' + '.mp4', np.asarray(frames_side), inputdict = {'-r':'100'}, outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite(video_path +  policy_name + '_' + str(ep)  + '.mp4', np.asarray(frames_front), inputdict = {'-r':'100'}, outputdict={"-pix_fmt": "yuv420p"})

    with open(pkl_path + policy_name + '_lp_' + str(ep) + '.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')
    