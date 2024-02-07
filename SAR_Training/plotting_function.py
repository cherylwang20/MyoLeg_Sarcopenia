from plotting_results import *
import math


##plot 1, perturbation AP-ML plot
plt. figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
for item in AP_pert_1:
    bos_final = item['bodyInfo']['bos'][-1].reshape(2, 4)
    bos_final = mplPath.Path(bos_final.T)
    within = bos_final.contains_point(item['bodyInfo']['com'][0])
    if within:
        color = 'red'
    else:
        print('AP, False')
        color = 'black'
    plt.scatter(item['modelInfo']['perturbationTime']*100, item['modelInfo']['perturbationMagnitude'], s =35,
                color = color)

plt.xlabel('Perturbation Step')
plt.title('AP Perturbation')
plt.ylabel('Perturbation Magnitude (N)')
#plt.xlim([100, 200])
plt.ylim([0, 50])

plt.subplot(1, 2, 2)
for item in ML_pert_1:
    bos_final = item['bodyInfo']['bos'][-1].reshape(2, 4)
    bos_final = mplPath.Path(bos_final.T)
    within = bos_final.contains_point(item['bodyInfo']['com'][0])
    if within:
        color = 'red'
    else:
        print('ML, False')
        color = 'black'
    plt.scatter(x = item['modelInfo']['perturbationTime']*100, y = item['modelInfo']['perturbationMagnitude'], s = 35,color=color) 

plt.title('ML Perturbation')
#plt.xlim([100, 200])
plt.ylim([0, 50])
plt.xlabel('Perturbation Step')
#plt.ylabel('Perturbation Magnitude (N)')

legend_elements = [Line2D([0], [0], marker='o', color='r', label='Stand',
                          markerfacecolor='r', markersize=6),
                   Line2D([0], [0], marker='o', color='black', label='Fall',
                          markerfacecolor='black', markersize=6)
                   ]

plt.legend(handles=legend_elements, bbox_to_anchor=(0.45, -0.3), loc="lower right", ncol = 2)
plt.savefig(image_path+ '/Perturbation_Strength.png' )
plt.show()
plt.close()

#plot the com variation
plt. figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
bins = [0, 10, 20, 30, 40, 50]

# Get the 'YlGn' colormap
colormap = plt.cm.plasma
norm = mcolors.BoundaryNorm(bins, colormap.N)


for i in range(len(pert_AP)):
    mean_pert = []
    pertAP_1_com, pertAP_2_com, pertAP_3_com, pertAP_4_com, pertAP_5_com = [], [], [], [], []
    pertAP_com = [pertAP_1_com, pertAP_2_com, pertAP_3_com, pertAP_4_com, pertAP_5_com]
    for j in pert_AP[i]:
        mean_pert.append(j["modelInfo"]["perturbationMagnitude"])
        pertAP_com[i].append(j["bodyInfo"]["com"])
    if not pertAP_com[i]:
        continue

    plt.plot((np.mean(pertAP_com[i], axis = 0)[:,0] + 0.03)*100, (np.mean(pertAP_com[i], axis = 0)[:,1]-0.07)*100, c=colormap(norm(np.mean(mean_pert))), alpha = 0.8, linewidth = 3)#, cmap=cmap, norm=norm)

plt.xlabel('X Position (cm)')
plt.title('AP Perturbation')
plt.ylabel('Y Position (cm)')
plt.ylim([-5, 7.5])
plt.xlim([-10, 10])

plt.subplot(1, 2, 2)
for i in range(len(pert_ML)):
    mean_pert = []
    pertML_1_com, pertML_2_com, pertML_3_com, pertML_4_com, pertML_5_com = [], [], [], [], []
    pertML_com = [pertML_1_com, pertML_2_com, pertML_3_com, pertML_4_com, pertML_5_com]
    for j in pert_ML[i]:
        mean_pert.append(j["modelInfo"]["perturbationMagnitude"])
        pertML_com[i].append(j["bodyInfo"]["com"])
    if not pertML_com[i]:
        continue

    plt.plot((np.mean(pertML_com[i], axis = 0)[:,0]+ 0.03)*100, (np.mean(pertML_com[i], axis = 0)[:,1] - 0.07)*100, c=colormap(norm(np.mean(mean_pert))), alpha = 0.8, linewidth = 3)

plt.xlabel('X Position (cm)')
plt.title('ML Perturbation')

sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=bins)
cbar.set_label('Perturbation Magnitude (N)')

plt.ylim([-5, 7.5])
plt.xlim([-10, 10])
plt.savefig(image_path+ '/com_location.png' )
plt.show()
plt.close()


#####plotting the joint angle over time for standing up case

ankle_angle_l, ankle_angle_r, hip_adduction_l, hip_adduction_r, hip_rotation_l, hip_rotation_r = [], [], [], [], [], []
hip_flexion_l, hip_flexion_r, knee_angle_l,  knee_angle_r, mtp_angle_l, mtp_angle_r, subtalar_angle_l, subtalar_angle_r = [], [], [], [], [], [], [], []

joint_name_qpos = [ankle_angle_l, ankle_angle_r, hip_adduction_l, hip_adduction_r, hip_rotation_l, 
                    hip_rotation_r, knee_angle_l,  knee_angle_r, mtp_angle_l, mtp_angle_r, hip_flexion_l, hip_flexion_r, subtalar_angle_l, subtalar_angle_r ]

ankle_angle_l_t, ankle_angle_r_t, hip_adduction_l_t, hip_adduction_r_t, hip_rotation_l_t = [], [], [], [], []
hip_rotation_r_t, knee_angle_l_t, knee_angle_r_t, mtp_angle_l_t, mtp_angle_r_t =[], [], [], [], []
hip_flexion_l_t, hip_flexion_r_t, subtalar_angle_l_t, subtalar_angle_r_t = [], [], [], []

joint_name_torque = [ankle_angle_l_t, ankle_angle_r_t, hip_adduction_l_t, hip_adduction_r_t, hip_rotation_l_t, 
                     hip_rotation_r_t, knee_angle_l_t, knee_angle_r_t, mtp_angle_l_t, mtp_angle_r_t, hip_flexion_l_t, 
                     hip_flexion_r_t, subtalar_angle_l_t, subtalar_angle_r_t]


joint_names_l = ['ankle_angle_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'mtp_angle_l',  'hip_flexion_l', 'subtalar_angle_l']
title = ['Ankle', 'Hip Adduction', 'Hip Rotation', 'Knee Angle', 'MTP Angle', 'Hip Flexion', 'Subtalar Angle']
joint_names_r = ['ankle_angle_r', 'hip_adduction_r','hip_rotation_r', 'knee_angle_r','mtp_angle_r','hip_flexion_r',  'subtalar_angle_r' ]


'''
for i in stand:
    for j in range(len(joint_name_qpos)):
        joint_name_qpos[j].append(data['jointInfo']['qpos'][joint_names_l[j]])
        joint_name_torque[j].append(data['jointInfo']['qtau'][joint_names_l[j]])
'''
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":14}) 

y_range_joint = [-50, 50]

colormap = plt.cm.plasma
norm = mcolors.BoundaryNorm(bins, colormap.N)
fig = plt.figure(figsize=(20, 12))
for i in range(0, 7):
    plt.subplot(4, 7, i+1)
    for j in range(len(pert_AP)):
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        mean_pert = []
        for k in pert_AP[j]:
            l[j].append(k['jointInfo']['qpos'][joint_names_l[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)*180/np.pi
        std = np.std(l[j], axis = 0)*180/np.pi
        plt.plot(np.mean(l[j], axis= 0)*180/np.pi, c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean -2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    plt.title(title[i])
    if i == 0:
        plt.ylabel('Left Joint Angle (deg)')
######

for i in range(0, 7):
    plt.subplot(4, 7, i+8)
    for j in range(len(pert_AP)):
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        mean_pert = []
        for k in pert_AP[j]:
            l[j].append(k['jointInfo']['qpos'][joint_names_r[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)*180/np.pi
        std = np.std(l[j], axis = 0)*180/np.pi
        plt.plot(np.mean(l[j], axis= 0)*180/np.pi, c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean -2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    if i == 0:
        plt.ylabel('Right Joint Angle (deg)')

for i in range(0, 7):
    plt.subplot(4, 7, i+15)
    for j in range(len(pert_AP)):
        mean_pert = []
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        for k in pert_AP[j]:
            l[j].append(k['jointInfo']['qtau'][joint_names_l[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)
        std = np.std(l[j], axis = 0)
        plt.plot(np.mean(l[j], axis= 0), c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean -2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    if i == 0:
        plt.ylabel('Left Joint Torque')


for i in range(0, 7):
    plt.subplot(4, 7, i+22)
    for j in range(len(pert_AP)):
        mean_pert = []
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        for k in pert_AP[j]:
            l[j].append(k['jointInfo']['qtau'][joint_names_r[i]])
            #print(k['jointInfo']['qtau']['subtalar_angle_r'])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)
        std = np.std(l[j], axis = 0)
        plt.plot(np.mean(l[j], axis= 0), c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean - 2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    if i == 0:
        plt.ylabel('Right Joint Torque')

sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm) 
sm.set_array([]) 

plt.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 50, 6), cax=cbar_ax) 
cbar.set_label('Perturbation Magnitude (N)')

plt.savefig(image_path+ '/joint_info_AP.png' )
plt.show()
plt.close()


###ML direction
norm = mcolors.BoundaryNorm(bins, colormap.N)
fig = plt.figure(figsize=(20, 12))
for i in range(0, 7):
    plt.subplot(4, 7, i+1)
    for j in range(len(pert_ML)):
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        mean_pert = []
        for k in pert_ML[j]:
            l[j].append(k['jointInfo']['qpos'][joint_names_l[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)*180/np.pi
        std = np.std(l[j], axis = 0)*180/np.pi
        plt.plot(np.mean(l[j], axis= 0)*180/np.pi, c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean -2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    plt.title(title[i])
    if i == 0:
        plt.ylabel('Left Joint Angle (deg)')
######

for i in range(0, 7):
    plt.subplot(4, 7, i+8)
    for j in range(len(pert_ML)):
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        mean_pert = []
        for k in pert_ML[j]:
            l[j].append(k['jointInfo']['qpos'][joint_names_r[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)*180/np.pi
        std = np.std(l[j], axis = 0)*180/np.pi
        plt.plot(np.mean(l[j], axis= 0)*180/np.pi, c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean -2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    if i == 0:
        plt.ylabel('Right Joint Angle (deg)')

for i in range(0, 7):
    plt.subplot(4, 7, i+15)
    for j in range(len(pert_ML)):
        mean_pert = []
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        for k in pert_ML[j]:
            l[j].append(k['jointInfo']['qtau'][joint_names_l[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)
        std = np.std(l[j], axis = 0)
        plt.plot(np.mean(l[j], axis= 0), c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean -2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    if i == 0:
        plt.ylabel('Left Joint Torque')


for i in range(0, 7):
    plt.subplot(4, 7, i+22)
    for j in range(len(pert_ML)):
        mean_pert = []
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l = [l1, l2, l3, l4, l5]
        for k in pert_ML[j]:
            l[j].append(k['jointInfo']['qtau'][joint_names_r[i]])
            #print(k['jointInfo']['qtau']['subtalar_angle_r'])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)
        #print(mean)
        std = np.std(l[j], axis = 0)
        plt.plot(np.mean(l[j], axis= 0), c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean - 2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    if i == 0:
        plt.ylabel('Right Joint Torque')

sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm) 
sm.set_array([]) 

plt.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 50, 6), cax=cbar_ax) 
cbar.set_label('Perturbation Magnitude (N)')

plt.savefig(image_path+ '/joint_info_ML.png' )
plt.show()
plt.close()

######draw muscle activation diagram


actuator_names =  ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 
                       'addmagIsch_r', 'addmagMid_l', 'addmagMid_r', 'addmagProx_l', 'addmagProx_r', 'bflh_l', 'bflh_r', 'bfsh_l', 
                       'bfsh_r', 'edl_l', 'edl_r', 'ehl_l', 'ehl_r', 'fdl_l', 'fdl_r', 'fhl_l', 'fhl_r', 'gaslat_l', 'gaslat_r', 
                       'gasmed_l', 'gasmed_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 
                       'glmed1_r', 'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 
                       'glmin3_l', 'glmin3_r', 'grac_l', 'grac_r', 'iliacus_l', 'iliacus_r', 'perbrev_l', 'perbrev_r', 'perlong_l', 
                       'perlong_r', 'piri_l', 'piri_r', 'psoas_l', 'psoas_r', 'recfem_l', 'recfem_r', 'sart_l', 'sart_r', 'semimem_l', 
                       'semimem_r', 'semiten_l', 'semiten_r', 'soleus_l', 'soleus_r', 'tfl_l', 'tfl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 
                       'tibpost_r', 'vasint_l', 'vasint_r', 'vaslat_l', 'vaslat_r', 'vasmed_l', 'vasmed_r']

active_actuator = []
for names in actuator_names:
    act_all = []
    for k in AP_pert:
        act = k['muscleInfo']['muscleActivation'][names]
        #print(len(act))
        act_all.append(act)
        #print(np.mean(act_all, axis=0))
    if np.max(np.mean(act_all, axis = 0)) > 0.2:
        active_actuator.append(names)



n1 = len(active_actuator)//4 + 1

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times New Roman"],  # specify font here
    "font.size":20}) 

#titleMuscle = ['addbrev_l', 'addbrev_r', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_r', 'addmagMid_l', 'addmagProx_l', 'addmagProx_r', 'bflh_l', 'bflh_r', 'bfsh_l', 'bfsh_r', 'edl_l', 'edl_r', 'ehl_l', 'fdl_r', 'fhl_r', 'gaslat_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 'glmed1_r', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin2_l', 'glmin3_l', 'glmin3_r', 'grac_l', 'iliacus_l', 'perbrev_l', 'perlong_l', 'perlong_r', 'piri_l', 'piri_r', 'psoas_l', 'recfem_l', 'recfem_r', 'sart_r', 'semiten_l', 'semiten_r', 'tfl_l', 'tfl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 'tibpost_r', 'vasint_r', 'vaslat_l', 'vaslat_r', 'vasmed_l', 'vasmed_r']

fig = plt.figure(figsize=(25, 25), constrained_layout = True)
plt.subplots_adjust(hspace=0.8)
for i in range(0, len(active_actuator)):
    plt.subplot(n1, 7, i+1)
    l1, l2, l3, l4, l5 = [], [], [], [], []
    l = [l1, l2, l3, l4, l5]
    for j in range(len(pert_AP)):
        mean_pert = []
        for k in pert_AP[j]:
            l[j].append(k['muscleInfo']['muscleActivation'][active_actuator[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)
        std = np.std(l[j], axis = 0)
        plt.title(active_actuator[i], usetex =False)
        plt.plot(np.mean(l[j], axis= 0), c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean - 2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    #print(active_actuator)
    plt.ylabel('Activation')
    plt.ylim([0, 1.1])

plt.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.5, 0.01, 0.4])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 50, 6), cax=cbar_ax) 
cbar.set_label('Perturbation Magnitude (N)')
plt.savefig(image_path + '/muscle_activation_AP.png')
plt.show()
plt.close()


## muscle activation in ML perturbation situation
active_actuator = []
for names in actuator_names:
    act_all = []
    for k in ML_pert:
        act = k['muscleInfo']['muscleActivation'][names]
        #print(len(act))
        act_all.append(act)
        #print(np.mean(act_all, axis=0))
    if np.max(np.mean(act_all, axis = 0)) > 0.2:
        active_actuator.append(names)

n1 = len(active_actuator)//4 + 1

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times New Roman"],  # specify font here
    "font.size":20}) 

fig = plt.figure(figsize=(25, 25), constrained_layout = True)
plt.subplots_adjust(hspace=0.8)
for i in range(0, len(active_actuator)):
    plt.subplot(n1, 7, i+1)
    l1, l2, l3, l4, l5 = [], [], [], [], []
    l = [l1, l2, l3, l4, l5]
    for j in range(len(pert_ML)):
        mean_pert = []
        for k in pert_ML[j]:
            l[j].append(k['muscleInfo']['muscleActivation'][active_actuator[i]])
            mean_pert.append(k["modelInfo"]["perturbationMagnitude"])
        mean = np.mean(l[j], axis = 0)
        std = np.std(l[j], axis = 0)
        plt.title(active_actuator[i], usetex =False)
        plt.plot(np.mean(l[j], axis= 0), c = colormap(norm(np.mean(mean_pert))), alpha = 0.7, linewidth = 2)
        plt.fill_between(range(steps), mean - 2*std, mean + 2*std, facecolor = colormap(norm(np.mean(mean_pert))), alpha = 0.3)
    #print(active_actuator)
    plt.ylabel('Activation')
    plt.ylim([0, 1.1])

plt.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.5, 0.01, 0.4])
cbar = plt.colorbar(sm, ticks=np.linspace(0, 50, 6), cax=cbar_ax) 
cbar.set_label('Perturbation Magnitude (N)')
plt.savefig(image_path + '/muscle_activation_ML.png')
plt.show()
plt.close()
