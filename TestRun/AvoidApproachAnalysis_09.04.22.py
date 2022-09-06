import numpy as np
import os
import pandas as pd
import pickle
from matplotlib import pyplot as plt

df = pd.read_pickle('AvoidApproach_TestData_09.04.22.pkl')

pose_dict = {}
A_factor_dict={}
v_dict = {}
omega_dict={}
onset_dict = {}
iid_dict={}
avoid_term_dict={}
approach_term_dict={}
for a in np.arange(-30,35,5):
    tag = str(a)
    pose_dict[tag]=[]
    A_factor_dict[tag]=[]
    v_dict[tag]=[]
    omega_dict[tag]=[]
    onset_dict[tag]=[]
    iid_dict[tag]=[]
    avoid_term_dict[tag]=[]
    approach_term_dict[tag]=[]
    
for i in range(len(df)):
    tag = str(int(np.round(df['angles'][i])))
    pose_dict[tag].append(df['poses'][i])
    A_factor_dict[tag].append(df['A_factors'][i])
    v_dict[tag].append(df['v'][i])
    omega_dict[tag].append(df['omega'][i])
    onset_dict[tag].append(df['onsets'][i])
    iid_dict[tag].append(df['iid'][i])
    avoid_term_dict[tag].append(df['avoid_term'][i])
    approach_term_dict[tag].append(df['approach_term'][i])

angle_tag = str(-20)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter([0],[0], c='r')
ax.scatter([-5], [0], c='g')
for poses, A in zip(pose_dict[angle_tag],A_factor_dict[angle_tag]):
    if A>-1:
        ax.plot(poses[:,0], poses[:,1], alpha=0.7, label=np.round(A,2))
ax.legend()
#ax.set_ylim([-5,1])
#ax.set_xlim([-6,2])
ax.set_aspect('equal')
plt.show()