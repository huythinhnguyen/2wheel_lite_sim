import numpy as np
import os
import sys
import time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt

from Simulation.Motion import State, Drive
from Control.SensorimotorLoops.BatEcho import AvoidApproach
from Sensors.BatEcho.Spatializer import Render

from Sensors.BatEcho import Setting as sensorconfig
from Control.SensorimotorLoops import Setting as controlconfig

import pandas as pd

DATE = '09.04.22b'
HIT_DISTANCE = 0.3


if __name__=='__main__':
    objects = np.asarray([2.,0.,1]).reshape(-1,3)
    pose_angles = np.arange(0, 3.1, 1.)*(np.pi/18)
    approach_factors = np.arange(0, 1.1, 0.25)
    episode=0

    episodes, A_factors, poses, vs, omegas=[],[],[],[],[]
    onsets, iids = [],[]
    approach_terms, avoid_terms = [], []
    angles = []

    for p_angle in pose_angles:
        pose = np.asarray([0., 0., p_angle])
        for A in approach_factors:
            episodes.append(episode)
            episode += 1
            A_factors.append(A)
            angles.append(np.degrees(p_angle))
            bat = State(pose=pose, dt=1/controlconfig.CHIRP_RATE)
            render = Render()
            controller = AvoidApproach(approach_factor = A)
            controller.kine_cache['v'] = controlconfig.MAX_LINEAR_VELOCITY
            pose_rec = np.array([]).reshape(0,3)
            v_rec, w_rec, onset_rec, iid_rec = [], [], [], []
            approach_rec, avoid_rec = [],[]

            for _ in range(500):
                pose_rec = np.vstack((pose_rec, bat.pose))
                echoes = render.run(bat.pose, objects)
                v, omega = controller.get_kinematic(echoes)
                bat.update_kinematic(kinematic=[v, omega])
                bat.update_pose()
                v_rec.append(v)
                w_rec.append(omega)
                onset_rec.append(controller.cache['onset_distance'])
                iid_rec.append(controller.cache['IID'])
                approach_rec.append(controller.cache['approach_term'])
                avoid_rec.append(controller.cache['avoid_term'])
                if np.sum(render.cache['inview'][:,0]<HIT_DISTANCE)>0:
                    print('episode', episode, 'approach factor', np.round(A,2), 'angle', np.round(np.degrees(p_angle)), 'HIT')
                    pose_rec = np.vstack((pose_rec, bat.pose))
                    break
                if np.linalg.norm(bat.pose[:2]) > 5:
                    print('episode', episode, 'approach factor', np.round(A,2), 'angle', np.round(np.degrees(p_angle)), 'OUT')
                    pose_rec = np.vstack((pose_rec, bat.pose))
                    break
            poses.append(pose_rec)
            vs.append(np.asarray(v_rec))
            omegas.append(np.asarray(w_rec))
            onsets.append(np.asarray(onset_rec))
            iids.append(np.asarray(iid_rec))
            approach_terms.append(np.asarray(approach_rec))
            avoid_terms.append(np.asarray(avoid_rec))
    data = {'episodes': episodes, 'angles': angles, 'A_factors': A_factors, 'poses': poses, 'v': vs, 'omega': omegas, 'onsets': onsets, 'iid': iids, 'avoid_term': avoid_terms, 'approach_term': approach_terms}
    df = pd.DataFrame(data)
    df.to_pickle('AvoidApproach_TestData_'+DATE+'.pkl')

