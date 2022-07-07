import numpy as np

from Simulation.Motion import State, Drive

from Simulation.ToyController import NaiveSquare

poses = np.array([]).reshape(0,3)
poses_hat = np.array([]).reshape(0,3)
desired_kinematics = np.array([]).reshape(0,2)
kinematics = np.array([]).reshape(0,2)
state = State()
drive = Drive()
c = NaiveSquare(init_pose = state.pose, width=1)
i = 0
while not c._run_ended:
    pose = state.pose
    pose_hat = c.gps.pose_hat
    poses = np.vstack((pose, pose))
    poses_hat = np.vstack((pose_hat, poses_hat))
    desired_kinematic = c.track_goal(pose)
    drive.kinematic_update(new_kinematic=desired_kinematic)
    state.update_kinematic(new_v=drive.kinematic[0], new_w=drive.kinematic[1])
    state.update_pose()
    desired_kinematics = np.vstack((desired_kinematics, desired_kinematic))
    kinematics = np.vstack((kinematics, drive.kinematic))
    i+=1
    print('Step\tpose\t\t\tpose_hat\t\t\tkinematic\t\tkinenatic_hat\t')
    print(i,'\t',np.round(pose,3),'\t\t\t',np.round(pose_hat,3),'\t\t\t',np.round(desired_kinematic,3),'\t\t',np.round(drive.kinematic,3))
    
print('DONE. SAVE DATA')    
np.savez('data.npz', poses=poses, poses_hat=poses_hat, desired_kinematics=desired_kinematics, kinematics=kinematics)

from matplotlib import pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.scatter(pose[:,0], pose[:,1], s=5, label='pose')
ax1.scatter(pose_hat[:,0], pose[:,1],s=5, label='pose_hat')
ax1.legend()
ax2.plot(desired_kinematics[:,0], 'desired v')
ax2.plot(kinematics[:,0], 'actual v')
ax3.plot(desired_kinematics[:,1], 'desired w')
ax3.plot(kinematics[:,1], 'actual w')
plt.show()