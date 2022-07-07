import numpy as np

from Simulation.Motion import State, Drive

from Simulation.ToyController import NaiveSquare

poses = np.array([]).reshape(0,3)
poses_hat = np.array([]).reshape(0,3)
desired_kinematics = np.array([]).reshape(0,2)
kinematics = np.array([]).reshape(0,2)
state = State()
drive = Drive()
c = NaiveSquare(init_pose = state.pose, width=5)
i = 0
print('Step\tpose\t\t\tpose_hat\t\t\tkinematic\t\tkinenatic_hat\t')
while not c._run_ended:
    pose = state.pose
    pose_hat = c.gps.pose_hat
    poses = np.vstack((poses, pose))
    poses_hat = np.vstack((pose_hat, poses_hat))
    desired_kinematic = c.track_goal(pose)
    drive.kinematic_update(new_kinematic=desired_kinematic)
    state.update_kinematic(new_v=drive.kinematic[0], new_w=drive.kinematic[1])
    state.update_pose()
    desired_kinematics = np.vstack((desired_kinematics, desired_kinematic))
    kinematics = np.vstack((kinematics, drive.kinematic))
    i+=1
    print(i,end='\t')
    print(np.round(pose,3),end='\t')
    print(np.round(pose_hat,3),end='\t')
    print(np.round(desired_kinematic,3),end='\t')
    print(np.round(drive.kinematic,3))
    if i > 10_000: break
    
print('DONE. SAVE DATA')    
np.savez('data.npz', poses=poses, poses_hat=poses_hat, desired_kinematics=desired_kinematics, kinematics=kinematics)

print(poses.shape)
print(poses_hat.shape)

from matplotlib import pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.scatter(poses[:,0], poses[:,1], s=1, label='pose', alpha=0.3)
ax1.scatter(poses_hat[:,0], poses_hat[:,1],s=1, label='pose_hat', alpha=0.3)
ax1.legend()
ax2.plot(desired_kinematics[:,0], label='desired v', alpha=0.5)
ax2.plot(kinematics[:,0], label='actual v', alpha=0.5)
ax2.legend()
ax3.plot(desired_kinematics[:,1], label='desired w', alpha=0.5)
ax3.plot(kinematics[:,1], label='actual w', alpha=0.5)
ax3.legend()
plt.show()