### 07/22/22:
- Tested ControlDubinsPath --> WORK!
- Implemented the Dubins Path Planner
- Maybe work on Animations next

### 07/20/22
Issues:
- Need to implement some way to do animation. Some sample Controller --> passion project.
- Starting to rebuild this for echo_gym simulation.
Fixes:
- Everything in Motion worked and tested. GPS won't be used for regular sim unless for demoing some other algorithm.
- Motion.Drive track correctly without noise. When added noise, tracking is terrible. (Feature/Bug???)
- Tuned all the variance for create2 in Setting
### 07/19/22
Issues:
- Removed noise from from in Drive but the Drive class still did not track pose accurately.
- Linear velo is about 0.87 of actual. Angular velo is shifted toward the negatives
Fixed:
- rotation around ICC works well!
- Motion.State work wonderfully!

### 07/05/22
Issues:
- ToyController was too complicated for testing.
- There might be bug in Motion.update()
- The rotation around ICC might be wrong.