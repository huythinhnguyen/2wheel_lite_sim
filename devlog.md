### 08/02/22:
- Adding some arena building Utils [PROTOTYPING]
  - build_walls --> GOOD
  - build_circle -->GOOD
  - build_arc --> mod from build circle
- Need to migrate EchoVR in:

### 07/27/22:
- Animation.Render.StillImage (still need to make save function) --> Use this to plot out episode's record.
- Built the GIF/MP4 generator in Render.Sequences
- Need to test
- Put an end to this and start integrating bat_snake

### 07/26/22:
- Good enough plotting function. Make some good animation function. Like GIF or video generation?? next

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