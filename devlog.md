### 08/19/22:
- A few math methods implemented. For the avoidance loop:
  >> The velocity profile is partially drived from the tau. using the same exponential equation without considering the acceleration term.
  >> Rmin is derived from velocity by keeping the centrifugal acceleration constant. For now, keep it 1G but can be bring up to 2-3G if sharper turn is needed.
- 2 simple algorithm is implemented: (1) set turning radius = Rmin, calculate omega=v/r. (2) same with (1) but multiplying omega for exp(-d_onset).


### 08/18/22:
- Put the kinematic velocity profile equation into the Sensory motor loop.
- Test independent strategy. Make some nice publishable figure!

### 08/17/22:
- Try to Finish the Cue and the Avoidance loops. Tested Cues
- Need to work on the Avoid v, omega function

### 08/16/22:
- Working on _get_onset_distance. Finished need test.

### 08/13/22:
- Everything tested. Working on the Sensorimotor loop next!!! Make sure to have it independent from EchoVR. and take only the compressed input.
- >>> Not much done today but working on cleaning the _get_onset_distance now. away to not constantly retrieving and BG echos.

### 08/12/22:
- Yesterday. Finished all including FoV --> But need to test FoV.
- Also, working on the obscursion today!

### 08/11/22:
- Yesterday, finished building the Spatializer. --> Need testing.
- Build some environment helper next.

### 08/10/22:
- cleaning up the EchoVR. Working on the attenuation today!


#### 08/03/22:
- Start working on porting echoVR

### 08/02/22:
- Adding some arena building Utils [PROTOTYPING]
  - build_walls --> GOOD
  - build_circle -->GOOD
  - build_arc --> GOOD


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