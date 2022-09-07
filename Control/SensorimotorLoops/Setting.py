from math import pi

GRAVI_ACCEL = 9.8

###################################################################
### VELOCITY LIMIT  ###############################################
###################################################################
MAX_LINEAR_VELOCITY = 5 # robot max velo is 0.5 m/s
MAX_ANGULAR_VELOCITY = 10*pi
MAX_ANGULAR_ACCELERATION = 1*MAX_ANGULAR_VELOCITY
LINEAR_VELOCITY_OFFSET = 0.
DECELERATION_FACTOR = 1 # Choose between 1 to 5 the higher the steeper the deceleration
CENTRIFUGAL_ACCEL = 2 * GRAVI_ACCEL
CHIRP_RATE = 40 # Hz

### OTHERS KINEMATIC PARAMETERS  ##################################
TAU_K = 0.1
LINEAR_DECEL_LIMIT = -1.1 * MAX_LINEAR_VELOCITY
LINEAR_ACCEL_LIMIT = 1 * GRAVI_ACCEL
BODY_RADIUS = 0.15
BAIL_DISTANCE_MULTIPLIER = 5
APPROACH_STEER_DAMPING = 10 # less than 10 is very unstable. However, plan B cap everything so it's fine!

###################################################################
###  ROBOT CONVERSION   ###########################################
###################################################################
ROBOT_CHIRP_RATE = 2.5 #Hz
ROBOT_BAT_CONVERSION_RATE = CHIRP_RATE / ROBOT_CHIRP_RATE
ROBOT_MAX_LINEAR_VELOCITY = MAX_LINEAR_VELOCITY / ROBOT_BAT_CONVERSION_RATE
ROBOT_MAX_ANGULAR_VELOCITY= MAX_ANGULAR_VELOCITY / ROBOT_BAT_CONVERSION_RATE
ROBOT_MAX_ANGULAR_ACCELERATION = MAX_ANGULAR_ACCELERATION / ROBOT_BAT_CONVERSION_RATE
ROBOT_LINEAR_VELOCITY_OFFSET = LINEAR_VELOCITY_OFFSET / ROBOT_BAT_CONVERSION_RATE
ROBOT_CONVERSION = False

###################################################################
###################################################################
###################################################################



