from math import pi

###################################################################
### VELOCITY LIMIT  ###############################################
###################################################################
MAX_LINEAR_VELOCITY = 2.5 # robot max velo is 0.5 m/s
MAX_ANGULAR_VELOCITY = 17*(pi/9)
LINEAR_VELOCITY_OFFSET = 0.
DECELERATION_FACTOR = 2 # Choose between 1 to 5 the higher the steeper the deceleration
CHIRP_RATE = 50 # Hz

###################################################################
###  ROBOT CONVERSION   ###########################################
###################################################################
ROBOT_CHIRP_RATE = 2.5 #Hz
ROBOT_BAT_CONVERSION_RATE = CHIRP_RATE / ROBOT_CHIRP_RATE
ROBOT_MAX_LINEAR_VELOCITY = MAX_LINEAR_VELOCITY / ROBOT_BAT_CONVERSION_RATE
ROBOT_MAX_ANGULAR_VELOCITY= MAX_ANGULAR_VELOCITY / ROBOT_BAT_CONVERSION_RATE
ROBOT_CONVERSION = False

###################################################################
###################################################################
###################################################################



