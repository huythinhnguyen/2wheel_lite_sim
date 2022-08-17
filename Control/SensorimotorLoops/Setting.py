from math import pi

###################################################################
### VELOCITY LIMIT  ###############################################
###################################################################
MAX_LINEAR_VELOCITY = 2.5 # robot max velo is 0.5 m/s
MAX_ANGULAR_VELOCITY = 17*(pi/9)
CRUISE_THRESHOLD = 1.0
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



