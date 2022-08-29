import numpy as np


SAMPLE_FREQ = 3e5
SPEED_OF_SOUND = 340
RAW_DATA_LENGTH = 7000
DISTANCE_ENCODING = np.arange(RAW_DATA_LENGTH) * 0.5 * (1/SAMPLE_FREQ) * SPEED_OF_SOUND
EMISSION_ENCODING = 0.33   # 0.45
EMISSION_INDEX = np.argmin(np.abs(DISTANCE_ENCODING - EMISSION_ENCODING)) + 1
### Background gaussian noise:
BACKGROUND_SIGMA = 0.6
BACKGROUND_MU = 5e-4
ANGLE_STEP = 1
OBJECTS_DICT = {'background': 0, 'pole': 1, 'plant': 2}
OUTWARD_SPREAD = 1
INWARD_SPREAD = 0.5
AIR_ABSORPTION = 1.31

### COMPRESSION SETTING:
QUIET_THRESHOLD = 0.5
N_SAMPLE = 125 # OR 140
COMPRESSED_DISTANCE_ENCODING = np.mean(DISTANCE_ENCODING.reshape(-1,N_SAMPLE), axis=1)
COMPRESSOR_NORMALIZED = False
QUIET_NORMALIZER = 8e-2

### VIEWER SETTING:
FOV_LINEAR = 3.
FOV_ANGULAR= 0.5*np.pi


### EAR GAIN CURVE SETTING: MODELING OF ROSE CURVE
NUMBER_OF_PEDALS = 3
ROSE_CURVE_B = 0.7
LEFT_EAR_FIT_ANGLE = (5/18)*np.pi
RIGHT_EAR_FIT_ANGLE=-(6/18)*np.pi



""" ECHOES MARKER  """ # COMMENT OUT WHEN NOT USED OR UPDATED
##################################
### 1ED, TRIPOD MOUNTED SENSOR ###
##################################
_POLE_STARTS  = {0.25: 0.13, 0.5: 0.38, 0.75: 0.62, 1.0: 0.88, 1.25: 1.12, 1.5: 1.36, 1.75: 1.62, 2.0: 1.87, 2.25: 2.11, 2.5: 2.35}
_POLE_ENDS    = {0.25: 0.47, 0.5: 0.66, 0.75: 0.88, 1.0: 1.1, 1.25: 1.31, 1.5: 1.53, 1.75: 1.76, 2.0: 1.98, 2.25: 2.22, 2.5: 2.48}
#_POLEREV_STARTS= {0.25: 0.89, 0.5: 0.79, 0.75: 0.94,1.0: 1.13, 1.25: 1.33, 1.5: 1.55, 1.75: 1.78, 2.0: 2.01, 2.25: 2.24, 2.5: 3.99}
#_POLEREV_ENDS  = {0.25: 1.13, 0.5: 0.99, 0.75: 1.15, 1.0: 1.33, 1.25: 1.51, 1.5: 1.71, 1.75: 1.92, 2.0: 2.14, 2.25: 2.35, 2.5: 3.99}
_PLANT_STARTS = {0.5: 0.26, 0.75: 0.49, 1.0: 0.76, 1.25: 0.99, 1.5: 1.24, 1.75: 1.49, 2.0: 1.74, 2.25: 1.91, 2.5: 2.24}
_PLANT_ENDS   = {0.5: 1.29, 0.75: 1.35, 1.0: 1.57, 1.25: 1.71, 1.5: 1.76, 1.75: 1.88, 2.0: 2.38, 2.25: 2.47, 2.5: 2.72}
