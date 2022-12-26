from math import pi

GRAVI_ACCEL = 9.8

CONTROL_OVERWRITE = {
    'CENTRIFUGAL_ACCEL': 5*GRAVI_ACCEL,
    'MAX_LINEAR_VELOCITY': 2.5,
    'MIN_LINEAR_VELOCITY': 0.0,
    'MAX_ANGULAR_VELOCITY': 20*pi,
    'BODY_RADIUS': 0.2,
    'BAIL_DISTANCE_MULTIPLIER': 4,
}

SENSOR_OVERWRITE = {}

def overwrite_config(control_config, sensor_config):
    for key, value in CONTROL_OVERWRITE.items():
        if hasattr(control_config, key): setattr(control_config, key, value)
        else: raise ValueError('Control config has no attribute {}'.format(key))
    for key, value in SENSOR_OVERWRITE.items():
        if hasattr(sensor_config, key): setattr(sensor_config, key, value)
        else: raise ValueError('Sensor config has no attribute {}'.format(key))
    return sensor_config, control_config