import numpy as np

create2_spec = {'wheelbase': 0.235,
           'wheel_velocity_range': [-0.5,0.5], # m/s, [min max]
           'wheel_diameter': 0.072,
           'body_radius': 0.175,
           'wheelbase_offset': 0.0,
           'wheel_velocity_var': 0.01,
           'steering_var': 0.004,
           'abs_min_turning_radius': 0.02}

steam_gps_var = [5e-6, 5e-6, 6e-3]

class Create2:
    def __init__(self, mode='default', L2R_bias=0.9, bias_var = 0.1):
        if mode == 'default':
            self.wheelbase = create2_spec['wheelbase']
            self.wheel_diameter = create2_spec['wheel_diameter']
            self.body_radius = create2_spec['body_radius']
            self.wheelbase_offset = create2_spec['wheelbase_offset']
            self.wheel_velocity_var = create2_spec['wheel_velocity_var']
            self.steering_var = create2_spec['steering_var']
            self.abs_min_turning_radius = create2_spec['abs_min_turning_radius']
            
        self.L2R_bias = L2R_bias