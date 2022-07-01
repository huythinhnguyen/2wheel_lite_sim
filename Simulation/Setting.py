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
    def __init__(self, mode='default', L2R_bias=0.9, bias_var = 0.1, velo_var = 0.05, custom_spec=None):
        spec = create2_spec if mode=='default' else custom_spec
        
        self.wheelbase = spec['wheelbase']
        self.wheel_diameter = spec['wheel_diameter']
        self.body_radius = spec['body_radius']
        self.wheelbase_offset = spec['wheelbase_offset']
        self.wheel_velocity_var = spec['wheel_velocity_var']
        self.steering_var = spec['steering_var']
        self.abs_min_turning_radius = spec['abs_min_turning_radius']
        
        self.L2R_bias = L2R_bias
        self.bias_var = bias_var
        self.velo_var = velo_var

        # Initialize a based velocity factor on each wheel:
        self.v_left_factor = 1 + np.sqrt(self.velo_var)*np.random.randn()
        bias = self.L2R_bias + np.sqrt(self.bias_var)*np.random.randn()
        self.v_right_factor = self.v_left_factor / bias


    def reset(self):
        # Initialize a based velocity factor on each wheel:
        self.v_left_factor = 1 + np.sqrt(self.velo_var)*np.random.randn()
        bias = self.L2R_bias + np.sqrt(self.bias_var)*np.random.randn()
        self.v_right_factor = self.v_left_factor / bias


        
