import numpy as np

create2_spec = {'wheelbase': 0.235,
                'wheel_velocity_range': [-0.5,0.5], # m/s, [min max]
                'min_wheel_speed': 0.011,
                'wheel_diameter': 0.072,
                'body_radius': 0.175,
                'wheelbase_offset': 0.0,
                'wheel_velocity_var': 1e-4, # variance of wheel_velocity from the expected setting. system is relatively OK with 1e-5 to 1e-6
                #'steering_var': 0.001, # NOT VERY USEFUL.
                'abs_min_turning_radius': 0.02,
                'L2R_bias_mean': 1.0,
                'L2R_bias_var': 1e-3, # aggressive but not extreme. 1e-1 is very extreme and 1e-6 is pretty stable.
                'velo_var': 1e-3, # make both wheel go a bit faster or slower. affect the turn more than going straight. Aiming to simulate different road surface resistance.
                'heading_offset_mean': 0.0, # not yet implemented
                'heading_offset_var': np.pi/36 # not yet implemeted
                }


steam_gps_var = [5e-6, 5e-6, 6e-3]

class Create2:
    def __init__(self, mode='default', custom_spec=None, noise=True):
        spec = create2_spec if mode=='default' else custom_spec
        
        self.wheelbase = spec['wheelbase']
        self.wheel_velocity_range = spec['wheel_velocity_range']
        self.wheel_diameter = spec['wheel_diameter']
        self.min_wheel_speed = spec['min_wheel_speed']
        self.body_radius = spec['body_radius']
        self.wheelbase_offset = spec['wheelbase_offset']
        self.wheel_velocity_var = spec['wheel_velocity_var'] if noise else 0.0
        #self.steering_var = spec['steering_var'] if noise else 0.0
        self.abs_min_turning_radius = spec['abs_min_turning_radius']
        
        self.L2R_bias_mean = spec['L2R_bias_mean'] if noise else 1.0
        self.L2R_bias_var = spec['L2R_bias_var'] if noise else 0.0
        self.velo_var = spec['velo_var'] if noise else 0.0
        self.heading_offset_mean = spec['heading_offset_mean'] if noise else 0.0
        self.heading_offset_var = spec['heading_offset_var'] if noise else 0.0

        # Initialize a based velocity factor on each wheel:
        self.v_left_factor = 1 + np.sqrt(self.velo_var)*np.random.randn()
        bias = self.L2R_bias_mean + np.sqrt(self.L2R_bias_var)*np.random.randn()
        self.v_right_factor = self.v_left_factor / bias
        self.heading_offset = self.heading_offset_mean + np.sqrt(self.heading_offset_var)*np.random.randn()


    def reset(self):
        # Initialize a based velocity factor on each wheel:
        self.v_left_factor = 1 + np.sqrt(self.velo_var)*np.random.randn()
        bias = self.L2R_bias_mean + np.sqrt(self.L2R_bias_var)*np.random.randn()
        self.v_right_factor = self.v_left_factor / bias
        self.heading_offset = self.heading_offset + np.sqrt(self.heading_offset_var)*np.random.randn()





        
