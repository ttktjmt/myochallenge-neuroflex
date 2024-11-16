import os
from myosuite.utils import gym; register=gym.register
import myosuite
import numpy as np
# from definitions import ROOT_DIR  # pylint: disable=import-error
from myosuite.envs.myo.myobase import register_env_with_variants
import  myosuite.envs.myo.myochallenge as myochallenge_module
import myosuite.envs.myo.myodm.myodm_v0 as myodex_module
#myosuite_path = os.path.join(ROOT_DIR, "data", "myosuite")

curr_dir = os.path.dirname(os.path.abspath( myochallenge_module.__file__))
myodex_base_dir = os.path.dirname(os.path.abspath( myodex_module.__file__))

register_env_with_variants(id='CustomBimanualEnv-v0',
        entry_point='envs.bimanual:CustomBimanualEnv',
        max_episode_steps=300,
        kwargs={
            'model_path': curr_dir + '/../assets/arm/myoarm_bionic_bimanual.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'obj_scale_change': [0.1, 0.05, 0.1],  # 10%, 5%, 10% scale variations in respective geom directions
            'obj_mass_change': (-0.050, 0.050),  # +-50gms
            'obj_friction_change': (0.1, 0.001, 0.00002)  # nominal: 1.0, 0.005, 0.0001
        }
    )
