import os
import pickle
import time
import sys
util_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
sys.path.append(util_path) # add utils path for local test

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
# import gymnasium as gym
# from myosuite.utils import gym

from utils import RemoteConnection

from stable_baselines3 import PPO

"""
Define your custom observation keys here
"""
DEFAULT_OBS_KEYS = [
    'time',
    'myohand_qpos',
    'myohand_qvel',
    'pros_hand_qpos',
    'pros_hand_qvel',
    'object_qpos',
    'object_qvel',
    'touching_body',
    'act',
]
custom_obs_keys = [
    "time",
    'myohand_qpos',
    'myohand_qvel',
    'pros_hand_qpos',
    'pros_hand_qvel',
    'object_qpos',
    'object_qvel',
    'start_pos',
    'goal_pos',
    'obj_pos',
    'reach_err',
    'pass_err',
    'act',
    "touching_body",
]

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

class Policy:
    def __init__(self, env):
        self.action_space = env.action_space

    def __call__(self, env):
        return self.action_space.sample()

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)

time.sleep(8)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

policy = Policy(rc)

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, DEFAULT_OBS_KEYS).shape
rc.set_output_keys(DEFAULT_OBS_KEYS)

model = PPO.load("baseline")

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"MANI-MPL: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = rc.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    for step in range(1000):
        if flag_trial: break

        ################################################
        ## Replace with your trained policy.
        # obs = rc.obsdict2obsvec(rc.get_obsdict(), rc.obs_keys)[1]
        obs = rc.obsdict2obsvec(rc.get_obsdict(), DEFAULT_OBS_KEYS)
        print(f"obs keys: {rc.get_obsdict().keys()}")
        # print(f"obs: {obs}")
        action, _ = model.predict(obs, deterministic=True) # obs shape is different
        # hard-coding the myoHand to release object
        action[30] = 1
        if step > 130:
        # if obs.
            action[32:40] = 0
            action[40:49] = 1

        # hard code grasping motion of MPL

        #hard coding the MPL to the desire position, since we know the actuation of the MPL is the last 17 index of action
        action[-17:] = np.array([-0.65001469 , 1.     ,    -0.23187843 , 0.59583695 , 0.92356688, -0.16,
                                -0.28 ,      -0.88   ,     0.25 ,      -0.846   ,   -0.24981132 ,-0.91823529,
                                -0.945  ,    -0.925   ,   -0.929   ,   -0.49    ,   -0.18      ])
        if step > 250:
            action[-17:] = np.array([-0.4199236 ,  1.      ,   -0.9840558 ,  0.35299219,  0.92356688,  0.02095238,
                                        -0.28    ,   -0.88  ,      0.25      , -0.846     , -0.24981132, -0.91823529,
                                        -0.945   ,   -0.925   ,   -0.929    ,  -0.49     ,  -0.918     ])
        if step > 350:
            action[-17:] = np.array([-0.4199236 ,  1.     ,    -0.9840558,   0.35299219 , 0.3910828 ,  0.02095238,
                                        -0.28    ,   -0.88     ,   0.25   ,    -0.846     , -0.24981132 ,-0.91823529,
                                        -0.945    ,  -0.925    ,  -0.929    ,  -0.49  ,     -0.918     ])
        ################################################

        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
