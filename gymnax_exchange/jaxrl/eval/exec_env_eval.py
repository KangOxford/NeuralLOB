# from jax import config
# config.update("jax_enable_x64",True)
# ============== testing scripts ===============
import os
import sys
import time 
import timeit
import random
import dataclasses
from ast import Dict
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import lax, flatten_util
# ----------------------------------------------
import gymnax
from gymnax.environments import environment, spaces
# sys.path.append('/Users/sasrey/AlphaTrade')
# sys.path.append('/homes/80/kang/AlphaTrade')
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
# ---------------------------------------------- 
import chex
from jax import config
import faulthandler
faulthandler.enable()
chex.assert_gpu_available(backend=None)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64",True)
config.update("jax_disable_jit", False) # use this during training
# config.update("jax_disable_jit", True) # Code snippet to disable all jitting.
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())
jax.numpy.set_printoptions(linewidth=183)


#=============import the policy==============#
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import checkpoints
from gymnax_exchange.jaxrl import actorCriticS5

# ================= imports =================#


from ast import Dict
from contextlib import nullcontext
# from email import message
# from random import sample
# from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, flatten_util
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.base_env import EnvParams as BaseEnvParams
from gymnax_exchange.jaxen.base_env import EnvState as BaseEnvState
from gymnax_exchange.jaxen.exec_env import ExecutionEnv as ExecutionEnv
from gymnax_exchange.utils import utils
import dataclasses
from flax.core import frozen_dict


import jax.numpy as jnp
from flax import serialization


config = {
    "HIDDEN_SIZE": 64,
    "MAX_TASK_SIZE": 100,  
    "REDUCE_ACTION_SPACE_BY": 10,  
    "CONT_ACTIONS": False,
    "JOINT_ACTOR_CRITIC_NET": True,
}

import jax
import jax.numpy as jnp
import time



# Load the saved policy parameters

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        ATFolder = "./training_oneDay"
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"

    ##load the data
    with open('/home/duser/AlphaTrade/params_file_decent-fire-227_02-02_11-13', 'rb') as f:
     params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

     #print(f"params shape: {jax.tree_map(lambda x: x.shape, params)}")
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    
    # print(env_params.message_data.shape, env_params.book_data.shape)
    config = {
        "ATFOLDER": ATFolder,
        "LR": 1e-4, # 1e-4, 5e-4, #5e-5, #1e-4,#2.5e-5,
        "LR_COS_CYCLES": 8,  # only relevant if ANNEAL_LR == "cosine"
        "ENT_COEF": 0., # 0., 0.001, 0, 0.1, 0.01, 0.001
        "NUM_ENVS": 1, #512, 1024, #128, #64, 1000,
        "TOTAL_TIMESTEPS": 2e6,  # 1e8, 5e7, # 50MIL for single data window convergence #,1e8,  # 6.9h
        "NUM_MINIBATCHES": 4, #8, 4, 2,
        "UPDATE_EPOCHS": 10, #10, 30, 5,
        "NUM_STEPS": 10, #20, 512, 500,
        "CLIP_EPS": 0.2,  # TODO: should we change this to a different value? 
        
        "GAMMA": 0.999,
        "GAE_LAMBDA": 0.99, #0.95,
        "VF_COEF": 1.0, #1., 0.01, 0.001, 1.0, 0.5,
        "MAX_GRAD_NORM": 5, # 0.5, 2.0,
        "ANNEAL_LR": 'cosine', # 'linear', 'cosine', False
        "NORMALIZE_ENV": False,  # only norms observations (not reward)
        
        "RNN_TYPE": "S5",  # "GRU", "S5"
        "HIDDEN_SIZE": 64,  # 128
        "ACTIVATION_FN": "relu", # "tanh", "relu", "leaky_relu", "sigmoid", "swish"
        "ACTION_NOISE_COLOR": 2.,  # 2  # only relevant if CONT_ACTIONS == True

        "RESET_ADAM_COUNT": True,  # resets Adam's t (count) every update
        "ADAM_B1": 0.99,  # 0.9
        "ADAM_B2": 0.99,
        "ADAM_EPS": 1e-5,  # 1e-4, 1e-6
        
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": 2, # 2 fix random episode #-1,
        "DEBUG": True,
        
        "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 1., #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "MAX_TASK_SIZE": 100,
        "TASK_SIZE": 100, # 500,
        "EPISODE_TIME": 60 * 5, # time in seconds
        "EP_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "CONT_ACTIONS": False,  # True
        "JOINT_ACTOR_CRITIC_NET": True,  # True, False
        "ACTOR_STD": "state_dependent",  # 'state_dependent', 'param', 'fixed'
        "REDUCE_ACTION_SPACE_BY": 10,
      
       
        # "ATFOLDER": "./training_oneMonth/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        
    }
    # env=ExecutionEnv(ATFolder,"sell",1)
    env = ExecutionEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        ep_type=config["EP_TYPE"],
        
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=1,
        task_size=config["MAX_TASK_SIZE"],
        episode_time=config["EPISODE_TIME"],  # in seconds
    )

    network = actorCriticS5.ActorCriticS5(env.action_space(env_params).shape[0], config=config)

    # Initialize hidden state
    init_hstate = actorCriticS5.ActorCriticS5.initialize_carry(
      config["NUM_ENVS"],actorCriticS5.ssm_size, actorCriticS5.n_layers
    )


    rng = jax.random.PRNGKey(0)
    reset_rng = jax.random.split(rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    done = jnp.array([False])

    for step in range(100):  # Run for 100 steps or until the episode ends
        rng, _rng = jax.random.split(rng)
        
        # Select action
        ac_in = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
        init_hstate, pi, value = network.apply(params, init_hstate, ac_in)
        
        action = pi.sample(seed=_rng)
        jax.debug.print("action:{}",action)
        log_prob = pi.log_prob(action)

        value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
        
        # Take step in environment
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
        
        print(f"Step {step}: Action {action}, Reward {reward}")
        
        if done.all():
            print("Episode finished")
            break


