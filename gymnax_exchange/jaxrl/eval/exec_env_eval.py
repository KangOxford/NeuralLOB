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


import wandb  # Import Weights & Biases for logging

# Initialize wandb
wandb.init(project="AlphaTrade_Eval", config={"run_type": "evaluation"})

# Load the trained model parameters
params_filename = "/home/duser/AlphaTrade/params_file_decent-fire-227_02-02_11-13"
with open(params_filename, 'rb') as f:
    params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

# Define environment configuration
config = {
    "ATFOLDER": "./training_oneDay",
    "NUM_ENVS": 1,
    "WINDOW_INDEX": 2,
    "TASKSIDE": "random",
    "ACTION_TYPE": "pure",
    "EPISODE_TIME": 300,
    "MAX_TASK_SIZE": 100,
    "EP_TYPE": "fixed_time",
    "RNN_TYPE": "S5",
    "HIDDEN_SIZE": 64,
    "REDUCE_ACTION_SPACE_BY": 10,
    "CONT_ACTIONS": False,
    "JOINT_ACTOR_CRITIC_NET": True,
}

# Initialize the environment
env = ExecutionEnv(
    alphatradePath=config["ATFOLDER"],
    task=config["TASKSIDE"],
    window_index=config["WINDOW_INDEX"],
    action_type=config["ACTION_TYPE"],
    episode_time=config["EPISODE_TIME"],
    max_task_size=config["MAX_TASK_SIZE"],
    ep_type=config["EP_TYPE"],
)

env_params = dataclasses.replace(
    env.default_params,
    task_size=config["MAX_TASK_SIZE"],
    episode_time=config["EPISODE_TIME"],
)

# Initialize the model
network = actorCriticS5.ActorCriticS5(env.action_space(env_params).shape[0], config=config)
init_hstate = actorCriticS5.ActorCriticS5.initialize_carry(config["NUM_ENVS"], actorCriticS5.ssm_size, actorCriticS5.n_layers)

# Initialize JAX random seed
rng = jax.random.PRNGKey(0)
reset_rng = jax.random.split(rng, config["NUM_ENVS"])

total_rewards = []
total_revenues = []
total_executed = []

episodes = 1  # Run for multiple episodes
for episode in range(episodes):
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    done = jnp.array([False])
    episode_reward = 0
    episode_revenue = 0
    episode_executed = 0
    
    for step in range(1000):  # Maximum 100 steps per episode
        rng, _rng = jax.random.split(rng)
        ac_in = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
        init_hstate, pi, value = network.apply(params, init_hstate, ac_in)
        
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
        entropy = pi.entropy().mean()
        
        # Take a step in the environment
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
        
        episode_reward += reward.sum()
        episode_revenue += info["total_revenue"].sum()
        episode_executed += info["quant_executed"].sum()
        # Log results
        wandb.log({
            "Step":step,
            "Episode": episode,
            "reward":reward,
            "total_revenue":info["total_revenue"],
            "quant_executed":info["quant_executed"],
            "average_price":info["average_price"],
            "advantage_reward":info["advantage_reward"],
            "Episode Reward": episode_reward,
            "Total Revenue": episode_revenue,
            "Quantity Executed": episode_executed,
            "Entropy": entropy,
        })
    
        
        if done.all():
            break
    
    total_rewards.append(episode_reward)
    total_revenues.append(episode_revenue)
    total_executed.append(episode_executed)
    
    
    print(f"Episode {episode}: Reward {episode_reward}, Revenue {episode_revenue}, Executed {episode_executed}")

# Final summary
print(f"Average Reward: {sum(total_rewards) / episodes}")
print(f"Average Revenue: {sum(total_revenues) / episodes}")
print(f"Average Quantity Executed: {sum(total_executed) / episodes}")
wandb.finish()


