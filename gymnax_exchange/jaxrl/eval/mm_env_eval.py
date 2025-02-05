import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)
import dataclasses

from purejaxrl.purejaxrl.wrappers import LogWrapper, FlattenObservationWrapper


#=============import the policy==============#
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import checkpoints


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
import pandas as pd 
import matplotlib.pyplot as plt 
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.base_env import EnvParams as BaseEnvParams
from gymnax_exchange.jaxen.base_env import EnvState as BaseEnvState
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv as MarketMakingEnv
from gymnax_exchange.utils import utils
import dataclasses
from flax.core import frozen_dict


import wandb  # Import Weights & Biases for logging

# Initialize wandb
wandb.init(project="AlphaTrade_Eval", config={"run_type": "evaluation"})

# Load the trained model parameters
params_filename = "/home/duser/AlphaTrade/params_file_eager-sea-243_02-04_19-42"
with open(params_filename, 'rb') as f:
    params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

# Define environment configuration
config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 6e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": 200, # 2 fix random episode #-1,
        "DEBUG": True,
        
        "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 1., #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "MAX_TASK_SIZE": 100,
        #"TASK_SIZE": 100, # 500,
        "EPISODE_TIME": 60 * 60, # time in secondss
        "EP_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": "/home/duser/AlphaTrade/testing"
    }

# Initialize the environment
env = MarketMakingEnv(
    alphatradePath=config["ATFOLDER"],
    task=config["TASKSIDE"],
    window_index=config["WINDOW_INDEX"],
    action_type=config["ACTION_TYPE"],
    episode_time=config["EPISODE_TIME"],
    ep_type=config["EP_TYPE"],
)

env_params = dataclasses.replace(
    env.default_params,
    #task_size=config["MAX_TASK_SIZE"],
    episode_time=config["EPISODE_TIME"],
)

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

# Initialize the model
network = ActorCritic(
            env.action_space(env_params).n
           , activation=config["ACTIVATION"] #
        )


# Initialize JAX random seed
rng = jax.random.PRNGKey(0)
reset_rng = jax.random.split(rng, config["NUM_ENVS"])

total_rewards = []
total_revenues = []
total_executed = []

episodes = 3 # Run for multiple episodes
for episode in range(episodes):
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    done = jnp.array([False])
    episode_reward = 0
    episode_revenue = 0
    episode_executed = 0
    # ============================
    # Run the test loop
    # ============================
    action_history = []
    
    for step in range(100000):
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(params, obsv)
        action = pi.sample(seed=_rng)
        action_history.append(int(action))  # Convert JAX tensor to Python int

        log_prob = pi.log_prob(action)
        entropy = pi.entropy().mean()
        
        
        # Take a step in the environment
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
        
        episode_reward += reward.sum()
       
        # Logging every N steps (to avoid spamming WandB)
        if step % 100 == 0:
            unique_actions, counts = np.unique(action_history, return_counts=True)
            action_distribution = {f"action_{int(a)}": int(c) for a, c in zip(unique_actions, counts)}
            wandb.log(action_distribution)
     
        # Log results
        wandb.log(
            data={
            "Step":step,
            "Episode": episode,
            "reward":reward,
            "total_PnL":info["total_PnL"],
            "buyQuant":info["buyQuant"],
            "sellQuant":info["sellQuant"],
            "inventory":info["inventory"]
            },commit=True
            )
    
        
        if done.all():
            break
    
    