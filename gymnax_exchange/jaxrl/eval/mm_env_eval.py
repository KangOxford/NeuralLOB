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
params_filename = "/home/duser/AlphaTrade/params_file_balmy-terrain-229_02-03_12-09"
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
        "EPISODE_TIME": 60 * 5, # time in seconds
        "EP_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": "/home/duser/AlphaTrade/training_oneDay"
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

episodes = 1  # Run for multiple episodes
for episode in range(episodes):
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    done = jnp.array([False])
    episode_reward = 0
    episode_revenue = 0
    episode_executed = 0

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data_trained.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    test_steps = 1500
    #ask_raw_orders_history = np.zeros((test_steps, 100, 6), dtype=int)
    #bid_raw_orders_history = np.zeros((test_steps, 100,6), dtype=int)
    rewards = np.zeros((test_steps, 1), dtype=int)
    inventory = np.zeros((test_steps, 1), dtype=int)
    total_PnL = np.zeros((test_steps, 1), dtype=int)
    buyQuant = np.zeros((test_steps, 1), dtype=int)
    sellQuant = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    
    ask_price = np.zeros((test_steps, 1), dtype=int)
    
 
    state_best_ask = np.zeros((test_steps, 1), dtype=int)
    state_best_bid = np.zeros((test_steps, 1), dtype=int)
    averageMidprice = np.zeros((test_steps, 1), dtype=int)
    average_best_bid =np.zeros((test_steps, 1), dtype=int)
    average_best_ask =np.zeros((test_steps, 1), dtype=int)
  


    output_dir = 'gymnax_exchange/test_scripts/test_outputs/'
   

    
   # book_vol_av_bid= np.zeros((test_steps, 1), dtype=int)
   # book_vol_av_ask = np.zeros((test_steps, 1), dtype=int)

    # ============================
    # Track the number of valid steps
    # ============================
    valid_steps = 0

    # ============================
    # Run the test loop
    # ============================
    
    for step in range(test_steps):  # Maximum 100 steps per episode       
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(params, obsv)
        action = pi.sample(seed=_rng)
        #jax.debug.print("action:{}",action)
        log_prob = pi.log_prob(action)
        entropy = pi.entropy().mean()
        
        # Take a step in the environment
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
        
        episode_reward += reward.sum()
       

        inventory[step] = info["inventory"]
        total_PnL[step] = info["total_PnL"]
        buyQuant[step] = info["buyQuant"]
        sellQuant[step] = info["sellQuant"]
        #agr_bid_price[i] = info["action_prices"][0]  
        bid_price[step] = info["action_prices"][:,0]  # Store best ask
        #bid_price_PP[i] = info["action_prices"][2]
        #agr_ask_price[i] = info["action_prices"][3]  
        ask_price[step] = info["action_prices"][:,1] 
        #ask_price_PP[i] = info["action_prices"][5]# Store best bid
        averageMidprice[step] = info["averageMidprice"]  # Store mid price
        average_best_bid[step]=info["average_best_bid"]
        average_best_ask[step]=info["average_best_ask"]
        valid_steps += 1
        # Log results
        wandb.log({
            "Step":step,
            "Episode": episode,
            "reward":reward,
            "total_PnL":info["total_PnL"],
            "buyQuant":info["buyQuant"],
            "sellQuant":info["sellQuant"],
        
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

plot_until_step = valid_steps 
rewards = rewards[:plot_until_step]
inventory = inventory[:plot_until_step]
total_PnL = total_PnL[:plot_until_step] 
buyQuant = buyQuant[:plot_until_step]
sellQuant = sellQuant[:plot_until_step]
bid_price = bid_price[:plot_until_step]

ask_price = ask_price[:plot_until_step]
averageMidprice = averageMidprice[:plot_until_step]
average_best_bid =average_best_bid[:plot_until_step]
average_best_ask =average_best_ask[:plot_until_step]
 


    # ============================
    # Save all data to CSV
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
data = np.hstack([rewards, inventory, total_PnL, buyQuant, sellQuant, bid_price, ask_price, averageMidprice])
    
    # Add column headers
column_names = ['Reward', 'Inventory', 'Total PnL', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice']
    
# Save data using pandas to handle CSV easily
df = pd.DataFrame(data, columns=column_names)
df.to_csv(reward_file, index=False)
    
print(f"Data saved to {reward_file}")

print(f"Last valid step {valid_steps}")
print(f"Last PnL: {total_PnL[-1]}")
    
    # ============================
    # Plotting all metrics on one page
    # ============================
    # Create a figure with subplots (3 rows and 3 columns to fit the new data)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust the grid as needed

    

# Plot each metric on a separate subplot
axes[0, 0].plot(range(plot_until_step), rewards, label="Reward", color='blue')
axes[0, 0].set_xlabel("Steps")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].set_title("Rewards Over Steps")
    
axes[0, 1].plot(range(plot_until_step), inventory, label="Inventory", color='green')
axes[0, 1].set_xlabel("Steps")
axes[0, 1].set_ylabel("Inventory")
axes[0, 1].set_title("Inventory Over Steps")
    
axes[0, 2].plot(range(plot_until_step), total_PnL, label="Total PnL", color='orange')
axes[0, 2].set_xlabel("Steps")
axes[0, 2].set_ylabel("Total PnL")
axes[0, 2].set_title("Total PnL Over Steps")
    
axes[1, 0].plot(range(plot_until_step), buyQuant, label="Buy Quantity", color='red')
axes[1, 0].set_xlabel("Steps")
axes[1, 0].set_ylabel("Buy Quantity")
axes[1, 0].set_title("Buy Quantity Over Steps")
    
axes[1, 1].plot(range(plot_until_step), sellQuant, label="Sell Quantity", color='purple')
axes[1, 1].set_xlabel("Steps")
axes[1, 1].set_ylabel("Sell Quantity")
axes[1, 1].set_title("Sell Quantity Over Steps")
    
# Combined plot for Bid Price, Ask Price, and Average Mid Price
axes[1, 2].plot(range(plot_until_step), bid_price, label="Bid Price", color='pink')
axes[1, 2].plot(range(plot_until_step), ask_price, label="Ask Price", color='cyan')
axes[1, 2].plot(range(plot_until_step), averageMidprice, label="Average Mid Price", color='magenta')

combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/trained.png'
plt.savefig(combined_plot_file)
plt.close()
    


