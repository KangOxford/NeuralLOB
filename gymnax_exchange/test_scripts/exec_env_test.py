import os
import sys
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
import time
import dataclasses
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd  
import chex
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import faulthandler

faulthandler.enable()

# ============================
# Configuration
# ============================
test_steps = 15000  # Adjusted for your test case; make sure this isn't too high

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "./training_oneDay"
    
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "buy",
        "MAX_TASK_SIZE": 2,
        "WINDOW_INDEX": 0,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 0.1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*60,  # 
    }

    # Set up random keys for JAX
    rng = jax.random.PRNGKey(50)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

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
    # Define environment parameters
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=0.0001,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )
    

    # Initialize the environment state
    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {state.start_index}")
    print("Time for reset: \n", time.time() - start)

    print(f"Number of available windows: {env.n_windows}")

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data_exec.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    
    rewards = np.zeros((test_steps, 1), dtype=int)
    quant_executed = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    ask_price = np.zeros((test_steps, 1), dtype=int)
 

    # ============================
    # Track the number of valid steps
    # ============================
    valid_steps = 0

    # ============================
    # Run the test loop
    # ============================
    for i in range(test_steps):
        # ==================== ACTION ====================
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        test_action = jnp.array([0,0])
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
        rewards[i] = reward
        quant_executed[i] = state.quant_executed
        ask_price[i] = state.best_asks[-1,0]  # Store best ask
        bid_price[i] = state.best_bids[-1,0]
       
    
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            break

    # ============================
    # Clip the arrays to remove trailing zeros
    # ============================
    plot_until_step = valid_steps 

    rewards = rewards[:plot_until_step]
    quant_executed = quant_executed[:plot_until_step] 
 
    bid_price = bid_price[:plot_until_step]
    ask_price = ask_price[:plot_until_step]

    # ============================
    # Save all data to CSV
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([rewards, bid_price, ask_price,  quant_executed])
    
    # Add column headers
    column_names = ['rewards', 'bid_price', 'ask_price ',  'quant_executed']
    
    # Save data using pandas to handle CSV easily
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(reward_file, index=False)
    
    print(f"Data saved to {reward_file}")
    print(f"Last valid step {valid_steps}")

    # ============================
    # Plotting all metrics
    # ============================
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust the grid as needed

    # Plot best ask, best bid, and mid price on the same graph
    axes[0, 0].plot(range(plot_until_step), bid_price, label="Bid Price", color='blue')
    axes[0, 0].plot(range(plot_until_step), ask_price, label="Ask Price", color='green')
   
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].set_title("Bid, Ask over steps")
    axes[0, 0].legend()

    # Plot rewards over steps
    axes[0, 1].plot(range(plot_until_step), rewards, label="Reward", color='blue')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_title("Rewards Over Steps")
    
    # Plot quantity executed over steps
    axes[0, 2].plot(range(plot_until_step), quant_executed, label="Quantity Executed", color='purple')
    axes[0, 2].set_xlabel("Steps")
    axes[0, 2].set_ylabel("Quantity Executed")
    axes[0, 2].set_title("Quantity Executed Over Steps")

    
    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/exec_test.png'
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")
