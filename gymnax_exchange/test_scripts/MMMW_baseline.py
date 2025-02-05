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
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
import faulthandler
import pandas as pd  
import chex

faulthandler.enable()



class MMMWAgent:
    def __init__(self, num_actions, learning_rate, initial_volume):
        self.num_actions = num_actions
        self.weights = jnp.ones(num_actions)*10 // num_actions  # Initialize equal weights
        self.eta = learning_rate  # Learning rate for MW updates
        self.initial_volume = initial_volume  # Total volume to distribute across actions
    
    def compute_volumes(self):
        """Allocate volumes based on weights."""
        return self.initial_volume * self.weights
    
    def update_weights(self, payoffs):
        """Update weights using the Multiplicative Weights Update rule."""
        new_weights = self.weights * jnp.exp(self.eta * payoffs)
        self.weights = new_weights / np.sum(new_weights)  # Normalize to sum to 1
        #jax.debug.print("self.weights :{}",self.weights)
    def act(self, prices,quants, mid_price):
        """
        Decide action quantities based on weights and previous trade outcomes.
        
        Arguments:
        - prices: Array of 6 action prices (corresponding to bid/ask levels).
        - trades: Array of trade data (n_messages x 8).
        - mid_prices: Average mid-prices for the current step.
        
        Returns:
        - action_volumes: Quantities to offer at each action price.
        """
        # Compute payoffs for each action
        payoffs = np.zeros(self.num_actions)
        for i in range(self.num_actions):
         payoff = jnp.abs(prices[i] - mid_price) * jnp.abs(quants[i])
         payoffs[i] = payoff
       # jax.debug.print("payoffs:{}",payoffs)
        payoffs_norm=payoffs/(jnp.sum(payoffs)+0.0001)# normalise, small add to stabilise
  
      #  jax.debug.print("payoffs_norm:{}",payoffs_norm)
         
        
        # Update weights using the payoffs
        self.update_weights(payoffs_norm)

        # Compute action volumes based on weights
        action_volumes = self.compute_volumes()
        weights=self.weights
       # jax.debug.print("weights:{}",weights)
        return action_volumes,weights
# Initialize the agent
mmmw_agent = MMMWAgent(num_actions=6, learning_rate=0.01, initial_volume=100)

# During each step
def get_action(env_state):
    # Compute action prices from the state
    action_prices=env_state.prev_executed[:,0]
    action_quants=env_state.prev_executed[:,1]

    mid_prices = env_state.mid_price  

    # Agent decides action volumes
    action_volumes,weights = mmmw_agent.act(action_prices, action_quants, mid_prices)
    action_volumes=jnp.array(action_volumes).astype(jnp.int32)
    #jax.debug.print("weights :{}",weights)
    return action_volumes, weights


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "./training_oneDay"
    
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "buy",
        "MAX_TASK_SIZE": 500,
        "WINDOW_INDEX": 250,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 0.1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 240,  # 
    }

    # Set up random keys for JAX
    rng = jax.random.PRNGKey(50)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Initialize the environment
    env = MarketMakingEnv(
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
    test_steps=500

    # Initialize the environment state
    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {state.start_index}")
    print("Time for reset: \n", time.time() - start)
    print("Inventory after reset: \n", state.inventory)
    print(f"Number of available windows: {env.n_windows}")

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data_MMMW.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    weights_hist = np.zeros((test_steps, 6), dtype=float)
    ask_raw_orders_history = np.zeros((test_steps, 100, 6), dtype=int)
    bid_raw_orders_history = np.zeros((test_steps, 100,6), dtype=int)
    rewards = np.zeros((test_steps, 1), dtype=int)
    inventory = np.zeros((test_steps, 1), dtype=int)
    total_PnL = np.zeros((test_steps, 1), dtype=int)
    buyQuant = np.zeros((test_steps, 1), dtype=int)
    sellQuant = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    agr_bid_price =np.zeros((test_steps, 1), dtype=int)
    ask_price = np.zeros((test_steps, 1), dtype=int)
    
    agr_ask_price =np.zeros((test_steps, 1), dtype=int)
    state_best_ask = np.zeros((test_steps, 1), dtype=int)
    state_best_bid = np.zeros((test_steps, 1), dtype=int)
    averageMidprice = np.zeros((test_steps, 1), dtype=int)
    average_best_bid =np.zeros((test_steps, 1), dtype=int)
    average_best_ask =np.zeros((test_steps, 1), dtype=int)
    inventory_pnl = np.zeros((test_steps, 1), dtype=int)    
    realized_pnl = np.zeros((test_steps, 1), dtype=int)   
    unrealized_pnl = np.zeros((test_steps, 1), dtype=int) 
    bid_price_PP = np.zeros((test_steps, 1), dtype=int)
    ask_price_PP=np.zeros((test_steps, 1), dtype=int)


    output_dir = 'gymnax_exchange/test_scripts/test_outputs/'
   

    
   # book_vol_av_bid= np.zeros((test_steps, 1), dtype=int)
   # book_vol_av_ask = np.zeros((test_steps, 1), dtype=int)

    # ============================
    # Track the number of valid steps
    # ============================
    valid_steps = 0

    for i in range(test_steps):
        # ==================== ACTION ====================
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        action, weights=get_action(state)
        #jax.debug.print("weights:{}",weights)
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, action, env_params)
         # Store data
        ask_raw_orders_history[i, :, :] = state.ask_raw_orders
        bid_raw_orders_history[i, :, :] = state.bid_raw_orders
        rewards[i] = reward
        weights_hist[i, :] = weights
        inventory[i] = info["inventory"]
        total_PnL[i] = info["total_PnL"]
        buyQuant[i] = info["buyQuant"]
        sellQuant[i] = info["sellQuant"]
        agr_bid_price[i] = info["action_prices"][0]  
        bid_price[i] = info["action_prices"][1]  # Store best ask
        bid_price_PP[i] = info["action_prices"][2]
        agr_ask_price[i] = info["action_prices"][3]  
        ask_price[i] = info["action_prices"][4] 
        ask_price_PP[i] = info["action_prices"][5]# Store best bid
        averageMidprice[i] = info["averageMidprice"]  # Store mid price
        average_best_bid[i]=info["average_best_bid"]
        average_best_ask[i]=info["average_best_ask"]
        inventory_pnl[i] = info["InventoryPnL"]  
        realized_pnl[i] = info["approx_realized_pnl"]  
        unrealized_pnl[i] = info["approx_unrealized_pnl"] 

        
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            break

    # ============================
    # Clip the arrays to remove trailing zeros
    # ============================

    plot_until_step = valid_steps 
    weights_hist = weights_hist[:plot_until_step, :]
    rewards = rewards[:plot_until_step]
    inventory = inventory[:plot_until_step]
    total_PnL = total_PnL[:plot_until_step] 
    buyQuant = buyQuant[:plot_until_step]
    sellQuant = sellQuant[:plot_until_step]
    bid_price = bid_price[:plot_until_step]
    agr_bid_price = agr_bid_price[:plot_until_step]
    agr_ask_price = agr_ask_price[:plot_until_step]
    ask_price = ask_price[:plot_until_step]
    averageMidprice = averageMidprice[:plot_until_step]
    average_best_bid =average_best_bid[:plot_until_step]
    average_best_ask =average_best_ask[:plot_until_step]
    inventory_pnl = inventory_pnl[:plot_until_step]
    realized_pnl = realized_pnl[:plot_until_step]
    unrealized_pnl = unrealized_pnl[:plot_until_step]
    bid_price_PP =  bid_price_PP[:plot_until_step]
    ask_price_PP =  ask_price_PP[:plot_until_step]
    #state_best_bid = state_best_bid[:valid_steps-1]
       # state_best_ask = state_best_ask[:valid_steps-1]
   # book_vol_av_bid= book_vol_av_bid[:valid_steps-1]
   # book_vol_av_ask= book_vol_av_ask[:valid_steps-1]

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
    print(f"Inventory PnL Mean: {inventory_pnl.mean()}")
    print(f"Last valid step {valid_steps}")
    print(f"Last PnL: {total_PnL[-1]}")
    
    ##save weights##

    column_names=['BI weight','BB weight','PP Bid weight','AI Weight','BA weight','PP weight']
    df2=pd.DataFrame(weights_hist, columns=column_names)
    df2.to_csv('gymnax_exchange/test_scripts/test_outputs/weights.csv', index=False)
    # ============================
    # Plotting all metrics on one page
    # ============================
    # Create a figure with subplots (3 rows and 3 columns to fit the new data)
    # ============================
    # Plotting all metrics on one page
    # ============================
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))  # Adjust the grid to 4x3 to add an extra row for weights

    # Plot each metric on the respective subplot
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
    axes[1, 2].plot(range(plot_until_step), bid_price_PP, label="Bid Price PP", color='orange')
    axes[1, 2].plot(range(plot_until_step), ask_price_PP, label="Ask Price PP", color='green')
    axes[1, 2].plot(range(plot_until_step), agr_bid_price, label="Bid Price Agr", color='yellow')
    axes[1, 2].plot(range(plot_until_step), agr_ask_price, label="Ask Price Agr", color='black')
    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask, Mid, Agr, and PP Prices Over Steps")
    axes[1, 2].legend()

    axes[2, 0].plot(range(plot_until_step), inventory_pnl, label="Inventory PnL", color='gold')
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Inventory PnL")
    axes[2, 0].set_title("Inventory PnL Over Steps")

    axes[2, 1].plot(range(plot_until_step), realized_pnl, label="Realized PnL", color='orange')
    axes[2, 1].set_xlabel("Steps")
    axes[2, 1].set_ylabel("Realized PnL")
    axes[2, 1].set_title("Realized PnL Over Steps")

    axes[2, 2].plot(range(plot_until_step), unrealized_pnl, label="Unrealized PnL", color='purple')
    axes[2, 2].set_xlabel("Steps")
    axes[2, 2].set_ylabel("Unrealized PnL")
    axes[2, 2].set_title("Unrealized PnL Over Steps")

    # Plot weights on a new graph
    axes[3, 0].plot(range(plot_until_step), weights_hist[:, 0], label="BI Weight", color='blue')
    axes[3, 0].plot(range(plot_until_step), weights_hist[:, 1], label="BB Weight", color='green')
    axes[3, 0].plot(range(plot_until_step), weights_hist[:, 2], label="PP Bid Weight", color='red')
    axes[3, 0].plot(range(plot_until_step), weights_hist[:, 3], label="AI Weight", color='purple')
    axes[3, 0].plot(range(plot_until_step), weights_hist[:, 4], label="BA Weight", color='orange')
    axes[3, 0].plot(range(plot_until_step), weights_hist[:, 5], label="PP Weight", color='cyan')
    axes[3, 0].set_xlabel("Steps")
    axes[3, 0].set_ylabel("Weight Value")
    axes[3, 0].set_title("Weights Over Steps")
    axes[3, 0].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/MMMW.png'
    plt.savefig(combined_plot_file)
    plt.close()
