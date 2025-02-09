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

# ============================
# Configuration
# ============================
test_steps = 15000 # Adjusted for your test case; make sure this isn't too high

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
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    
    #ask_raw_orders_history = np.zeros((test_steps, 100, 6), dtype=int)
    #bid_raw_orders_history = np.zeros((test_steps, 100,6), dtype=int)
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

    # ============================
    # Run the test loop
    # ============================
    for i in range(test_steps):
        # ==================== ACTION ====================
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        test_action= test_action = env.action_space().sample(key_policy) 
        #test_action = 1
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
        #ask_raw_orders_history[i, :, :] = state.ask_raw_orders
        #bid_raw_orders_history[i, :, :] = state.bid_raw_orders
        rewards[i] = reward
        inventory[i] = info["inventory"]
        total_PnL[i] = info["total_PnL"]
        buyQuant[i] = info["buyQuant"]
        sellQuant[i] = info["sellQuant"]
        #agr_bid_price[i] = info["action_prices"][0]  
        bid_price[i] = info["action_prices"][0]  # Store best ask
        #bid_price_PP[i] = info["action_prices"][2]
        #agr_ask_price[i] = info["action_prices"][3]  
        ask_price[i] = info["action_prices"][1] 
        #ask_price_PP[i] = info["action_prices"][5]# Store best bid
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

    rewards = rewards[:plot_until_step]
    inventory = inventory[:plot_until_step]
    total_PnL = total_PnL[:plot_until_step] 
    buyQuant = buyQuant[:plot_until_step]
    sellQuant = sellQuant[:plot_until_step]
    bid_price = bid_price[:plot_until_step]
    #agr_bid_price = agr_bid_price[:plot_until_step]
    #agr_ask_price = agr_ask_price[:plot_until_step]
    ask_price = ask_price[:plot_until_step]
    averageMidprice = averageMidprice[:plot_until_step]
    average_best_bid =average_best_bid[:plot_until_step]
    average_best_ask =average_best_ask[:plot_until_step]
    inventory_pnl = inventory_pnl[:plot_until_step]
    realized_pnl = realized_pnl[:plot_until_step]
    unrealized_pnl = unrealized_pnl[:plot_until_step]
    #bid_price_PP =  bid_price_PP[:plot_until_step]
    #ask_price_PP =  ask_price_PP[:plot_until_step]
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
   # axes[1, 2].plot(range(plot_until_step), average_best_bid, label="Average Best Bid", color='red')
   # axes[1, 2].plot(range(plot_until_step), average_best_ask, label="Average Best Ask", color='blue')
    #axes[1, 2].plot(range(plot_until_step), bid_price_PP, label="Bid Price PP", color='orange')
    #axes[1, 2].plot(range(plot_until_step), ask_price_PP, label="Ask Price PP", color='green')
    #axes[1, 2].plot(range(plot_until_step), agr_bid_price, label="Bid Price Agr", color='yellow')
    #axes[1, 2].plot(range(plot_until_step), agr_ask_price, label="Ask Price Agr", color='black')
    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask, Mid,Agr, and PP Prices Over Steps")
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

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/reward_symmetrically_dampened_0.0001_lambda_all_steps.png'
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")



    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Ensure output directory exists

    os.makedirs(output_dir, exist_ok=True)





    # ============================
    # Sort and Filter Order Books
    # ============================
    def sort_and_filter_order_books(ask_raw_orders_history, bid_raw_orders_history):
        """
        Filters and sorts the order books:
        - Removes rows where the price is invalid (e.g., -1).
        - Asks: ascending order by price.
        - Bids: descending order by price.
        """
        sorted_ask_history = []
        sorted_bid_history = []
        for step in range(len(ask_raw_orders_history)):
            # Filter invalid asks (price != -1)
            valid_asks = ask_raw_orders_history[step][ask_raw_orders_history[step][:, 0] != -1]
            # Sort valid asks in ascending order by price
            sorted_asks = valid_asks[valid_asks[:, 0].argsort()]

            # Filter invalid bids (price != -1)
            valid_bids = bid_raw_orders_history[step][bid_raw_orders_history[step][:, 0] != -1]
            # Sort valid bids in descending order by price
            sorted_bids = valid_bids[valid_bids[:, 0].argsort()[::-1]]

            sorted_ask_history.append(sorted_asks)
            sorted_bid_history.append(sorted_bids)
        return sorted_ask_history, sorted_bid_history


    # Sort and filter the order books
   # sorted_ask_raw_orders_history, sorted_bid_raw_orders_history = sort_and_filter_order_books(
    #    ask_raw_orders_history, bid_raw_orders_history
    #)
    
   # print(f"final ask{ask_raw_orders_history[-1][:, :]}")
   # print(f"final ask -2 {ask_raw_orders_history[-2][:, :]}")
   # print(f"final bid{bid_raw_orders_history[-1][:, :]}")
   # print(f"final bid -2 {bid_raw_orders_history[-2][:, :]}")

    # ============================
    # Animation of Cumulative Sum
    # ============================
    def animate_order_book_evolution(sorted_ask_history, sorted_bid_history, valid_steps, output_file):
        """
        Creates an animation of the cumulative sum graphs over time.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # Initialization function
        def init():
            ax.clear()
            ax.set_xlim(0, 1)  # Temporary values, adjusted dynamically
            ax.set_ylim(0, 1)  # Temporary values, adjusted dynamically
            ax.set_xlabel("Price")
            ax.set_ylabel("Cumulative Quantity")
            ax.set_title("Limit Order Book Evolution")

        # Animation update function
        def update(step):
            ax.clear()

            # Extract data for the current step
            ask_prices = sorted_ask_history[step][:, 0]
            ask_quantities = sorted_ask_history[step][:, 1].cumsum()
            bid_prices = sorted_bid_history[step][:, 0]
            bid_quantities = sorted_bid_history[step][:, 1].cumsum()

            # Update axis limits dynamically
            ax.set_xlim(min(bid_prices.min(), ask_prices.min()), max(bid_prices.max(), ask_prices.max()))
            ax.set_ylim(0, max(ask_quantities.max(), bid_quantities.max()) * 1.1)

            # Plot bids
            ax.fill_between(
                bid_prices, 0, bid_quantities, step="pre", color="green", alpha=0.5, label="Bid Depth"
            )

            # Plot asks
            ax.fill_between(
                ask_prices, 0, ask_quantities, step="pre", color="red", alpha=0.5, label="Ask Depth"
            )

            # Add vertical dashed line
            ax.axvline(x=(max(bid_prices)+min(ask_prices))/2, color="black", linestyle="--", linewidth=1)

            # Set labels and legend
            ax.set_xlabel("Price")
            ax.set_ylabel("Cumulative Quantity")
            ax.set_title(f"Limit Order Book Evolution at Step {step}")
            ax.legend()

        # Create animation
     #   ani = animation.FuncAnimation(
    #        fig, update, frames=valid_steps, init_func=init, interval=200, repeat=False
    #    )
    #    ani.save(output_file, writer="imagemagick")
     #   plt.close()
     #   print(f"Animation saved to {output_file}")


    # Save animation
   # animate_order_book_evolution(
    #    sorted_ask_raw_orders_history,
     #   sorted_bid_raw_orders_history,
      #  valid_steps,
      #  os.path.join(output_dir, "order_book_evolution.gif"),
    #)


    def save_last_order_book_frame(sorted_ask_history, sorted_bid_history, valid_steps, output_file):
        """
        Save the last frame of the cumulative sum graph as a PNG file.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # Extract data for the last step
        step = valid_steps 
        ask_prices = sorted_ask_history[step][:, 0]
        ask_quantities = sorted_ask_history[step][:, 1].cumsum()
        bid_prices = sorted_bid_history[step][:, 0]
        bid_quantities = sorted_bid_history[step][:, 1].cumsum()

        # Update axis limits dynamically
        ax.set_xlim(min(bid_prices.min(), ask_prices.min()), max(bid_prices.max(), ask_prices.max()))
        ax.set_ylim(0, max(ask_quantities.max(), bid_quantities.max()) * 1.1)

        # Plot bids
        ax.fill_between(
            bid_prices, 0, bid_quantities, step="pre", color="green", alpha=0.5, label="Bid Depth"
        )

        # Plot asks
        ax.fill_between(
            ask_prices, 0, ask_quantities, step="pre", color="red", alpha=0.5, label="Ask Depth"
        )

        # Add vertical dashed line
        ax.axvline(x=max(bid_prices), color="black", linestyle="--", linewidth=1)

        # Set labels and legend
        ax.set_xlabel("Price")
        ax.set_ylabel("Cumulative Quantity")
        ax.set_title(f"Limit Order Book Evolution at Step {step}")
        ax.legend()

        # Save the last frame as a PNG file
        plt.savefig(output_file, format="png")
        plt.close()
        print(f"Last frame saved to {output_file}")


 


    # 2. Save the last frame as PNG
  #  save_last_order_book_frame(
   #     sorted_ask_raw_orders_history,
   #     sorted_bid_raw_orders_history,
   #     valid_steps,
   #     os.path.join(output_dir, "last_order_book_frame.png")
   # )



