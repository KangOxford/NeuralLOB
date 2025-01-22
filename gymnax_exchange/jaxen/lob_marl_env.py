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






def step_env(
    self, key: chex.PRNGKey, state: EnvState, input_actions: jax.Array, params: EnvParams
) -> Tuple[Dict[str, jax.Array], EnvState, Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
    """
    JAX-compatible Multi-Agent Environment Step with Agent-Specific Inventory Updates.
    """
    num_agents = input_actions.shape[0]/self.num_actions
    agent_ids = jnp.arange(num_agents)

    # Sample order of agents
    key, subkey = jax.random.split(key)
    sampled_order = jax.random.permutation(subkey, agent_ids)

    # Initialize the message queue
    total_messages = jnp.zeros((0, 6))  

    def process_agent(carry, agent_id):
        key, state, total_messages = carry

        # Generate a unique key for the agent
        agent_key = jax.random.fold_in(key, agent_id)

        # Reshape action and create messages
        agent_action = self._reshape_action(input_actions[agent_id], state, params, agent_key)
        action_msgs = self._getActionMsgs(agent_action, state, params)
        
        # Handle cancellations
        cnl_msg_bid = job.getCancelMsgs(
            state.bid_raw_orders,
            self.trader_unique_id + agent_id,
            self.n_actions // 2,
            1
        )
        cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.trader_unique_id + agent_id,
            self.n_actions // 2,
            -1
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        # Filter redundant messages
        filtered_action_msgs, filtered_cnl_msgs = self._filter_messages(action_msgs, cnl_msgs)

        # Append to the global message queue
        total_messages = jnp.concatenate([total_messages, filtered_cnl_msgs, filtered_action_msgs], axis=0)
        return (key, state, total_messages), None

    # Use lax.scan to process all agents in the sampled order
    (key, state, total_messages), _ = jax.lax.scan(
        process_agent, (key, state, total_messages), sampled_order
    )

    # Add external data messages
    data_messages = self._get_data_messages(
        params.message_data,
        state.start_index,
        state.step_counter,
        state.init_time[0] + params.episode_time
    )
    total_messages = jnp.concatenate([total_messages, data_messages], axis=0)

    # Process all messages through the order book
    trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
    (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
        total_messages,
        (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
        self.stepLines
    )

    # Forward-fill best prices if missing
    bestasks = self._ffill_best_prices(bestasks[-self.stepLines + 1:], state.best_asks[-1, 0])
    bestbids = self._ffill_best_prices(bestbids[-self.stepLines + 1:], state.best_bids[-1, 0])

    # Process each agent for rewards and stats, including inventory updates
    def process_agent_stats(agent_id):
        # Get agent-specific trades
        agent_trades = job.get_agent_trades(trades, self.trader_unique_id + agent_id)
        
        
        

        # Calculate reward and additional stats
        reward,  = self._get_reward(state, params, trades, bestasks, bestbids)

        updated_inventory = state.inventory[agent_id] + extras["new_inventory"]

        # Generate observation for the agent
        obs = self._get_obs(state, params)

        # Check termination condition for the agent
        done = self.is_terminal(state, params)

        # Construct the agent's info dictionary
        info = {
            "inventory": updated_inventory,
            "market_share": extras["market_share"],
            "PnL": extras["PnL"],
            "approx_realized_pnl": extras["approx_realized_pnl"],
            "approx_unrealized_pnl": extras["approx_unrealized_pnl"]
        }

        return obs, reward, done, info, updated_inventory

    # Vectorized processing for rewards, stats, and inventory updates
    agent_results = jax.vmap(process_agent_stats)(agent_ids)
    observations, rewards, dones, infos, updated_inventories = agent_results

    # Convert to dictionaries
    observations = {str(i): obs for i, obs in enumerate(observations)}
    rewards = {str(i): reward for i, reward in enumerate(rewards)}
    dones = {str(i): done for i, done in enumerate(dones)}
    infos = {str(i): info for i, info in enumerate(infos)}

    # Check if all agents are done
    dones["__all__"] = jnp.all(jnp.array(list(dones.values())))

    # Update shared environment state, including inventories
    state = EnvState(
        prev_action=input_actions,  # Store the actions taken by all agents
        ask_raw_orders=asks,
        bid_raw_orders=bids,
        trades=trades,
        init_time=state.init_time,
        time=state.time,
        customIDcounter=state.customIDcounter,
        window_index=state.window_index,
        step_counter=state.step_counter + 1,
        max_steps_in_episode=state.max_steps_in_episode,
        start_index=state.start_index,
        best_asks=bestasks,
        best_bids=bestbids,
        init_price=state.init_price,
        mid_price=state.mid_price,
        inventory=updated_inventories,  # Updated agent-specific inventories
        total_PnL=state.total_PnL,
        bid_passive_2=state.bid_passive_2,
        quant_bid_passive_2=state.quant_bid_passive_2,
        ask_passive_2=state.ask_passive_2,
        quant_ask_passive_2=state.quant_ask_passive_2,
        delta_time=state.delta_time
    )

    return observations, state, rewards, dones, infos

