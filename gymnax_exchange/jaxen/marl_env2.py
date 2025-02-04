"""
-----------
Multi-Agent Environment. 
The main idea is that both agents interact with one order book, but each
agent has its own sub-state, actions, rewards, and observations.
"""

import os
import sys
import time
import dataclasses
from functools import partial
from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct
import numpy as np

import gymnax
from gymnax.environments import environment, spaces

sys.path.append(os.path.abspath("/home/duser/AlphaTrade"))
sys.path.append(".")

from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv, EnvState as BaseEnvState, EnvParams as BaseEnvParams
from gymnax_exchange.utils import utils

import faulthandler
faulthandler.enable()
chex.assert_gpu_available(backend=None)
print("Num Jax Devices:", jax.device_count(), "Device List:", jax.devices())
jax.numpy.set_printoptions(linewidth=183)


@struct.dataclass
class EnvState(BaseEnvState):
    """
    Merged multi-agent state with sub-states for:
      1) Market Maker (mm_* fields)
      2) Execution (exe_* fields)
    plus the shared order book from BaseEnvState (inherited).
    """
    # ---------- The shared BaseEnvState fields (ask_raw_orders, etc.) ----------
    # Stuff from BaseEnvState: ask_raw_orders, bid_raw_orders, trades, etc.

    # -------------------- Market Maker sub-state --------------------
    mm_prev_action: chex.Array
    mm_prev_executed: chex.Array
    mm_best_asks: chex.Array
    mm_best_bids: chex.Array
    mm_init_price: int
    mm_inventory: int
    mm_mid_price: int
    mm_total_PnL: float
    mm_bid_passive_2: int
    mm_quant_bid_passive_2: int
    mm_ask_passive_2: int
    mm_quant_ask_passive_2: int
    mm_delta_time: float

    # -------------------- Execution sub-state --------------------
    exe_prev_action: chex.Array
    exe_prev_executed: chex.Array
    exe_best_asks: chex.Array
    exe_best_bids: chex.Array
    exe_init_price: int
    exe_task_to_execute: int
    exe_quant_executed: int
    exe_total_revenue: float
    exe_drift_return: float
    exe_advantage_return: float
    exe_slippage_rm: float
    exe_price_adv_rm: float
    exe_price_drift_rm: float
    exe_vwap_rm: float
    exe_is_sell_task: int
    exe_trade_duration: float
    exe_price_passive_2: int
    exe_quant_passive_2: int
    exe_delta_time: float


@struct.dataclass
class EnvParams(BaseEnvParams):
    mm_reward_lambda: float = 0.0001
    exe_reward_lambda: float = 1.0
    exe_task_size: int = 100

    mm_action_type: str = "pure"         
    mm_n_ticks_in_book: int = 2         
    mm_max_task_size: int = 500           


class MARLEnv(BaseLOBEnv):
    """
    Multi-Agent environment that combines:
      - a Market Maker (MM)
      - a Order Execution Agent (EXE)

    Both agents share a single order book, but have separate sub-states, actions, and rewards.
    """

    def __init__(
        self,
        alphatradePath: str,
        window_index: int,
        episode_time: int,
        ep_type: str = "fixed_time",
        # Additional multi-agent arguments:
        mm_trader_id: int = 1111,
        exe_trader_id: int = 2222,
        mm_reward_lambda: float = 0.0001,
        exe_reward_lambda: float = 1.0,
        exe_task_size: int = 100,

        mm_action_type: str = "pure",
        mm_n_ticks_in_book: int = 2,
        mm_max_task_size: int = 500,
    ):
        super().__init__(
            alphatradePath,
            window_index,    
            episode_time,     
            ep_type=ep_type
        )

        # Basic environment info
        self.window_index = window_index
        self.episode_time = episode_time
        self.ep_type      = ep_type

        # Market Maker config
        self.mm_trader_id        = mm_trader_id
        self.mm_reward_lambda    = mm_reward_lambda
        self.mm_action_type      = mm_action_type
        self.mm_n_ticks_in_book  = mm_n_ticks_in_book
        self.mm_max_task_size    = mm_max_task_size
        self.mm_n_actions        = 6  

        # Execution config
        self.exe_trader_id       = exe_trader_id
        self.exe_reward_lambda   = exe_reward_lambda
        self.exe_task_size       = exe_task_size
        self.exe_n_actions       = 2  

    def default_params(self) -> EnvParams:
        base_params = super().default_params
        return dataclasses.replace(
            base_params
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams
    ) -> Tuple[Dict[str, jnp.ndarray], EnvState]:
        obs_base, base_state = super().reset_env(key, params)

        init_state = EnvState(
            ask_raw_orders      = base_state.ask_raw_orders,
            bid_raw_orders      = base_state.bid_raw_orders,
            trades             = base_state.trades,
            init_time          = base_state.init_time,
            time               = base_state.time,
            customIDcounter    = base_state.customIDcounter,
            window_index       = base_state.window_index,
            step_counter       = base_state.step_counter,
            max_steps_in_episode= base_state.max_steps_in_episode,
            start_index        = base_state.start_index,
            
            # -------------------- Market Maker sub-state --------------------
            mm_prev_action     = jnp.zeros((self.mm_n_actions, 2), jnp.int32),
            mm_prev_executed   = jnp.zeros((self.mm_n_actions, 2), jnp.int32),
            mm_best_asks       = jnp.zeros((self.stepLines, 2), jnp.int32),
            mm_best_bids       = jnp.zeros((self.stepLines, 2), jnp.int32),
            mm_init_price      = 0,
            mm_inventory       = 0,
            mm_mid_price       = 0,
            mm_total_PnL       = 0.0,
            mm_bid_passive_2   = 0,
            mm_quant_bid_passive_2= 0,
            mm_ask_passive_2   = 0,
            mm_quant_ask_passive_2= 0,
            mm_delta_time      = 0.0,

            # -------------------- Execution sub-state --------------------
            exe_prev_action    = jnp.zeros((self.exe_n_actions, 2), jnp.int32),
            exe_prev_executed  = jnp.zeros((self.exe_n_actions,), jnp.int32),
            exe_best_asks      = jnp.zeros((self.stepLines, 2), jnp.int32),
            exe_best_bids      = jnp.zeros((self.stepLines, 2), jnp.int32),
            exe_init_price     = 0,
            exe_task_to_execute= self.exe_task_size,
            exe_quant_executed = 0,
            exe_total_revenue  = 0.0,
            exe_drift_return   = 0.0,
            exe_advantage_return=0.0,
            exe_slippage_rm    = 0.0,
            exe_price_adv_rm   = 0.0,
            exe_price_drift_rm = 0.0,
            exe_vwap_rm        = 0.0,
            exe_is_sell_task   = 0,
            exe_trade_duration = 0.0,
            exe_price_passive_2= 0,
            exe_quant_passive_2= 0,
            exe_delta_time     = 0.0,
        )

        # Set initial best ask/bid for both agents
        best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(
            init_state.ask_raw_orders, 
            init_state.bid_raw_orders
        )
        mid_p = (best_ask[0] + best_bid[0]) // 2

        init_state = init_state.replace(
            mm_best_asks=jnp.resize(best_ask, (self.stepLines,2)),
            mm_best_bids=jnp.resize(best_bid, (self.stepLines,2)),
            mm_init_price=mid_p,
            mm_mid_price=mid_p,

            exe_best_asks=jnp.resize(best_ask, (self.stepLines,2)),
            exe_best_bids=jnp.resize(best_bid, (self.stepLines,2)),
            exe_init_price=mid_p
        )

        # Build separate agent observations
        obs_dict = {
            "market_maker": self._get_obs_mm(init_state, params),
            "execution":    self._get_obs_exe(init_state, params)
        }
        return obs_dict, init_state
    

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        actions: Dict[str, jnp.ndarray],
        params: EnvParams
    ) -> Tuple[Dict[str, jnp.ndarray], EnvState, Dict[str, float], bool, Dict[str, Dict]]:
    
        ############################################################################
        # (A) 1) Gather external data messages
        ############################################################################
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )
        
        ############################################################################
        # (B) Market Maker: build new order msgs, cancellations
        ############################################################################
        mm_input_action = actions["market_maker"]
        mm_action = self._reshape_action_mm(mm_input_action, state, params, key)
        mm_action_msgs = self._getActionMsgs_mm(mm_action, state, params)
        mm_action_prices = mm_action_msgs[:, 3]

        # Cancel all previous MM orders each step
        mm_cnl_msg_bid = job.getCancelMsgs(
            state.bid_raw_orders,
            self.mm_trader_id,
            self.mm_n_actions // 2,  # half for bids
            1
        )
        mm_cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.mm_trader_id,
            self.mm_n_actions // 2,
            -1
        )
        mm_cnl_msgs = jnp.concatenate([mm_cnl_msg_bid, mm_cnl_msg_ask], axis=0)

        # Net action & cancellation if new action not bigger than cancellation
        mm_action_msgs, mm_cnl_msgs = self._filter_messages_mm(mm_action_msgs, mm_cnl_msgs)

        ############################################################################
        # (C) Execution logic: build new order msgs, cancellations
        ############################################################################
        exe_input_action = actions["execution"]
        exe_action = self._reshape_action_exe(exe_input_action, state, params, key)
        exe_action_msgs = self._getActionMsgs_exe(exe_action, state, params)
        exe_action_prices = exe_action_msgs[:, 3]

        side_for_cancels = 1 - state.exe_is_sell_task * 2
        raw_order_side = jax.lax.cond(
            state.exe_is_sell_task == 1,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        exe_cnl_msgs = job.getCancelMsgs(
            raw_order_side,
            self.exe_trader_id,
            self.exe_n_actions,
            side_for_cancels
        )
        # Net action & cancellation
        exe_action_msgs, exe_cnl_msgs = self._filter_messages_exe(exe_action_msgs, exe_cnl_msgs)

        ############################################################################
        # (D) Combine all agent messages + data, then do a single main LOB scan
        ############################################################################
        combined_agent_msgs = jnp.concatenate([
            mm_cnl_msgs, mm_action_msgs,
            exe_cnl_msgs, exe_action_msgs
        ], axis=0)

        # Combine with external data
        total_messages = jnp.concatenate([combined_agent_msgs, data_messages], axis=0)

        # Determine time from last message
        time_final = total_messages[-1, -2:]

        # single pass
        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        )

        # Forward-fill best prices (for the last stepLines)
        bestasks = self._ffill_best_prices_mm(
            bestasks[-self.stepLines+1:],  # or _ffill_best_prices_exe
            state.mm_best_asks[-1, 0]      # uses old mm's best_asks?
        )
        bestbids = self._ffill_best_prices_mm(
            bestbids[-self.stepLines+1:],
            state.mm_best_bids[-1, 0]
        )

        ############################################################################
        # (E) 'Force Market Order if Done' for each agent
        ############################################################################

        # 1) Market Maker forced flatten
        (asks, bids, trades), (mm_new_bestask, mm_new_bestbid), mm_new_id_counter, mm_new_time, mm_mkt_exec_q, mm_doom_quant = \
            self._force_market_order_if_done_mm(
                bestasks[-1],  # last best ask
                bestbids[-1],  # last best bid
                time_final,    # "time"
                asks, bids, trades,
                state,         
                params
            )
        #  extend best asks/bids arrays
        bestasks = jnp.concatenate([bestasks, jnp.resize(mm_new_bestask, (1,2))], axis=0)
        bestbids = jnp.concatenate([bestbids, jnp.resize(mm_new_bestbid, (1,2))], axis=0)

        # 2) Execution forced flatten or forced final order
        quant_executed_so_far = state.exe_quant_executed  # or we compute after we find trades
        (asks, bids, trades), (exe_new_bestask, exe_new_bestbid), exe_new_id_ctr, exe_new_time, exe_mkt_exec_q, exe_doom_quant = \
            self._force_market_order_if_done_exe(
                quant_executed_so_far,  
                bestasks[-1],
                bestbids[-1],
                mm_new_time,   
                asks, bids, trades,
                state,
                params
            )
        bestasks = jnp.concatenate([bestasks, jnp.resize(exe_new_bestask, (1,2))], axis=0)
        bestbids = jnp.concatenate([bestbids, jnp.resize(exe_new_bestbid, (1,2))], axis=0)

        # final time, id counter after both forced steps
        final_time = exe_new_time
        final_id_ctr = exe_new_id_ctr

        ############################################################################
        # (F) Split out each agent’s trades, do `_get_executed_by_action` & `_get_reward`
        ############################################################################
        # 1) Market Maker
        mm_agent_trades = job.get_agent_trades(trades, self.mm_trader_id) #TODO: could change the function to add an array or dict here
        mm_executions = self._get_executed_by_action(
            mm_agent_trades, mm_action, state, mm_action_prices
        )
        mm_reward, mm_extras = self._get_reward_mm(
            state,
            params,
            trades,
            bestasks,
            bestbids
        )
        mm_bid_passive_2, mm_quant_bid_passive_2, mm_ask_passive_2, mm_quant_ask_passive_2 = \
            self._get_pass_price_quant_mm(state)

        # 2) Execution
        exe_agent_trades = job.get_agent_trades(trades, self.exe_trader_id)
        exe_executions = self._get_executed_by_action(
            exe_agent_trades, exe_action, state, exe_action_prices
        )
        exe_reward, exe_extras = self._get_reward_exe(
            state,
            params,
            trades
        )
        # e.g. quant_executed_this_step, etc. from single-agent code
        quant_executed_this_step = exe_executions.sum()
        new_exe_quant_executed = state.exe_quant_executed + exe_extras["agentQuant"]

        # trade duration step from single-agent code
        trade_duration_step = (
            jnp.abs(exe_agent_trades[:, 1])
            / state.task_to_execute
            * (exe_agent_trades[:, -2] - state.init_time[0])
        ).sum()
        new_exe_trade_duration = state.exe_trade_duration + trade_duration_step

        exe_price_passive_2, exe_quant_passive_2 = self._get_pass_price_quant_exe(state)

        ############################################################################
        # (G) Build the updated environment state with sub-states for both agents
        ############################################################################
        # For Market Maker sub-state:
        new_mm_inventory = mm_extras["end_inventory"]
        new_mm_totalPnL  = state.mm_total_PnL + mm_extras["PnL"]

        # For Execution sub-state:
        new_exe_total_revenue = state.exe_total_revenue + exe_extras["revenue"]

        final_state = state.replace(
            # Shared order book/trades
            ask_raw_orders = asks,
            bid_raw_orders = bids,
            trades = trades,
            time = final_time,
            customIDcounter = final_id_ctr,
            best_asks = bestasks,
            best_bids = bestbids,

            # Market Maker sub-state
            mm_prev_action = jnp.vstack([mm_action_prices, mm_action]).T,
            mm_prev_executed = mm_executions,
            mm_inventory = new_mm_inventory,
            mm_total_PnL = new_mm_totalPnL,
            mm_best_asks = bestasks,  
            mm_best_bids = bestbids,
            mm_mid_price = mm_extras["mid_price"],
            bid_passive_2 = mm_bid_passive_2,
            quant_bid_passive_2 = mm_quant_bid_passive_2,
            ask_passive_2 = mm_ask_passive_2,
            quant_ask_passive_2 = mm_quant_ask_passive_2,

            # Execution sub-state
            exe_prev_action = jnp.vstack([exe_action_prices, exe_action]).T,
            exe_prev_executed = exe_executions,
            exe_quant_executed = new_exe_quant_executed,
            exe_total_revenue = new_exe_total_revenue,
            drift_return = state.drift_return + exe_extras["drift"],
            advantage_return = state.advantage_return + exe_extras["advantage"],
            slippage_rm = exe_extras["slippage_rm"],
            price_adv_rm = exe_extras["price_adv_rm"],
            price_drift_rm = exe_extras["price_drift_rm"],
            vwap_rm = exe_extras["vwap_rm"],
            exe_trade_duration = new_exe_trade_duration,
            exe_price_passive_2 = exe_price_passive_2,
            exe_quant_passive_2 = exe_quant_passive_2,

            # Step/time
            step_counter = state.step_counter + 1,
            # e.g. from single-agent code for delta_time
            delta_time = final_time[0] + final_time[1]/1e9 - state.time[0] - state.time[1]/1e9
        )

        done = self.is_terminal(final_state, params)

        ############################################################################
        # (H) Build Observations, Rewards, Info
        ############################################################################
        obs_dict = {
            "market_maker": self._get_obs_mm(final_state, params),
            "execution":    self._get_obs_exe(final_state, params)
        }
        reward_dict = {
            "market_maker": float(mm_reward),
            "execution":    float(exe_reward)
        }
        info_dict = {
            "market_maker": {
                "reward": mm_reward,
                "inventory": new_mm_inventory,
                "mkt_forced_quant": mm_mkt_exec_q + mm_doom_quant,
                "doom_quant": mm_doom_quant,
                "PnL": new_mm_totalPnL,
                # add all the stuff later
            },
            "execution": {
                "reward": exe_reward,
                "quant_executed": new_exe_quant_executed,
                "mkt_forced_quant": exe_mkt_exec_q + exe_doom_quant,
                "doom_quant": exe_doom_quant,
                "total_revenue": new_exe_total_revenue,
                # add all the stuff later
            },
            "__done__": done
        }

        return obs_dict, final_state, reward_dict, done, info_dict



    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Condition merges 'time done' for MM and 'task done' for Execution.
        """
        if self.ep_type == "fixed_time":
            done_time = (params.episode_time - (state.time - state.init_time)[0] <= 5)
            exe_done  = (state.exe_task_to_execute - state.exe_quant_executed <= 0)
            return bool(done_time or exe_done)
        elif self.ep_type == "fixed_steps":
            done_steps= (state.max_steps_in_episode - state.step_counter <= 1)
            exe_done  = (state.exe_task_to_execute - state.exe_quant_executed <= 0)
            return bool(done_steps or exe_done)
        else:
            raise ValueError(f"Unknown ep_type: {self.ep_type}")

    # =========================================================================
    # Market Maker: actions, reward, obs
    # =========================================================================
    def _getActionMsgs_mm(
        self,
        mm_action: jnp.ndarray,
        state: EnvState,
        params: EnvParams,
        key: chex.PRNGKey
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Builds the market maker's order + cancellation messages, similar to single-agent logic.
        """
        shaped_action = self._reshape_action_mm(mm_action, state, params, key)
        action_msgs = self._build_mm_order_msgs(shaped_action, state, params, key)

        # Cancel old orders from MM
        cnl_msg_bid = job.getCancelMsgs(
            state.bid_raw_orders,
            self.mm_trader_id,
            self.mm_n_actions // 2,  # half for bids
            1
        )
        cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.mm_trader_id,
            self.mm_n_actions // 2,
            -1
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        # Optionally filter partial cancellations
        action_msgs, cnl_msgs = self._filter_messages_mm(action_msgs, cnl_msgs)
        return action_msgs, cnl_msgs

    def _reshape_action_mm(self, mm_action: jax.Array, state: EnvState, params: EnvParams, key: chex.PRNGKey) -> jax.Array:
        """
        If mm_action_type is 'delta', do a twap-like approach, etc.
        Otherwise (pure), just return mm_action as is.
        """
        if params.mm_action_type == "delta":
            remaining_steps = state.max_steps_in_episode - state.step_counter
            step_quant = jnp.ceil(state.mm_inventory / jnp.maximum(remaining_steps, 1)).astype(jnp.int32)
            twap_delta = jax.random.permutation(key, jnp.array([step_quant//2, step_quant//2]))
            mm_action = mm_action + twap_delta
        return mm_action

    def _build_mm_order_msgs(self, mm_action: jax.Array, state: EnvState, params: EnvParams, key: chex.PRNGKey) -> jnp.ndarray:
        """
        Convert mm_action into actual LOB messages (side=buy/sell, quantity, price, etc.).
        For instance, half are bids around best_bid, half are asks around best_ask.
        """
        n_actions = self.mm_n_actions
        types = jnp.ones((n_actions,), jnp.int32)

        # half are buys, half are sells
        sides_bids = jnp.ones((n_actions // 2,), jnp.int32)
        sides_asks = -jnp.ones((n_actions // 2,), jnp.int32)
        sides = jnp.concatenate([sides_bids, sides_asks])

        trader_ids = jnp.ones((n_actions,), jnp.int32) * self.mm_trader_id
        order_ids = (
            (jnp.ones((n_actions,), jnp.int32) * (self.mm_trader_id + state.customIDcounter))
            + jnp.arange(n_actions)
        )

        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (n_actions, 2)
        )

        # Example: measure best_ask, best_bid from the last recorded array
        best_ask_price = state.mm_best_asks[-1, 0]
        best_bid_price = state.mm_best_bids[-1, 0]

        # Construct an example price array
        # Just as example: 3 buy prices around best_bid, 3 sell around best_ask
        buy_prices = jnp.array([
            best_bid_price,
            best_bid_price - params.mm_n_ticks_in_book * self.tick_size,
            best_bid_price + self.tick_size
        ], dtype=jnp.int32)
        sell_prices = jnp.array([
            best_ask_price,
            best_ask_price + params.mm_n_ticks_in_book * self.tick_size,
            best_ask_price - self.tick_size
        ], dtype=jnp.int32)
        all_prices = jnp.concatenate([buy_prices, sell_prices])

        quants = mm_action.astype(jnp.int32).clip(0, params.mm_max_task_size)

        # Build (type, side, quant, price, trader_id, order_id, time_s, time_ns)
        action_msgs = jnp.stack([types, sides, quants, all_prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)

        return action_msgs

    def _filter_messages_mm(self, action_msgs: jnp.ndarray, cnl_msgs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        #TODO partial-cancellation or net-out logic from single-agent code.
        
        """
        return action_msgs, cnl_msgs

    def _get_reward_mm(
        self,
        state: EnvState,
        params: EnvParams,
        trades: jnp.ndarray,
        mm_agent_trades: jnp.ndarray,
        bestasks: jnp.ndarray,
        bestbids: jnp.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        #TODO implement
        """
     
        pass



    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
            if self.ep_type == "fixed_time":
                done_time = (params.episode_time - (state.time - state.init_time)[0] <= 5)
                exe_done  = (state.exe_task_to_execute - state.exe_quant_executed <= 0)
                return bool(done_time or exe_done)
            elif self.ep_type == "fixed_steps":
                done_steps= (state.max_steps_in_episode - state.step_counter <= 1)
                exe_done  = (state.exe_task_to_execute - state.exe_quant_executed <= 0)
                return bool(done_steps or exe_done)
            else:
                raise ValueError(f"Unknown ep_type: {self.ep_type}")

    # =======================
    # Market Maker side
    # =======================
    def _get_pass_price_quant_mm(self, state: EnvState) -> Tuple[int, int, int, int]:
        n_ticks_in_book = getattr(self, "mm_n_ticks_in_book", 2)

        bid_passive_2 = state.mm_best_bids[-1, 0] - self.tick_size * n_ticks_in_book
        ask_passive_2 = state.mm_best_asks[-1, 0] + self.tick_size * n_ticks_in_book
        quant_bid_passive_2 = job.get_volume_at_price(state.bid_raw_orders, bid_passive_2)
        quant_ask_passive_2 = job.get_volume_at_price(state.ask_raw_orders, ask_passive_2)
        return bid_passive_2, quant_bid_passive_2, ask_passive_2, quant_ask_passive_2

    def _filter_messages_mm(self, action_msgs: jax.Array, cnl_msgs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        @partial(jax.vmap, in_axes=(0, None))
        def p_in_cnl(p, prices_cnl):
            return jnp.where((prices_cnl == p) & (p != 0), True, False)

        def matching_masks(prices_a, prices_cnl):
            res = p_in_cnl(prices_a, prices_cnl)
            return jnp.any(res, axis=1), jnp.any(res, axis=0)

        @jax.jit
        def argsort_rev(arr):
            """ 'arr' sorted in descending order (LTR tie-breaker) """
            return (arr.shape[0] - 1 - jnp.argsort(arr[::-1]))[::-1]

        @jax.jit
        def rank_rev(arr):
            """ Rank array in descending order, ties resolved left-to-right """
            return jnp.argsort(argsort_rev(arr))

        a_mask, c_mask = matching_masks(action_msgs[:, 3], cnl_msgs[:, 3])
        a_i = jnp.where(a_mask, size=a_mask.shape[0], fill_value=-1)[0]
        c_i = jnp.where(c_mask, size=c_mask.shape[0], fill_value=-1)[0]

        a = jnp.where(a_i == -1, 0, action_msgs[a_i][:, 2])
        c = jnp.where(c_i == -1, 0, cnl_msgs[c_i][:, 2])

        rel_cnl_quants = (c >= a) * a
        # Subtract from original arrays
        action_msgs = action_msgs.at[:, 2].set(
            action_msgs[:, 2] - rel_cnl_quants[rank_rev(a_mask)]
        )
        cnl_msgs = cnl_msgs.at[:, 2].set(
            cnl_msgs[:, 2] - rel_cnl_quants[rank_rev(c_mask)]
        )
        # Zero out any action msgs that are now 0
        action_msgs = jnp.where(
            (action_msgs[:, 2] == 0).T,
            0,
            action_msgs.T
        ).T
        return action_msgs, cnl_msgs

    def _ffill_best_prices_mm(self, prices_quants: jax.Array, last_valid_price: int) -> jax.Array:
        def ffill(arr, inval=-1):
            def f(prev, x):
                new = jnp.where(x != inval, x, prev)
                return (new, new)
            _, out = jax.lax.scan(f, inval, arr)
            return out

        # if first new price is invalid
        prices_quants = prices_quants.at[0, 0:2].set(
            jnp.where(
                (prices_quants[0, 0] == -1),
                jnp.array([last_valid_price, 0]),
                prices_quants[0, 0:2]
            )
        )
        prices_quants = prices_quants.at[:, 1].set(
            jnp.where(prices_quants[:, 0] == -1, 0, prices_quants[:, 1])
        )
        # forward fill
        prices_quants = prices_quants.at[:, 0].set(ffill(prices_quants[:, 0]))
        return prices_quants


    # =========================================================================
    # Execution Agent: actions, reward, obs
    # =========================================================================
    def _getActionMsgs_exe(
        self,
        exe_action: jnp.ndarray,
        state: EnvState,
        params: EnvParams,
        key: chex.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        shaped_action = self._reshape_action_exe(exe_action, state, params, key)
        action_msgs = self._build_exe_order_msgs(shaped_action, state, params, key)

        side = jnp.where(state.exe_is_sell_task == 1, -1, 1)
        if side == 1:
            cnl_msgs = job.getCancelMsgs(state.bid_raw_orders, self.exe_trader_id, self.exe_n_actions, side)
        else:
            cnl_msgs = job.getCancelMsgs(state.ask_raw_orders, self.exe_trader_id, self.exe_n_actions, side)

        action_msgs, cnl_msgs = self._filter_messages_exe(action_msgs, cnl_msgs)
        return action_msgs, cnl_msgs

    def _reshape_action_exe(self, exe_action: jnp.ndarray, state: EnvState, params: EnvParams, key: chex.PRNGKey) -> jnp.ndarray:
        # Minimal example. Real code would replicate ExecutionEnv logic
        return exe_action

    def _build_exe_order_msgs(self, exe_action: jnp.ndarray, state: EnvState, params: EnvParams, key: chex.PRNGKey) -> jnp.ndarray:
        n_actions = self.exe_n_actions
        side = jnp.where(state.exe_is_sell_task == 1, -1, 1)

        best_ask = jnp.int32((state.exe_best_asks[-1, 0]) // self.tick_size * self.tick_size)
        best_bid = jnp.int32((state.exe_best_bids[-1, 0]) // self.tick_size * self.tick_size)

        market_price = jnp.where(side == 1, best_ask, best_bid)
        offset = jnp.arange(n_actions, dtype=jnp.int32) * self.tick_size

        types = jnp.ones((n_actions,), jnp.int32)
        sides = jnp.ones((n_actions,), jnp.int32) * side
        quants= exe_action.astype(jnp.int32)
        prices= market_price + offset*jnp.where(side==1, -1, +1)

        trader_ids = jnp.ones((n_actions,), jnp.int32)*self.exe_trader_id
        order_ids  = (jnp.ones((n_actions,), jnp.int32)*(self.exe_trader_id + state.customIDcounter)) + jnp.arange(n_actions)
        times = jnp.resize(state.time + params.time_delay_obs_act, (n_actions,2))

        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        return action_msgs

    def _filter_messages_exe(self, action_msgs: jnp.ndarray, cnl_msgs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return action_msgs, cnl_msgs

    def _get_reward_exe(
        self,
        state: EnvState,
        params: EnvParams,
        trades: jnp.ndarray,
        exe_agent_trades: jnp.ndarray,
        bestasks: jnp.ndarray,
        bestbids: jnp.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        executed_qty = jnp.abs(exe_agent_trades[:,1]).sum()
        new_quant_executed = state.exe_quant_executed + executed_qty
        reward = float(executed_qty) * 0.01 * float(jnp.sign(1 - state.exe_is_sell_task*2))

        info_dict = {
            "quant_executed": float(new_quant_executed),
            "total_revenue": float(state.exe_total_revenue),
        }
        return reward, info_dict

    # =========================================================================
    # _update_state: unify new sub-state after each step
    # =========================================================================
    def _update_state(
        self,
        old_state: EnvState,
        asks: jnp.ndarray,
        bids: jnp.ndarray,
        trades: jnp.ndarray,
        bestasks: jnp.ndarray,
        bestbids: jnp.ndarray,
        mm_info: Dict[str, float],
        exe_info: Dict[str, float],
        params: EnvParams
    ) -> EnvState:
        final_best_ask = bestasks[-1]
        final_best_bid = bestbids[-1]

        # Market Maker updates
        new_mm_inventory = mm_info.get("inventory", old_state.mm_inventory)
        new_mm_totalPnL  = old_state.mm_total_PnL  # or mm_info.get(...)

        # Execution updates
        new_exe_executed = exe_info.get("quant_executed", old_state.exe_quant_executed)
        new_exe_revenue  = old_state.exe_total_revenue  # or exe_info.get(...)

        # Time logic
        new_time = old_state.time 

        # Bump step
        new_step = old_state.step_counter + 1

        return old_state.replace(
            ask_raw_orders = asks,
            bid_raw_orders = bids,
            trades = trades,
            step_counter = new_step,
            time = new_time,

            mm_best_asks = bestasks,
            mm_best_bids = bestbids,
            mm_inventory = new_mm_inventory,
            mm_total_PnL = new_mm_totalPnL,

            exe_best_asks = bestasks,
            exe_best_bids = bestbids,
            exe_quant_executed = new_exe_executed,
            exe_total_revenue  = new_exe_revenue
        )


    def _get_obs_full(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # Note: uses entire observation history between steps
        # TODO: if we want to use this, we need to roll forward the RNN state with every step

        best_asks, best_bids = state.best_asks[:,0], state.best_bids[:,0]
        best_ask_qtys, best_bid_qtys = state.best_asks[:,1], state.best_bids[:,1]
        
        obs = {
            #"is_sell_task": state.is_sell_task,
            "p_aggr": jnp.where(state.is_sell_task, best_bids, best_asks),
            "q_aggr": jnp.where(state.is_sell_task, best_bid_qtys, best_ask_qtys), 
            "p_pass": jnp.where(state.is_sell_task, best_asks, best_bids),
            "q_pass": jnp.where(state.is_sell_task, best_ask_qtys, best_bid_qtys), 
            "p_mid": (best_asks+best_bids)//2//self.tick_size*self.tick_size, 
            "p_pass2": jnp.where(state.is_sell_task, best_asks+self.tick_size*self.n_ticks_in_book, best_bids-self.tick_size*self.n_ticks_in_book), # second_passives
            "spread": best_asks - best_bids,
            "shallow_imbalance": state.best_asks[:,1]- state.best_bids[:,1],
            "time": state.time,
            "episode_time": state.time - state.init_time,
            "init_price": state.init_price,
           # "task_size": state.task_to_execute,
           # "executed_quant": state.quant_executed,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
        }
        p_mean = 3.5e7
        p_std = 1e6
        means = {
            #"is_sell_task": 0,
            "p_aggr": p_mean,
            "q_aggr": 0,
            "p_pass": p_mean,
            "q_pass": 0,
            "p_mid": p_mean,
            "p_pass2":p_mean,
            "spread": 0,
            "shallow_imbalance":0,
            "time": jnp.array([0, 0]),
            "episode_time": jnp.array([0, 0]),
            "init_price": p_mean,
            "task_size": 0,
           # "executed_quant": 0,
            "step_counter": 0,
            "max_steps": 0,
        }
        stds = {
            #"is_sell_task": 1,
            "p_aggr": p_std,
            "q_aggr": 100,
            "p_pass": p_std,
            "q_pass": 100,
            "p_mid": p_std,
            "p_pass2": p_std,   
            "spread": 1e4,
            "shallow_imbalance": 10,
            "time": jnp.array([1e5, 1e9]),
            "episode_time": jnp.array([1e3, 1e9]),
            "init_price": p_std,
            "task_size": 500,
          #  "executed_quant": 500,
            "step_counter": 300,
            "max_steps": 300,
        }
        obs = self.normalize_obs(obs, means, stds)
        obs, _ = jax.flatten_util.ravel_pytree(obs)
        return obs

    def normalize_obs(
            self,
            obs: Dict[str, jax.Array],
            means: Dict[str, jax.Array],
            stds: Dict[str, jax.Array]
        ) -> Dict[str, jax.Array]:
        """ normalized observation by substracting 'mean' and dividing by 'std'
            (config values don't need to be actual mean and std)
        """
        obs = jax.tree_map(lambda x, m, s: (x - m) / s, obs, means, stds)
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """ Action space of the environment. """
        if self.action_type == 'delta':
            # return spaces.Box(-5, 5, (self.n_actions,), dtype=jnp.int32)
            return spaces.Box(-100, 100, (self.n_actions,), dtype=jnp.int32)
        else:
            # return spaces.Box(0, 100, (self.n_actions,), dtype=jnp.int32)
            return spaces.Box(0, self.max_task_size, (self.n_actions,), dtype=jnp.int32)
    
       

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        #space = spaces.Box(-10,10,(809,),dtype=jnp.float32) 
        # space = spaces.Box(-10, 10, (21,), dtype=jnp.float32) 
        space = spaces.Box(-10, 10, (25,), dtype=jnp.float32) 
        return space

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return NotImplementedError















# =============================================================================
# =============================================================================
# =============================   MAIN  =======================================
# =============================================================================
# =============================================================================

if __name__ == "__main__":
    import sys
    import time
    import dataclasses
    import jax
    import jax.numpy as jnp

    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        # Provide a default folder or fallback
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
        print("No folder arg specified. Using default:", ATFolder)

    config = {
        "ATFOLDER": ATFolder,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 240,
        # For the Market Maker
        "MM_TRADER_ID": 1111,
        "MM_REWARD_LAMBDA": 0.0001,
        "MM_N_ACTIONS": 6, 
        # For Execution
        "EXE_TRADER_ID": 2222,
        "EXE_REWARD_LAMBDA": 1.0,
        "EXE_TASK_SIZE": 100,
    }

    # Create RNG keys
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)


    # Instantiate the multi-agent env
    env = MARLEnv(
        alphatradePath=config["ATFOLDER"],
        window_index=100,       # example window index
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        mm_trader_id=config["MM_TRADER_ID"],
        exe_trader_id=config["EXE_TRADER_ID"],
        mm_reward_lambda=config["MM_REWARD_LAMBDA"],
        exe_reward_lambda=config["EXE_REWARD_LAMBDA"],
        exe_task_size=config["EXE_TASK_SIZE"],
    )

    # 3) Create the default params
    env_params = env.default_params()
    env_params = dataclasses.replace(
        env_params,
        episode_time=config["EPISODE_TIME"],
    )

    # 4) Reset the environment
    start = time.time()
    obs_dict, state = env.reset_env(key_reset, env_params)
    print("Time for reset:", time.time() - start)

    print("Observation (market_maker):", obs_dict["market_maker"])
    print("Observation (execution):",    obs_dict["execution"])

    # 5) Step in a loop with random actions for each agent
    for i in range(1, 20):
        print("-" * 30)
        print(f"Step {i}...")

        key_policy, sub1 = jax.random.split(key_policy)
        key_step, sub2   = jax.random.split(key_step)

        # Example: Market Maker has mm_n_actions=6
        mm_random_action = jax.random.randint(sub1, (env.mm_n_actions,), 0, 100)
        # Example: Execution has exe_n_actions=2
        exe_random_action= jax.random.randint(sub2, (env.exe_n_actions,), 0, 50)

        actions = {
            "market_maker": mm_random_action,
            "execution":    exe_random_action
        }

        # Step the environment
        start_step = time.time()
        obs_dict, state, reward_dict, done, info_dict = env.step_env(
            key_step, state, actions, env_params
        )
        print(f"Step time: {time.time() - start_step:.4f}s")

        # Print some info
        print("Obs (MM)", obs_dict["market_maker"])
        print("Obs (EXE)", obs_dict["execution"])
        print("Reward Dict:", reward_dict)
        print("Done?", done)

        if done:
            print("="*20, "Episode Finished", "="*20)
            break

    # 6) Optional: test vmap for many parallel envs
    enable_vmap = False
    if enable_vmap:
        vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))
        vmap_step  = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

        num_envs = 4
        vmap_keys = jax.random.split(rng, num_envs)

        # Reset all in parallel
        obs_batch, state_batch = vmap_reset(vmap_keys, env_params)
        print("Parallel reset obs:", obs_batch)

        # Sample random actions for each env
        def random_actions_for_env(k):
            k1, k2 = jax.random.split(k)
            mm_a = jax.random.randint(k1, (env.mm_n_actions,), 0, 100)
            ex_a = jax.random.randint(k2, (env.exe_n_actions,), 0, 50)
            return {"market_maker": mm_a, "execution": ex_a}

        vmap_act_sample = jax.vmap(random_actions_for_env, in_axes=(0,))
        action_batch = vmap_act_sample(vmap_keys)

        obs_next, state_next, rewards, dones, infos = vmap_step(vmap_keys, state_batch, action_batch, env_params)
        print("Obs Next (parallel):", obs_next)
        print("Rewards (parallel):", rewards)
