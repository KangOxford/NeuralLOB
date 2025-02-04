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
class MARLEnvState(BaseEnvState):
    """
     Multi-agent state with sub-states for:
      1) Market Maker (mm_* fields)
      2) Execution (exe_* fields)
    and the shared order book from BaseEnvState (inherited).
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
class MARLEnvParams(BaseEnvParams):
    mm_reward_lambda: float = 0.0001
    exe_reward_lambda: float = 1.0
    exe_task_size: int = 100


class MARLEnv(BaseLOBEnv):

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
    ):
        super().__init__(
            alphatradePath,
            window_index,     # this is 'window_selector'
            episode_time,     # this is 'sliceTimeWindow'
            ep_type=ep_type
        )
        
        # fields for MARLEnv:
        self.window_index = window_index
        self.episode_time = episode_time
        self.ep_type      = ep_type

        self.mm_trader_id        = mm_trader_id
        self.exe_trader_id       = exe_trader_id
        self.mm_reward_lambda    = mm_reward_lambda
        self.exe_reward_lambda   = exe_reward_lambda
        self.exe_task_size       = exe_task_size

        self.mm_n_actions = 6
        self.exe_n_actions = 2


    def default_params(self) -> MARLEnvParams:
        base_params = super().default_params()  


        return dataclasses.replace(
            base_params,
            mm_reward_lambda=self.mm_reward_lambda,
            exe_reward_lambda=self.exe_reward_lambda,
            exe_task_size=self.exe_task_size
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: MARLEnvParams
    ) -> Tuple[Dict[str, jnp.ndarray], MARLEnvState]:
        """
        For now just reset base env and set the rest of the parameters
        """

        obs_base, base_state = super().reset_env(key, params)

        init_state = MARLEnvState(
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
            
            # Market Maker sub-state
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

            # Execution sub-state
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

        # Build new observations for each agent
        obs_dict = {
            "market_maker": self._get_obs_mm(init_state, params),
            "execution":    self._get_obs_exe(init_state, params)
        }
        return obs_dict, init_state


    # =========================================================================
    # step_env
    # =========================================================================
    def step_env(
            self,
            key: chex.PRNGKey,
            state: MARLEnvState,
            actions: Dict[str, jnp.ndarray],
            params: MARLEnvParams
    ) -> Tuple[Dict[str, jnp.ndarray], MARLEnvState, Dict[str, float], bool, Dict[str, Dict]]:
        """
        Processes both agents: Market Maker and Execution, in one step.
        Returns (obs_dict, new_state, reward_dict, done, info_dict).
        """
        # 1) Gather data messages
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )

        # 2) Build agent messages (actions + cancels)
        mm_action_msgs, mm_cnl_msgs = self._getActionMsgs_mm(actions["market_maker"], state, params, key)
        exe_action_msgs, exe_cnl_msgs= self._getActionMsgs_exe(actions["execution"], state, params, key)

        # Combine agent messages + data messages
        combined_agent_msgs = jnp.concatenate([mm_cnl_msgs, exe_cnl_msgs, mm_action_msgs, exe_action_msgs], axis=0)
        total_messages = jnp.concatenate([combined_agent_msgs, data_messages], axis=0)

        # 3) Process all messages in one pass
        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        )

        # 4) Separate trades for each agent
        mm_agent_trades  = job.get_agent_trades(trades, self.mm_trader_id)
        exe_agent_trades = job.get_agent_trades(trades, self.exe_trader_id)

        # 5) Compute each agent's reward
        mm_rew, mm_info   = self._get_reward_mm(state, params, trades, mm_agent_trades, bestasks, bestbids)
        exe_rew, exe_info = self._get_reward_exe(state, params, trades, exe_agent_trades, bestasks, bestbids)

        # 6) Update multi-agent state
        new_state = self._update_state(state, asks, bids, trades, bestasks, bestbids, mm_info, exe_info, params)

        # 7) Build new observations
        obs_dict = {
            "market_maker": self._get_obs_mm(new_state, params),
            "execution":    self._get_obs_exe(new_state, params)
        }

        # 8) Check if done
        done = self.is_terminal(new_state, params)

        # 9) Build final reward/info dict
        reward_dict = {
            "market_maker": float(mm_rew),
            "execution":    float(exe_rew)
        }
        info_dict = {
            "market_maker": mm_info,
            "execution":    exe_info,
            "__done__": done
        }

        return obs_dict, new_state, reward_dict, done, info_dict

    # =========================================================================
    # is_terminal: merges both agent conditions
    # =========================================================================
    def is_terminal(self, state: MARLEnvState, params: MARLEnvParams) -> bool:
        """
        Time-based done for Market Maker, plus 'task done' for Execution agent.
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
        state: MARLEnvState,
        params: MARLEnvParams,
        key: chex.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  
        # 1) Reshape the raw action
        shaped_action = self._reshape_action_mm(mm_action, state, params, key)

        # 2) Build order messages
        action_msgs = self._build_mm_order_msgs(shaped_action, state, params, key)

        # 3) Build cancellation messages
        cnl_msg_bid = job.getCancelMsgs(
            state.bid_raw_orders,
            self.mm_trader_id,
            self.mm_n_actions // 2,  # half for bids
            1
        )
        cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.mm_trader_id,
            self.mm_n_actions // 2,  # half for asks
            -1
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        # 4) filter them
        action_msgs, cnl_msgs = self._filter_messages_mm(action_msgs, cnl_msgs)
        return action_msgs, cnl_msgs

    def _reshape_action_mm(self, mm_action: jax.Array, state: MARLEnvState, params: MARLEnvParams, key: chex.PRNGKey) -> jax.Array:
        """
        Optionally modifies the raw action if 'delta' style. Otherwise returns as is.
        """
        # For demonstration: if user sets "delta", do a naive approach
        def twapV3():
            remainingTime = params.episode_time - jnp.array((state.time - state.init_time)[0], dtype=jnp.int32)
            ifMarketOrder = (remainingTime <= 60)  # last minute is market
            remainInv = state.mm_inventory
            remainSteps = state.max_steps_in_episode - state.step_counter
            stepQuant = jnp.ceil(remainInv / jnp.maximum(remainSteps, 1)).astype(jnp.int32)
            # random split
            half = stepQuant // 2
            limit_quants  = jax.random.permutation(key, jnp.array([stepQuant - half, half]))
            market_quants = jnp.array([stepQuant, stepQuant])
            return jnp.where(ifMarketOrder, market_quants, limit_quants)

        action_type_str = getattr(params, "action_type", "pure") 
        if "delta" in action_type_str.lower():
            add_action = twapV3()
            mm_action = mm_action + jnp.pad(add_action, (0, self.mm_n_actions - add_action.shape[0]))
        return mm_action

    def _build_mm_order_msgs(self, mm_action: jax.Array, state: MARLEnvState, params: MARLEnvParams, key: chex.PRNGKey) -> jnp.ndarray:
        n_actions = self.mm_n_actions
        types = jnp.ones((n_actions,), jnp.int32)
        # half are buys, half are sells
        sides_bids = jnp.ones((n_actions // 2,), jnp.int32)
        sides_asks = -jnp.ones((n_actions // 2,), jnp.int32)
        sides = jnp.concatenate([sides_bids, sides_asks])

        trader_ids = jnp.ones((n_actions,), jnp.int32) * self.mm_trader_id
        order_ids = (jnp.ones((n_actions,), jnp.int32)*(self.mm_trader_id + state.customIDcounter)) + jnp.arange(n_actions)

        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (n_actions, 2)
        )

        # measure best_ask, best_bid from the last 100 stored
        best_ask = jnp.int32((state.mm_best_asks[-100:, 0].mean() // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.mm_best_bids[-100:, 0].mean() // self.tick_size) * self.tick_size)

        buy_prices  = jnp.array([best_bid, best_bid - self.tick_size, best_bid + self.tick_size], dtype=jnp.int32)
        sell_prices = jnp.array([best_ask, best_ask + self.tick_size, best_ask - self.tick_size], dtype=jnp.int32)
        all_prices  = jnp.concatenate([buy_prices, sell_prices], axis=0)

        quants = mm_action.astype(jnp.int32)
        action_msgs = jnp.stack([types, sides, quants, all_prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)  # shape [N, 8]

        return action_msgs

    def _filter_messages_mm(self, action_msgs: jnp.ndarray, cnl_msgs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        @partial(jax.vmap, in_axes=(0, None))
        def p_in_cnl(p, prices_cnl):
            return jnp.where((prices_cnl == p) & (p != 0), True, False)

        def matching_masks(prices_a, prices_cnl):
            res = p_in_cnl(prices_a, prices_cnl)
            return jnp.any(res, axis=1), jnp.any(res, axis=0)

        a_mask, c_mask = matching_masks(action_msgs[:, 3], cnl_msgs[:, 3])

        # For simplicity, no partial cancellation TODO 
        return action_msgs, cnl_msgs

    def _get_reward_mm(
        self,
        state: MARLEnvState,
        params: MARLEnvParams,
        trades: jnp.ndarray,
        mm_agent_trades: jnp.ndarray,
        bestasks: jnp.ndarray,
        bestbids: jnp.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        TODO: Implement actual function
        """
        agent_exec_qty = jnp.abs(mm_agent_trades[:, 1]).sum()
        reward = 0.0
        info_dict = {
            "inventory": float(state.mm_inventory),
            "total_PnL": float(state.mm_total_PnL),
            "best_ask": float(bestasks[-1, 0]),
            "best_bid": float(bestbids[-1, 0]),
        }
        return reward, info_dict

    def _get_obs_mm(self, state: MARLEnvState, params: MARLEnvParams) -> jnp.ndarray:
        """
         TODO: Implement actual function
        """
        time_elapsed = state.time[0] + state.time[1]/1e9 - (state.init_time[0] + state.init_time[1]/1e9)
        time_remaining = params.episode_time - time_elapsed

        obs = jnp.array([
            state.mm_inventory,
            state.mm_best_bids[-1, 0],
            state.mm_best_asks[-1, 0],
            time_remaining
        ], dtype=jnp.float32)
        return obs

    # =========================================================================
    # Execution Agent: actions, reward, obs
    # =========================================================================
    def _getActionMsgs_exe(
        self,
        exe_action: jnp.ndarray,
        state: MARLEnvState,
        params: MARLEnvParams,
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

    def _reshape_action_exe(self, exe_action: jnp.ndarray, state: MARLEnvState, params: MARLEnvParams, key: chex.PRNGKey) -> jnp.ndarray:
        # TODO add actual code
        return exe_action

    def _build_exe_order_msgs(self, exe_action: jnp.ndarray, state: MARLEnvState, params: MARLEnvParams, key: chex.PRNGKey) -> jnp.ndarray:
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
        state: MARLEnvState,
        params: MARLEnvParams,
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
        old_state: MARLEnvState,
        asks: jnp.ndarray,
        bids: jnp.ndarray,
        trades: jnp.ndarray,
        bestasks: jnp.ndarray,
        bestbids: jnp.ndarray,
        mm_info: Dict[str, float],
        exe_info: Dict[str, float],
        params: MARLEnvParams
    ) -> MARLEnvState:
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



# =============================================================================
# ==============================   MAIN  =======================================
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
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
        print("Using default folder:", ATFolder)

    # Configuration parameters
    config = {
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 300,  # 5 minutes
        "WINDOW_INDEX": 1,
        "MM_TRADER_ID": 1111,
        "MM_REWARD_LAMBDA": 0.0001,
        "EXE_TRADER_ID": 2222,
        "EXE_REWARD_LAMBDA": 1.0,
        "EXE_TASK_SIZE": 100,
    }

    # Create RNG keys
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the multi-agent environment
    env = MARLEnv(
        alphatradePath=ATFolder,
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        mm_trader_id=config["MM_TRADER_ID"],
        exe_trader_id=config["EXE_TRADER_ID"],
        mm_reward_lambda=config["MM_REWARD_LAMBDA"],
        exe_reward_lambda=config["EXE_REWARD_LAMBDA"],
        exe_task_size=config["EXE_TASK_SIZE"],
    )

    # Get default environment parameters and override if necessary
    env_params = env.default_params()
    env_params = dataclasses.replace(env_params, episode_time=config["EPISODE_TIME"])

    # Reset the environment
    obs, state = env.reset_env(key_reset, env_params)
    print("Reset done.")
    print("Market Maker Observation:", obs["market_maker"])
    print("Execution Observation:", obs["execution"])

    # Run a loop with random actions for each agent
    for step in range(1, 20):
        print("=" * 40)
        print(f"Step {step}")
        
        # Sample actions from each sub-environment's action space.
        action_mm = env.mm_env.action_space().sample(key_policy)
        action_exe = env.exe_env.action_space().sample(key_policy)
        
        # Wrap the actions to ensure they are at least 1D
        action_mm = jnp.atleast_1d(action_mm)
        action_exe = jnp.atleast_1d(action_exe)
        actions = {"market_maker": action_mm, "execution": action_exe}

        # Perform a step
        start_time = time.time()
        obs, state, rewards, done, info = env.step_env(key_step, state, actions, env_params)
        elapsed = time.time() - start_time
        print(f"Step time: {elapsed:.4f} sec")
        print("Rewards:", rewards)
        print("Info:", info)
        print("Market Maker Observation:", obs["market_maker"])
        print("Execution Observation:", obs["execution"])
        
        if done:
            print("Episode finished!")
            break
