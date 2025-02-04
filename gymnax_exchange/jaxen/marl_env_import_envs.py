import os, sys, time, dataclasses
from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct
import jax.tree_util as jtu


sys.path.append(os.path.abspath("/home/duser/AlphaTrade"))

from mm_env import MarketMakingEnv, EnvState as MMState, EnvParams as MMParams
from exec_env import ExecutionEnv, EnvState as EXEState, EnvParams as EXEParams
from gymnax_exchange.jaxen.base_env import BaseLOBEnv, EnvState as BaseState, EnvParams as BaseParams

# Define a combined (multi–agent) state that extends the base order book state
@struct.dataclass
class MultiAgentState(BaseState):
    # Sub–state for market maker and execution agent.
    mm_state: MMState
    exe_state: EXEState

# Define a combined parameters class.
@struct.dataclass
class MultiAgentParams(BaseParams):
    mm_params: MMParams
    exe_params: EXEParams

# define the MARL environment.
class MARLEnv(BaseLOBEnv):
    def __init__(self,
                 alphatradePath: str,
                 window_index: int,
                 episode_time: int,
                 ep_type: str = "fixed_time",
                 mm_trader_id: int = 1111,
                 exe_trader_id: int = 2222,
                 mm_reward_lambda: float = 0.0001,
                 exe_reward_lambda: float = 1.0,
                 exe_task_size: int = 100,
                 mm_action_type: str = "pure",
                 mm_n_ticks_in_book: int = 2,
                 mm_max_task_size: int = 500):
        # Initialize the base environment
        super().__init__(alphatradePath, window_index, episode_time, ep_type=ep_type)
        # Create the market making sub–env
        self.mm_env = MarketMakingEnv(
            alphatradePath,
            window_index=window_index,
            action_type=mm_action_type,
            episode_time=episode_time,
            max_task_size=mm_max_task_size,
            rewardLambda=mm_reward_lambda,
            ep_type=ep_type
        )
        # Create the execution sub–env
        self.exe_env = ExecutionEnv(
            alphatradePath,
            task="buy",  
            window_index=window_index,
            action_type="pure", 
            episode_time=episode_time,
            max_task_size=exe_task_size,
            rewardLambda=exe_reward_lambda,
            ep_type=ep_type
        )
        self.mm_trader_id = mm_trader_id
        self.exe_trader_id = exe_trader_id

    def default_params(self) -> MultiAgentParams:
        # Get the base parameters from BaseLOBEnv
        base_params = super().default_params
        # Get the sub–env default parameters
        mm_params = self.mm_env.default_params
        exe_params = self.exe_env.default_params
        # Combine them into a MultiAgentParams instance.
        return MultiAgentParams(
            **dataclasses.asdict(base_params),
            mm_params=mm_params,
            exe_params=exe_params
        )

    def reset_env(self, key: chex.PRNGKey, params: MultiAgentParams) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState]:
        # Split keys for each sub–env
        key_mm, key_exe, key = jax.random.split(key, 3)
        mm_obs, mm_state = self.mm_env.reset_env(key_mm, params.mm_params)
        exe_obs, exe_state = self.exe_env.reset_env(key_exe, params.exe_params)
        # The shared base state is taken from mm_state
        base_state = mm_state  
        # Manually copy the base state fields
        multi_state = MultiAgentState(
            ask_raw_orders = base_state.ask_raw_orders,
            bid_raw_orders = base_state.bid_raw_orders,
            trades = base_state.trades,
            init_time = base_state.init_time,
            time = base_state.time,
            customIDcounter = base_state.customIDcounter,
            window_index = base_state.window_index,
            step_counter = base_state.step_counter,
            max_steps_in_episode = base_state.max_steps_in_episode,
            start_index = base_state.start_index,
            # And now add the agent–specific states:
            mm_state = mm_state,
            exe_state = exe_state
        )
        multi_obs = {"market_maker": mm_obs, "execution": exe_obs}
        return multi_obs, multi_state


    def step_env(self,
                 key: chex.PRNGKey,
                 state: MultiAgentState,
                 actions: Dict[str, jnp.ndarray],
                 params: MultiAgentParams
                 ) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState, Dict[str, float], bool, Dict[str, Dict]]:

        # Split keys for each agent (and one extra if needed)
        key_mm, key_exe, key = jax.random.split(key, 3)

        # -------------------------------------------------------
        # (A) Build External Data Messages (common to both agents)
        # -------------------------------------------------------
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )

        # -------------------------------------------------------
        # (B) Build Market Maker messages
        # -------------------------------------------------------
        # Use the MM env’s message-building functions
        mm_raw_action = self.mm_env._reshape_action(actions["market_maker"],
                                                    state.mm_state,
                                                    params.mm_params,
                                                    key_mm)
        mm_order_msgs = self.mm_env._getActionMsgs(mm_raw_action,
                                                   state.mm_state,
                                                   params.mm_params)
        mm_cnl_msgs = job.getCancelMsgs(
            state.bid_raw_orders,  # using the shared order book from the base state
            self.mm_trader_id,
            self.mm_env.n_actions // 2,
            1
        )
        mm_cnl_msgs_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.mm_trader_id,
            self.mm_env.n_actions // 2,
            -1
        )
        mm_cnl_msgs = jnp.concatenate([mm_cnl_msgs, mm_cnl_msgs_ask], axis=0)

        # Do filtering to net cancellations in MM)
        mm_order_msgs, mm_cnl_msgs = self.mm_env._filter_messages(mm_order_msgs, mm_cnl_msgs)

        # -------------------------------------------------------
        # (C) Build Execution messages
        # -------------------------------------------------------
        exe_raw_action = self.exe_env._reshape_action(actions["execution"],
                                                      state.exe_state,
                                                      params.exe_params,
                                                      key_exe)
        exe_order_msgs = self.exe_env._getActionMsgs(exe_raw_action,
                                                     state.exe_state,
                                                     params.exe_params)
        
        # For execution, decide which side to cancel (depending on task)
        side_for_exe = 1 - state.exe_state.is_sell_task * 2
        raw_order_side = jax.lax.cond(
            state.exe_state.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        exe_cnl_msgs = job.getCancelMsgs(
            raw_order_side,
            self.exe_trader_id,
            self.exe_env.n_actions,
            side_for_exe
        )
        exe_order_msgs, exe_cnl_msgs = self.exe_env._filter_messages(exe_order_msgs, exe_cnl_msgs)

        # -------------------------------------------------------
        # (D) Combine all agent messages with data messages
        # -------------------------------------------------------
        combined_msgs = jnp.concatenate([
            mm_cnl_msgs,
            mm_order_msgs,
            exe_cnl_msgs,
            exe_order_msgs,
            data_messages
        ], axis=0)

        # -------------------------------------------------------
        # (E) Process combined messages through the order book
        # -------------------------------------------------------


        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
        (new_asks, new_bids, new_trades), (new_bestasks, new_bestbids) = job.scan_through_entire_array_save_bidask(
            combined_msgs,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        )
        # Forward-fill best prices if necessary:
        new_bestasks = self._ffill_best_prices(new_bestasks[-self.stepLines+1:], state.best_asks[-1, 0])
        new_bestbids = self._ffill_best_prices(new_bestbids[-self.stepLines+1:], state.best_bids[-1, 0])
        # Update time and ID counter
        final_time = combined_msgs[-1, -2:] + params.time_delay_obs_act
        final_id_ctr = state.customIDcounter + self.mm_env.n_actions + 1  # adjust as needed

        # -------------------------------------------------------
        # (F) Compute agent-specific rewards and observations
        # -------------------------------------------------------
        mm_agent_trades = job.get_agent_trades(new_trades, self.mm_trader_id)
        mm_reward, mm_info = self.mm_env._get_reward(state.mm_state, params.mm_params, new_trades, new_bestasks, new_bestbids)
        mm_obs = self.mm_env._get_obs(state.mm_state, params.mm_params)

        exe_agent_trades = job.get_agent_trades(new_trades, self.exe_trader_id)
        exe_reward, exe_info = self.exe_env._get_reward(state.exe_state, params.exe_params, new_trades)
        exe_obs = self.exe_env._get_obs(state.exe_state, params.exe_params)

        # -------------------------------------------------------
        # (G) Update the multi–agent state
        # -------------------------------------------------------
        # Update the shared base state fields
        base_state = state  
        new_shared_state = {
            "ask_raw_orders": new_asks,
            "bid_raw_orders": new_bids,
            "trades": new_trades,
            "time": final_time,
            "customIDcounter": final_id_ctr,
            "best_asks": new_bestasks,
            "best_bids": new_bestbids,
            "step_counter": state.step_counter + 1
        }
        new_mm_state = state.mm_state.replace(**new_shared_state)
        new_exe_state = state.exe_state.replace(**new_shared_state)

        new_state = MultiAgentState(
            ask_raw_orders=new_asks,
            bid_raw_orders=new_bids,
            trades=new_trades,
            init_time=state.init_time,
            time=final_time,
            customIDcounter=final_id_ctr,
            window_index=state.window_index,
            step_counter=state.step_counter + 1,
            max_steps_in_episode=state.max_steps_in_episode,
            start_index=state.start_index,
            mm_state=new_mm_state,
            exe_state=new_exe_state
        )

        obs = {"market_maker": mm_obs, "execution": exe_obs}
        rewards = {"market_maker": float(mm_reward), "execution": float(exe_reward)}
        done = self.is_terminal(new_state, params)
        info = {"market_maker": mm_info, "execution": exe_info}
        return obs, new_state, rewards, done, info


    def action_space(self, params: Optional[MultiAgentParams] = None):
        # Return a dictionary of action spaces
        mm_space = self.mm_env.action_space(params.mm_params if params is not None else None)
        exe_space = self.exe_env.action_space(params.exe_params if params is not None else None)
        return {"market_maker": mm_space, "execution": exe_space}

    def observation_space(self, params: Optional[MultiAgentParams] = None):
        mm_space = self.mm_env.observation_space(params.mm_params if params is not None else None)
        exe_space = self.exe_env.observation_space(params.exe_params if params is not None else None)
        return {"market_maker": mm_space, "execution": exe_space}


# --- Example main function to test the MARL environment ---
if __name__ == "__main__":
    import sys
    import time
    import dataclasses

    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
        print("Using default folder:", ATFolder)

    config = {
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 300,  # for example, 5 minutes
        "WINDOW_INDEX": 1,
        # sub–env parameters:
        "MM_TRADER_ID": 1111,
        "MM_REWARD_LAMBDA": 0.0001,
        "MM_ACTION_TYPE": "pure",
        "MM_MAX_TASK_SIZE": 500,
        "EXE_TRADER_ID": 2222,
        "EXE_REWARD_LAMBDA": 1.0,
        "EXE_TASK_SIZE": 100,
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the MARL environment.
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
        mm_action_type=config["MM_ACTION_TYPE"],
        mm_max_task_size=config["MM_MAX_TASK_SIZE"]
    )
    # Get the default combined parameters.
    env_params = env.default_params()

    env_params = dataclasses.replace(env_params, episode_time=config["EPISODE_TIME"])

    # Reset the environment.
    obs, state = env.reset_env(key_reset, env_params)
    print("Reset done. Market maker obs:", obs["market_maker"])
    print("Execution obs:", obs["execution"])

    # run a loop that samples random actions for each agent.
    for i in range(1, 20):
        print("=" * 40)
        print(f"Step {i}")
        # Get random actions from each agent’s action space.
        action_mm = env.mm_env.action_space().sample(key_policy)
        action_exe = env.exe_env.action_space().sample(key_policy)
        actions = {"market_maker": action_mm, "execution": action_exe}
        obs, state, rewards, done, info = env.step_env(key_step, state, actions, env_params)
        print("Step rewards:", rewards)
        print("Step info:", info)
        if done:
            print("Episode finished!")
            break
