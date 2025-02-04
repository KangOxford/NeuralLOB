"""
Market Making Environment for Limit Order Book with variable start time for episodes. 

University of Oxford
Corresponding Author: 
Kang Li     (kang.li@keble.ox.ac.uk)
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
Peer Nagy   (peer.nagy@reuben.ox.ac.uk)
V1.0 



Module Description
This module extends the base simulation environment for limit order books 
 using JAX for high-performance computations, specifically tailored for 
 execution tasks in financial markets. It is particularly designed for 
 reinforcement learning applications focusing on 
 optimal trade execution strategies.

Key Components
EnvState:   Dataclass to encapsulate the current state of the environment, 
            including the raw order book, trades, and time information.
EnvParams:  Configuration class for environment-specific parameters, 
            such as task details, message and book data, and episode timing.
MarketMakingEnv: Environment class inheriting from BaseLOBEnv, 
              offering specialized methods for order placement and 
              execution tasks in trading environments. 


Functionality Overview
__init__:           Initializes the execution environment, setting up paths 
                    for data, action types, and task details. 
                    It includes pre-processing and initialization steps 
                    specific to execution tasks.
default_params:     Returns the default parameters for execution environment,
                    adjusting for tasks such as buying or selling.
step_env:           Advances the environment by processing actions and market 
                    messages. It updates the state and computes the reward and 
                    termination condition based on execution-specific criteria.
reset_env:          Resets the environment to a state appropriate for a new 
                    execution task. Initializes the order book and sets initial
                    state specific to the execution context.
is_terminal:        Checks whether the current state is terminal, based on 
                    the number of steps executed or tasks completed.

action_space:       Defines the action space for execution tasks, including 
                    order types and quantities.
observation_space:  Define the observation space for execution tasks.
state_space:        Describes the state space of the environment, tailored 
                    for execution tasks with components 
                    like bids, asks, and trades.
reset_env:          Resets the environment to a specific state for execution. 
                    It selects a new data window, initializes the order book, 
                    and sets the initial state for execution tasks.
_getActionMsgs:      Generates action messages based on 
                    the current state and action. 
                    It determines the type, side, quantity, 
                    and price of orders to be executed.
                    including detailed order book information and trade history
_get_obs:           Constructs and returns the current observation for the 
                    execution environment, derived from the state.
_get_state_from_data:
_reshape_action:
_best_prices_impute
_get_reward:
name, num_actions:  Inherited methods providing the name of the environment 
                    and the number of possible actions.


                
_get_data_messages: Inherited method to fetch market messages for a given 
                    step from all available messages.
"""

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
#sys.path.append('.')
print(os.getcwd())
#print(os.listdir('/home/duser/AlphaTrade/training_oneDay/data/Flow_10'))
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
# ================= imports ==================


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
from gymnax_exchange.utils import utils
import dataclasses

import jax.tree_util as jtu

@struct.dataclass
class EnvState(BaseEnvState):
    prev_action: chex.Array
    #TODO: Look into how we re-calc prev executed for market maker.
    prev_executed: chex.Array
    # Potentially could be moved to base,
    # so long as saving of best ask/bids is base behaviour. 
    best_asks: chex.Array
    best_bids: chex.Array
    init_price: int
    inventory:int
    mid_price:int
    total_PnL: float
    bid_passive_2 :int
    quant_bid_passive_2 :int
    ask_passive_2:int
    quant_ask_passive_2:int
    delta_time: float

@struct.dataclass
class EnvParams(BaseEnvParams):
    reward_lambda: float 

class MarketMakingEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, task, window_index, action_type, episode_time,
            max_task_size = 500, rewardLambda=0.0001, ep_type="fixed_time"):
        
        #Define Execution-specific attributes.
        self.n_ticks_in_book = 5 # Depth of PP actions
        self.action_type = action_type # 'delta' or 'pure'
        self.max_task_size = max_task_size #Functions as max trade size for us
        self.inventory=0
        self.market_share=0.
        self.rewardLambda = rewardLambda #
        # TODO: fix!! this can be overwritten in the base class
        self.n_actions = 8 # 4: (FT, M, NT, PP), 3: (FT, NT, PP), 2 (FT, NT), 1 (FT)
        
        super().__init__(
            alphatradePath,
            window_index,
            episode_time,
            ep_type,
        )

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        base_params = super().default_params
        flat_tree = jtu.tree_flatten(base_params)[0]
        #TODO: Clean this up to not have a magic number
        # BaseEnvParams
        base_vals = flat_tree[0:5] #Considers the base parameter values other than init state.
        state_vals = flat_tree[5:] #Considers the state values
        return EnvParams(
            *base_vals,
            EnvState(*state_vals),
            #self.max_task_size,
            reward_lambda=self.rewardLambda
        )


    def step_env(
        self, key: chex.PRNGKey, state: EnvState, input_action: jax.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        #=======================================#
        #====Load data messages for next step===#
        #=======================================#      
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )
        #data_messages=self._get_generative_messages(params.message_data,100)
       # jax.debug.print("data_messages :{}",data_messages)
        #=======================================#
        #======Process agent actions ===========#
        #=======================================#
        #jax.debug.print("action :{}",input_action)
        #def one_hot_action(action: int, n_actions: int):
         #   return jnp.full((n_actions,), action, dtype=jnp.int32)  # Fill the array with the action value


        #jax.debug.print("input_action.shape[0]:{}",input_action.shape[0])
        #input_action = one_hot_action(input_action, self.n_actions)
        #input_action=jnp.array([input_action])
        action = self._reshape_action(input_action, state, params,key)
        #jax.debug.print("action:{}",action)
        action_msgs = self._getActionMsgsV2(action, state, params)
        action_prices = action_msgs[:, 3]
        #jax.debug.print("action_msgs:{}",action_msgs)

        #Cancel all previous agent orders each step, send fresh
        cnl_msg_bid = job.getCancelMsgs(
            state.bid_raw_orders,
            self.trader_unique_id,
            1,#self.n_actions//2, 
            1  # bids
        )
        cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.trader_unique_id,
            1,#self.n_actions//2,
            -1  # ask side
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)
             
        
        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self._filter_messages(action_msgs, cnl_msgs)
        
        
        #=======================================#
        #===Process all messages through book===#
        #=======================================#

        # Add to the top of the data messages
        total_messages = jnp.concatenate([cnl_msgs, action_msgs, data_messages], axis=0)
        #jax.debug.print("total_messages :{}",total_messages)
        # Save time of final message to add to state
        time = total_messages[-1, -2:]
        # To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
        # Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines
        )
        # If best price is not available in the current step, use the last available price
        # TODO: check if we really only want the most recent stepLines prices (+1 for the additional market order)
        bestasks, bestbids = (
            self._ffill_best_prices(
                bestasks[-self.stepLines+1:],
                state.best_asks[-1, 0]
            ),
            self._ffill_best_prices(
                bestbids[-self.stepLines+1:],
                state.best_bids[-1, 0]
            )
        )
        agent_trades = job.get_agent_trades(trades, self.trader_unique_id)
        executions = self._get_executed_by_action(agent_trades, action, state,action_prices)
        #jax.debug.print("agent_trades:{}",agent_trades)
        #jax.debug.print("executions:{}",executions)
        #=======================================#
        #===force inventory sale at episode end=#
        #=======================================#
       # (asks, bids, trades), (new_bestask, new_bestbid), new_id_counter, new_time, mkt_exec_quant, doom_quant = \
        #    self._force_market_order_if_done(
        #         bestasks[-1], bestbids[-1], time, asks, bids, trades, state, params)
        bestasks = jnp.concatenate([bestasks,bestasks[-1:,:] ], axis=0, dtype=jnp.int32)
        bestbids = jnp.concatenate([bestbids, bestbids[-1:,:]], axis=0, dtype=jnp.int32)
        new_id_counter = state.customIDcounter + 3#self.n_actions + 1
        new_time = time + params.time_delay_obs_act

        bid_passive_2,quant_bid_passive_2,ask_passive_2,quant_ask_passive_2 = self._get_pass_price_quant(state)
        # TODO: consider adding quantity before (in priority) to each price / level

        # TODO: use the agent quant identification from the separate function _get_executed_by_level instead of _get_reward
        reward, extras = self._get_reward(state, params, trades,bestasks,bestbids)
        
        state = EnvState(
            prev_action = action_prices,#jnp.vstack([action_prices, action]).T,  # includes prices and quantitites 
            #TODO: implement prev_executed and get this on the state. 
            prev_executed = executions, # include prices and quantities 
            ask_raw_orders = asks,
            bid_raw_orders = bids,
            trades = trades,
            init_time = state.init_time,
            #time = time,
            time = new_time,
            # customIDcounter = state.customIDcounter + self.n_actions + 1,
            customIDcounter = new_id_counter,
            window_index = state.window_index,
            step_counter = state.step_counter + 1,
            max_steps_in_episode = state.max_steps_in_episode,
            start_index = state.start_index,
            best_asks = bestasks,
            best_bids = bestbids,
            init_price = state.init_price,
            mid_price=extras["mid_price"],
            #######
            inventory=extras["end_inventory"],
            total_PnL = state.total_PnL + extras["PnL"],
            bid_passive_2 = bid_passive_2,
            quant_bid_passive_2 = quant_bid_passive_2,
            ask_passive_2=ask_passive_2,
            quant_ask_passive_2=quant_ask_passive_2,            
            delta_time = new_time[0] + new_time[1]/1e9 - state.time[0] - state.time[1]/1e9,
        )
        #jax.debug.print("state.inv :{}",state.inventory)
        #jax.debug.print("reward:{}",reward)
        done = self.is_terminal(state, params)
        #jax.debug.print("inventory:{}",state.inventory)
        #jax.debug.print("reward :{}",reward)
        #jax.debug.print("done:{}",done)
        average_best_bid=bestbids[:, 0].mean()// self.tick_size * self.tick_size
        average_best_ask=bestasks[:, 0].mean()// self.tick_size * self.tick_size
        #jax.debug.print("average_best_bid:{}",average_best_bid)
        #jax.debug.print("average_best_ask:{}",average_best_ask)

        info = {
            "reward":reward,
            "window_index": state.window_index,
            "total_PnL": state.total_PnL,                           
            "current_step": state.step_counter,
            "done": done,
            "inventory": state.inventory,
            #"mkt_forced_quant": mkt_exec_quant + doom_quant,
            #"doom_quant": doom_quant,
            "market_share":extras["market_share"],
            "buyQuant":extras["buyQuant"],
            "sellQuant":extras["sellQuant"],
            "other_exec_quants":extras["other_exec_quants"],
            "averageMidprice":extras["averageMidprice"],
            "Step_PnL":extras["PnL"],
            "action_prices":action_prices,
            "average_best_bid":average_best_bid,
            "average_best_ask":average_best_ask,
            "InventoryPnL":extras["InventoryPnL"],
            "approx_realized_pnl":extras["approx_realized_pnl"],
            "approx_unrealized_pnl": extras["approx_unrealized_pnl"]
        }
       
        return self._get_obs(state, params), state, reward, done, info
    

    def reset_env(
            self,
            key : chex.PRNGKey,
            params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
        """ Reset the environment to init state (pre computed from data)."""
        key_, key = jax.random.split(key)
        _, state = super().reset_env(key, params)
        bid_passive_2,quant_bid_passive_2,ask_passive_2,quant_ask_passive_2 = self._get_pass_price_quant(state)
        state = dataclasses.replace(state, bid_passive_2=bid_passive_2, quant_bid_passive_2=quant_bid_passive_2,ask_passive_2=ask_passive_2,quant_ask_passive_2=quant_ask_passive_2)
        obs = self._get_obs(state, params)
        return obs, state
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """ Check whether state is terminal.
         For a market making task, we run untill time completes. This is hardcoded 
          as 5 seconds before the end of the episode or one step before """
        if self.ep_type == 'fixed_time':
            # TODO: make the 5 sec a function of the step size
            time_left=(params.episode_time - (state.time - state.init_time)[0] )
            #jax.debug.print("time_left :{}",time_left)
            return (
                (params.episode_time - (state.time - state.init_time)[0] <= 5)  # time over (last 5 seconds)
            )
        elif self.ep_type == 'fixed_steps':
            return (
                (state.max_steps_in_episode - state.step_counter <= 1)  # last step
               
            )
        else:
            raise ValueError(f"Unknown episode type: {self.ep_type}")
   
    def _get_pass_price_quant(self, state):
        """Get price and quanitity n_ticks into books"""
        bid_passive_2=state.best_bids[-1, 0] - self.tick_size * self.n_ticks_in_book
        ask_passive_2=state.best_asks[-1, 0] + self.tick_size * self.n_ticks_in_book
        quant_bid_passive_2 = job.get_volume_at_price(state.bid_raw_orders, bid_passive_2)
        quant_ask_passive_2 = job.get_volume_at_price(state.ask_raw_orders, ask_passive_2)
        return bid_passive_2,quant_bid_passive_2,ask_passive_2,quant_ask_passive_2
    
    def _get_state_from_data(self,first_message,book_data,max_steps_in_episode,window_index,start_index):
        """Reset state from data"""
        base_state = super()._get_state_from_data(first_message, book_data, max_steps_in_episode, window_index, start_index)
        base_vals = jtu.tree_flatten(base_state)[0]
        best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(base_state.ask_raw_orders,base_state.bid_raw_orders)
        M = (best_bid[0] + best_ask[0]) // 2 // self.tick_size * self.tick_size 

        return EnvState(
            *base_vals,
           # prev_action=jnp.zeros((self.n_actions, 2), jnp.int32),
            #prev_executed=jnp.zeros((self.n_actions,2 ), jnp.int32),
            prev_action=jnp.zeros((2), jnp.int32),#jnp.zeros((2 ,2), jnp.int32),
            prev_executed=jnp.zeros((2,2 ), jnp.int32),
            best_asks=jnp.resize(best_ask,(self.stepLines,2)),
            best_bids=jnp.resize(best_bid,(self.stepLines,2)),
            init_price=M,
            mid_price=M,
            inventory=0,
            total_PnL=0.,
            # updated on reset:
            bid_passive_2 = 0,
            quant_bid_passive_2 = 0,
            ask_passive_2=0,
            quant_ask_passive_2=0,
            delta_time=0.,
        )

    def _reshape_action(self, action : jax.Array, state: EnvState, params : EnvParams, key:chex.PRNGKey) -> jax.Array:
        '''This is not used at all in MM Env. Delta is an exec env type where they trade above a baseline of twap'''
        def twapV3(state, env_params):
            # ---------- ifMarketOrder ----------
            remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
            marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
            ifMarketOrder = (remainingTime <= marketOrderTime)
            # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
            # ---------- ifMarketOrder ----------
            # ---------- quants ----------
            remainedQuant =state.inventory
            #state.task_to_execute - state.quant_executed
            remainedStep = state.max_steps_in_episode - state.step_counter
            stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
            limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True)
            market_quants = jnp.array([stepQuant,stepQuant])
            quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
            # ---------- quants ----------
            return jnp.array(quants) 
      
        #we don't use truncate_actoin, we can trade into negative inventory etc
        def truncate_action(action, remainQuant):
            action = jnp.round(action).clip(0, remainQuant).astype(jnp.int32)
            # scaledAction = utils.clip_by_sum_int(action, remainQuant)
            scaledAction = jnp.where(
                action.sum() <= remainQuant,
                action,
                utils.hamilton_apportionment_permuted_jax(action, remainQuant, key)
            ).astype(jnp.int32)
            return scaledAction

        if self.action_type == 'delta':
            action = twapV3(state, params) + action
        return action
      
    def _filter_messages(
            self, 
            action_msgs: jax.Array,
            cnl_msgs: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
        """ Filter out cancelation messages, when same actions should be placed again.
            NOTE: only simplifies cancellations if new action size <= old action size.
                  To prevent multiple split orders, new larger orders still cancel the entire old order.
            TODO: consider allowing multiple split orders
            ex: at one level, 3 cancel & 1 action --> 2 cancel, 0 action
        """
        @partial(jax.vmap, in_axes=(0, None))
        def p_in_cnl(p, prices_cnl):
            return jnp.where((prices_cnl == p) & (p != 0), True, False)
        def matching_masks(prices_a, prices_cnl):
            res = p_in_cnl(prices_a, prices_cnl)
            return jnp.any(res, axis=1), jnp.any(res, axis=0)
        @jax.jit
        def argsort_rev(arr):
            """ 'arr' sorted in descending order (LTR priority tie-breaker) """
            return (arr.shape[0] - 1 - jnp.argsort(arr[::-1]))[::-1]
        @jax.jit
        def rank_rev(arr):
            """ Rank array in descending order, with ties having left-to-right priority. """
            return jnp.argsort(argsort_rev(arr))
        
        # jax.debug.print("action_msgs\n {}", action_msgs)
        # jax.debug.print("cnl_msgs\n {}", cnl_msgs)

        a_mask, c_mask = matching_masks(action_msgs[:, 3], cnl_msgs[:, 3])
        # jax.debug.print("a_mask \n{}", a_mask)
        # jax.debug.print("c_mask \n{}", c_mask)
        # jax.debug.print("MASK DIFF: {}", a_mask.sum() - c_mask.sum())
        
        a_i = jnp.where(a_mask, size=a_mask.shape[0], fill_value=-1)[0]
        a = jnp.where(a_i == -1, 0, action_msgs[a_i][:, 2])
        c_i = jnp.where(c_mask, size=c_mask.shape[0], fill_value=-1)[0]
        c = jnp.where(c_i == -1, 0, cnl_msgs[c_i][:, 2])
        
        # jax.debug.print("a_i \n{}", a_i)
        # jax.debug.print("a \n{}", a)
        # jax.debug.print("c_i \n{}", c_i)
        # jax.debug.print("c \n{}", c)

        rel_cnl_quants = (c >= a) * a
        # rel_cnl_quants = jnp.maximum(0, c - a)
        # jax.debug.print("rel_cnl_quants {}", rel_cnl_quants)
        # reduce both cancel and action message quantities to simplify
        action_msgs = action_msgs.at[:, 2].set(
            action_msgs[:, 2] - rel_cnl_quants[rank_rev(a_mask)])
            # action_msgs[:, 2] - rel_cnl_quants[utils.rank_rev(a_mask)])
        # set actions with 0 quant to dummy messages
        action_msgs = jnp.where(
            (action_msgs[:, 2] == 0).T,
            0,
            action_msgs.T,
        ).T
        cnl_msgs = cnl_msgs.at[:, 2].set(cnl_msgs[:, 2] - rel_cnl_quants[rank_rev(c_mask)])
            # cnl_msgs[:, 2] - rel_cnl_quants[utils.rank_rev(c_mask)])
        # jax.debug.print("action_msgs NEW \n{}", action_msgs)
        # jax.debug.print("cnl_msgs NEW \n{}", cnl_msgs)

        return action_msgs, cnl_msgs

    def _ffill_best_prices(self, prices_quants, last_valid_price):
        def ffill(arr, inval=-1):
            """ Forward fill array values `inval` with previous value """
            def f(prev, x):
                new = jnp.where(x != inval, x, prev)
                return (new, new)
            # initialising with inval in case first value is already invalid
            _, out = jax.lax.scan(f, inval, arr)
            return out

        # if first new price is invalid (-1), copy over last price
        prices_quants = prices_quants.at[0, 0:2].set(
            jnp.where(
                # jnp.repeat(prices_quants[0, 0] == -1, 2),
                prices_quants[0, 0] == -1,
                jnp.array([last_valid_price, 0]),
                prices_quants[0, 0:2]
            )
        )
        # set quantity to 0 if price is invalid (-1)
        prices_quants = prices_quants.at[:, 1].set(
            jnp.where(prices_quants[:, 0] == -1, 0, prices_quants[:, 1])
        )
        # forward fill new prices if some are invalid (-1)
        prices_quants = prices_quants.at[:, 0].set(ffill(prices_quants[:, 0]))
        # jax.debug.print("prices_quants\n {}", prices_quants)
        return prices_quants
 
    def _get_executed_by_price(self, agent_trades: jax.Array) -> jax.Array:
        """ 
        Get executed quantity by price from trades. Results are sorted by increasing price. 
        NOTE: this will not work for aggressive orders eating through the book (size limited by actions)
        TODO: make this more general for aggressive actions?
        """
        price_levels, r_idx = jnp.unique(
            agent_trades[:, 0], return_inverse=True, size=3, fill_value=0)#self.n_actions+1
        quant_by_price = jax.ops.segment_sum(jnp.abs(agent_trades[:, 1]), r_idx, num_segments=3)#self.n_actions+1
        price_quants = jnp.vstack((price_levels[1:], quant_by_price[1:])).T
        return price_quants
    
    def _get_executed_by_level(self, agent_trades: jax.Array, actions: jax.Array, state: EnvState) -> jax.Array:
        """ Get executed quantity by level from trades. Results are sorted from aggressive to passive
            using previous actions. (0 actions are skipped)
            NOTE: this will not work for aggressive orders eating through the book (size limited by actions)
            TODO: make this more general for aggressive actions?
            UPDATE FOR MM_Env: leave in order?
        """
       # is_sell_task = state.is_sell_task
        price_quants = self._get_executed_by_price(agent_trades)
        # sort from aggr to passive
        #price_quants = jax.lax.cond(
           # is_sell_task,
        #    lambda: price_quants,
         #   lambda: price_quants[::-1],  # for buy task, most aggressive is highest price
        #)
        #put executions in non-zero action places (keeping the order)
        price_quants = price_quants[jnp.argsort(jnp.argsort(actions <= 0))]
        return price_quants
    
    def _get_executed_by_action(self, agent_trades: jax.Array, actions: jax.Array, state: EnvState,action_prices:jax.Array) -> jax.Array:
        """ Get executed quantity by level from trades. 
        """
        #TODO: This will have an issue if we buy and sell at the same price. This should be avoided anyway.
        #TODO: Put in a safe guard for that.
        def find_index_safe(x, action_prices):
            # Create a mask for matching prices
            match_mask = action_prices == x
            has_match = jnp.any(match_mask)
            first_match = jnp.argmax(match_mask)  # Returns the first index of True, or 0 if no match
            return jax.lax.cond(
                has_match,
                lambda _: first_match,  # Return the index if a match exists
                lambda _: -1,           # Return -1 otherwise
                operand=None
            )

        # Map prices to indices
        price_to_index = jax.vmap(lambda x: find_index_safe(x, action_prices))(agent_trades[:, 0])
        #jax.debug.print("action_prices:{}",action_prices)
        #jax.debug.print("agent_trades :{}",agent_trades)

        # Create masks for valid indices
        valid_indices = price_to_index >= 0
        num_prices = 2#self.n_actions

        # Mask trades and indices instead of boolean indexing
        valid_trades = jnp.where(valid_indices, agent_trades[:, 1], 0)
        #jax.debug.print("valid_trades:{}",valid_trades)
        valid_price_to_index = jnp.where(valid_indices, price_to_index, 0)

        # Sum trades by price level
        executions = jax.ops.segment_sum(valid_trades, valid_price_to_index, num_segments=num_prices)
       # Create a 2D array with price levels and corresponding trade quantities
        price_quantity_pairs = jnp.stack([action_prices, executions], axis=-1)

        # Optionally, you can print or debug the final result
        #jax.debug.print("Price and Quantity Pairs: {}", price_quantity_pairs)

        return price_quantity_pairs
        #best_price = jax.lax.cond(
           # state.is_sell_task,
       #     lambda: state.best_bids[-1, 0],
       #     lambda: state.best_asks[-1, 0]
       # )
       # aggr_trades_mask = jax.lax.cond(
           # state.is_sell_task,
       #     lambda: agent_trades[:, 0] <= best_price,
        #    lambda: agent_trades[:, 0] >= best_price
        #)
        #exec_quant_aggr = jnp.where(
        #    aggr_trades_mask,
        #    jnp.abs(agent_trades[:, 1]),
        #    0
        #).sum()
        # jax.debug.print('best_price\n {}', best_price)
        # jax.debug.print('exec_quant_aggr\n {}', exec_quant_aggr)
        
       # price_quants_pass = self._get_executed_by_price(
            # agent_trades[~aggr_trades_mask]
        #    jnp.where(
         #       jnp.expand_dims(aggr_trades_mask, axis=1),
          #      0,
           #     agent_trades
           # )
        #)
        # jax.debug.print('price_quants_pass\n {}', price_quants_pass)
        # sort from aggr to passive
        #price_quants = jax.lax.cond(
         # #  state.is_sell_task,
          #  lambda: price_quants_pass,
           # lambda: price_quants_pass[::-1],  # for buy task, most aggressive is highest price
       # )
        # put executions in non-zero action places (keeping the order)
        #price_quants = price_quants[jnp.argsort(jnp.argsort(actions[1:] <= 0))]
        #price_quants = jnp.concatenate(
         #   (jnp.array([[best_price, exec_quant_aggr]]), price_quants),
        #)
        # jax.debug.print("actions {} \n price_quants {} \n", actions, price_quants)
        # return quants only (aggressive prices could be multiple)
       # return price_quants
    def _getActionMsgsV2(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''Transform discrete action into bid and ask order messages based on current best prices.'''
        # Compute best_ask and best_bid using a rolling average to reduce variance
        best_ask = jnp.int32((state.best_asks[-100:].mean(axis=0)[0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-100:].mean(axis=0)[0] // self.tick_size) * self.tick_size)
        
        # Convert action to integer scalar (assuming action is a single-element array)
        #action = jax.lax.convert_element_type(action[0], jnp.int32)  # Ensure it's a scalar
        
        # Define mappings for each action: [0-7]
        bid_offsets = jnp.array([0, 0, 0, 0, -1, 1, -1, -5], dtype=jnp.int32)
        ask_offsets = jnp.array([0, 0, 0, 0, -1, 1, 1, 5], dtype=jnp.int32)
        bid_quants = jnp.array([0, 10, 0, 10, 10, 10, 10, 10], dtype=jnp.int32)
        ask_quants = jnp.array([0, 10, 10, 0, 10, 10, 10, 10], dtype=jnp.int32)

        #Testing aggr
        #bid_offsets = jnp.array([0, 0, 10, -1, -1, 1, 0, 0], dtype=jnp.int32)
        #ask_offsets = jnp.array([0, -10, 0, 1, -1, 1, 0, 0], dtype=jnp.int32)
        #bid_quants = jnp.array([0, 0, 10, 1, 1, 1, 0, 1], dtype=jnp.int32)
        #ask_quants = jnp.array([0, 10, 0, 1, 1, 1, 1, 0], dtype=jnp.int32)

        #Testing aggr
        #bid_offsets = jnp.array([0, 0, 10, -1, -1, 1, 0, 0], dtype=jnp.int32)
        #ask_offsets = jnp.array([0, -10, 0, 1, -1, 1, 0, 0], dtype=jnp.int32)
        #bid_quants = jnp.array([0, 0, 10, 1, 1, 1, 0, 1], dtype=jnp.int32)
        #ask_quants = jnp.array([0, 10, 0, 1, 1, 1, 1, 0], dtype=jnp.int32)
        
        tick_offset = self.n_ticks_in_book * self.tick_size  # Total price offset per direction
        
        # Get parameters for current action
        bid_offset = bid_offsets[action]
        ask_offset = ask_offsets[action]
        bid_quant = bid_quants[action]
        ask_quant = ask_quants[action]
        
        # Calculate prices with bounds checking
        bid_price = best_bid + bid_offset * tick_offset
        ask_price = best_ask + ask_offset * tick_offset
        bid_price = jnp.maximum(bid_price, 0)  # Prevent negative prices
        ask_price = jnp.maximum(ask_price, 0)
        
        # --------------- Construct messages ---------------
        # Message components (2 messages: bid then ask)
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, self.trader_unique_id, dtype=jnp.int32)
        
        # Generate unique order IDs
        base_id = self.trader_unique_id + state.customIDcounter
        order_ids = base_id + jnp.array([0, 1], dtype=jnp.int32)
        
        # Time fields (replicated for each message)
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )
        
        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        
        return action_msgs
    
    def _getActionMsgs(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''Shape the action quantities in to messages sent the order book at the 
        prices levels determined from the orderbook'''
        def normal_quant_price(price_levels: jax.Array, action: jax.Array):
            def combine_mid_nt(quants, prices):
                quants = quants \
                    .at[2].set(quants[2] + quants[1]) \
                    .at[1].set(0)
                prices = prices.at[1].set(-1)
                return quants, prices

            quants = action.astype(jnp.int32)
            #We do the [-1] elsewhere
            #prices = jnp.array(price_levels[:-1])

            #Possibly need to do this for mm?? It wont be at n_actions =4 though !!
            if self.n_actions == 4:
                # if mid_price == near_touch_price: combine orders into one
                return jax.lax.cond(
                    price_levels[1] == price_levels[2],
                    combine_mid_nt,
                    lambda q, p: (q, p),
                    quants, prices
                )
            else:
                return quants, prices
        
            
        def buy_task_prices(best_ask, best_bid):
            # FT = best_ask
            FT = ((best_ask) // self.tick_size * self.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            BI = best_bid + self.tick_size*self.n_ticks_in_book #BID inside, slightly more aggresive buying
            NT = best_bid
            PP = best_bid - self.tick_size*self.n_ticks_in_book
            MKT = job.MAX_INT
            if action.shape[0]//2 == 4:
                return FT, M, NT, PP, MKT
            elif action.shape[0]//2 == 3:
                return BI, NT, PP, MKT
            elif action.shape[0]//2 == 2:
                return NT, PP, MKT
            elif action.shape[0]//2 == 1:
                return NT, MKT

        def sell_task_prices(best_ask, best_bid):
            # FT = best_bid
            # essentially convert to market order (20% lower price than best bid)
            FT = ((best_bid) // self.tick_size * self.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            AI = best_ask - self.tick_size*self.n_ticks_in_book #Ask inside, slightly more aggresive selling
            NT = best_ask
            PP = best_ask + self.tick_size*self.n_ticks_in_book
            MKT = 0
            if action.shape[0]//2 == 4:
                return FT, M, NT, PP, MKT
            elif action.shape[0]//2 == 3:
                return AI, NT, PP, MKT
            elif action.shape[0]//2 == 2:
                return NT, PP, MKT
            elif action.shape[0]//2 == 1:
                return NT, MKT

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types = jnp.ones((self.n_actions,), jnp.int32)
        sides_bids = jnp.ones((self.n_actions // 2,), jnp.int32)  # Use integer division to ensure result is an int
        sides_asks = (-1) * jnp.ones((self.n_actions // 2,), jnp.int32)
        sides = jnp.concatenate([sides_bids, sides_asks])
        trader_ids = jnp.ones((self.n_actions,), jnp.int32) * self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids = (jnp.ones((self.n_actions,), jnp.int32) *
                    (self.trader_unique_id + state.customIDcounter)) \
                    + jnp.arange(0, self.n_actions) #Each message has a unique ID
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (self.n_actions, 2)
        )
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        #best_ask, best_bid = state.best_asks[-1, 0], state.best_bids[-1, 0]
        #Trade off the average over the last 100 messages to avoid the variance:
        best_ask = jnp.int32((state.best_asks[-100:].mean(axis=0)[0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-100:].mean(axis=0)[0] // self.tick_size) * self.tick_size)
        #best_ask, best_bid = state.best_asks[-1, 0], state.best_bids[-1, 0]

        sell_levels=sell_task_prices(best_ask, best_bid)
        sell_levels = jnp.array(sell_levels[:-1])

        buy_levels=buy_task_prices(best_ask, best_bid)
        buy_levels = jnp.array(buy_levels[:-1])

        price_levels=jnp.concatenate([buy_levels,sell_levels])
        

        # --------------- 02 info for deciding prices ---------------

   
        quants = action.astype(jnp.int32)
        #quants=jnp.array([0,0])
        prices=price_levels
     
        #quants, prices = normal_quant_price(price_levels, action)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        #jax.debug.print('action_msgs\n {}', action_msgs)
        return action_msgs
        # ============================== Get Action_msgs ==============================
    def _trade_at_midprice(self,
            #quant_left: jax.Array,
            bestask: jax.Array,
            bestbid: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: EnvState,
            params: EnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:
        """WIP; WILL MAKE IT SO THE AGENT JUST SELLS ALL AT MID PRICE. NEEDS SOME WORK!!"""
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
             
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = (self.trader_unique_id == executed[:, 6]) | (self.trader_unique_id == executed[:, 7]) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0) 

        #Find agent Buys and Agent sells from agent Trades:
        #The below mask puts passive buys or aggresive buys into "agent buys".
        #Logic: Q>0, TIDs=BUY; Q<0 TIDa= BUY
        mask_buy = (((agentTrades[:, 1] >= 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] < 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        mask_sell = (((agentTrades[:, 1] < 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] >= 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        agent_buys=jnp.where(mask_buy[:, jnp.newaxis], agentTrades, 0)
        agent_sells=jnp.where(mask_sell[:, jnp.newaxis], agentTrades, 0)



        #Find amount bought and sold in the step
        buyQuant=jnp.abs(agent_buys[:, 1]).sum()
        sellQuant=jnp.abs(agent_sells[:, 1]).sum()

        #Calculate the change in inventory & the new inventory
        inventory_delta = buyQuant - sellQuant
        new_inventory=state.inventory+inventory_delta
        #-----check if ep over-----#
        if self.ep_type == 'fixed_time':
            remainingTime = params.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= 5  # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1
        mid_price= (bestbid+bestask)//(2*self.tick_size)* self.tick_size
       

        def place_midprice_trade(trades, price, quant, time):
            '''Place a doom trade at a trade at mid price to close out our mm agent at the end of the episode.'''
            mid_trade = job.create_trade(
                price, quant, -666666,  self.trader_unique_id + state.customIDcounter+ 1 +self.n_actions, *time, -666666, self.trader_unique_id)
            trades = job.add_trade(trades, mid_trade)
            return trades
        
        trades = jax.lax.cond(
            ep_is_over & (jnp.abs(new_inventory) > 0),  # Check if episode is over and we still have remaining quantity
            place_midprice_trade,  # Place amidprice trade
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, mid_price, jnp.sign(state.inventory) * state.inventory, time  # Inv +ve means incoming is sell so standing buy.
        )#jnp.sign(state.inventory) * quant_still_left

        next_id = state.customIDcounter + self.n_actions + 1
        
        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant
    
    
    def _force_market_order_if_done(
            self,
            #quant_left: jax.Array,
            bestask: jax.Array,
            bestbid: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: EnvState,
            params: EnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:
        """ Force a market order if episode is over (either in terms of time or steps).
         Cancel all agent trades and place a market trade. If this is unmatched, cancel any remaing volume
          and place an artificial trade at a bad price. """
        
        
        
        
        def create_mkt_order():
            '''Create a market order by either placing a limit
            order at 0 or max int. Buy if inventory is less than zero and
            visa versa'''
            is_sell_task = jnp.where(state.inventory > 0, 1, 0)
            mkt_p = (1 - is_sell_task) * job.MAX_INT // self.tick_size * self.tick_size
            side = (1 - is_sell_task*2)
            # TODO: this addition wouldn't work if the ns time at index 1 increases to more than 1 sec
            new_time = time + params.time_delay_obs_act
            mkt_msg = jnp.array([
                # type, side, quant, price
                #NOTE: MAKING ZERO TO TEST SELL AT MID PRICE jnp.abs(state.inventory)
                1, side, 0 , mkt_p,
                self.trader_unique_id,
                self.trader_unique_id + state.customIDcounter + self.n_actions,  # unique order ID for market order
                *new_time,  # time of message
            ])
            next_id = state.customIDcounter + self.n_actions + 1
            return mkt_msg, next_id, new_time

        def create_dummy_order():
            '''To comply with fixed array constraints, 
            create a dummy trade when the episode is not over'''
            next_id = state.customIDcounter + self.n_actions
            return jnp.zeros((8,), dtype=jnp.int32), next_id, time 
        

        def place_doom_trade(trades, price, quant, time):
            '''Place a doom trade at a punishment price for any unmatched
            market order. If this is placed, the orderbook will be completly drained.'''
            doom_trade = job.create_trade(
                price, quant, -666666,  self.trader_unique_id + state.customIDcounter+ 1 +self.n_actions, *time, -666666, self.trader_unique_id)
            trades = job.add_trade(trades, doom_trade)
            return trades
         
        #-----check if ep over-----#
        if self.ep_type == 'fixed_time':
            remainingTime = params.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= 5  # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1

        #----filter the market or dummy order through---#
        order_msg, id_counter, time = jax.lax.cond(
            ep_is_over,
            create_mkt_order,
            create_dummy_order
        )
        #==============Cancel previous orders by the agent prior to the market order=========###
        #Cancel all previous agent orders before the market order so that we do not trade with ourselves.
        cnl_msg_bid = job.getCancelMsgs(
            bids,
            self.trader_unique_id,
            self.n_actions//2, 
            1  # bids
        )
        cnl_msg_ask = job.getCancelMsgs(
            asks,
            self.trader_unique_id,
            self.n_actions//2,
            -1  # ask side
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)
        
        (asks, bids, trades), (new_bestask, new_bestbid) = job.scan_through_entire_array_save_bidask(
            cnl_msgs, 
            (asks, bids, trades),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines
        )
   
        #Filter our new message through the orderbook#
        (asks, bids, trades), (new_bestask, new_bestbid) = job.cond_type_side_save_bidask(
            (asks, bids, trades),
            order_msg
        )
        
        # make sure best prices use the most recent available price and are not negative
        bestask = jax.lax.cond(
            new_bestask[0] <= 0,
            lambda: jnp.array([bestask[0], 0]),
            lambda: new_bestask,
        )
        bestbid = jax.lax.cond(
            new_bestbid[0] <= 0,
            lambda: jnp.array([bestbid[0], 0]),
            lambda: new_bestbid,
        )

        #==============Cancel previous orders by the agent prior to the market order=========###
        #Cancel all previous agent orders before the doom order. This avoids the "best bid" or " best ask"
        #corresponding to the left over market price#
        cnl_msg_bid = job.getCancelMsgs(
            bids,
            self.trader_unique_id,
            1, 
            1  # bids
        )
        cnl_msg_ask = job.getCancelMsgs(
            asks,
            self.trader_unique_id,
            1,
            -1  # ask side
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        (asks, bids, trades), (new_bestask, new_bestbid) = job.scan_through_entire_array_save_bidask(
            cnl_msgs, 
            (asks, bids, trades),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines
        )
       
        # make sure best prices use the most recent available price and are not negative
        bestask = jax.lax.cond(
            new_bestask[1][0] <= 0, #Price after second cancel message
            lambda: jnp.array([bestask[0], 0]),
            lambda: new_bestask[1],
        )
        bestbid = jax.lax.cond(
            new_bestbid[1][0] <= 0,
            lambda: jnp.array([bestbid[0], 0]),
            lambda: new_bestbid[1],
        )     

        ###TODO: check matching
        mkt_exec_quant = jnp.where(
            trades[:, 3] == order_msg[5],
            jnp.abs(trades[:, 1]),  # executed quantity
            0
        ).sum()        
        # assume execution at really unfavorable price if market order doesn't execute (worst case)
        # create artificial trades for this
        quant_still_left = jnp.abs(state.inventory) - mkt_exec_quant
       # jax.debug.print('quant_still_left: {}', quant_still_left)
        # assume doom price with 25% extra cost
        is_sell_task = jnp.where(state.inventory > 0, 1, 0)

        
        doom_price = jax.lax.cond(
            is_sell_task,
            #lambda: ((0.75 * bestbid[0]) // self.tick_size * self.tick_size).astype(jnp.int32),
            #lambda: ((1.25 * bestask[0]) // self.tick_size * self.tick_size).astype(jnp.int32),
            lambda: ((bestbid[0]+bestask[0])//2 // self.tick_size * self.tick_size).astype(jnp.int32),
            lambda: ((bestbid[0]+bestask[0])//2 // self.tick_size * self.tick_size).astype(jnp.int32), #For sell at opposite test
        )
        #jax.debug.print('ep_is_over: {}; quant_still_left: {}; remainingTime: {}; doom price :{}', ep_is_over, quant_still_left, remainingTime,doom_price)
        trades = jax.lax.cond(
            ep_is_over & (quant_still_left > 0),  # Check if episode is over and we still have remaining quantity
            place_doom_trade,  # Place a doom trade with unfavorable price
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, doom_price, 0, time  # Inv +ve means incoming is sell so standing buy.
        )#jnp.sign(state.inventory) * quant_still_left
        agent_trades = job.get_agent_trades(trades, self.trader_unique_id)
       # price_quants = self._get_executed_by_price(agent_trades)
        doom_quant = ep_is_over * quant_still_left

        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant

    def _get_reward(self, state: EnvState, params: EnvParams, trades: chex.Array,bestasks :chex.Array, bestbids: chex.Array) -> jnp.int32:
        '''Return the reward. There are a few options for reward funciton and assocaited hyper parameters:
        
        
        '''
        # ====================get reward and revenue ==========================================#
        # Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
             
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = (self.trader_unique_id == executed[:, 6]) | (self.trader_unique_id == executed[:, 7]) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        otherTrades = jnp.where(mask2[:, jnp.newaxis], 0, executed)
    

        #Find agent Buys and Agent sells from agent Trades:
        #The below mask puts passive buys or aggresive buys into "agent buys".
        #Logic: Q>0, TIDs=BUY; Q<0 TIDa= BUY
        mask_buy = (((agentTrades[:, 1] >= 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] < 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        mask_sell = (((agentTrades[:, 1] < 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] >= 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        agent_buys=jnp.where(mask_buy[:, jnp.newaxis], agentTrades, 0)
        agent_sells=jnp.where(mask_sell[:, jnp.newaxis], agentTrades, 0)



        #Find amount bought and sold in the step
        buyQuant=jnp.abs(agent_buys[:, 1]).sum()
        sellQuant=jnp.abs(agent_sells[:, 1]).sum()

        #Find total traded volume
        TradedVolume=buyQuant+sellQuant

        #Calculate the change in inventory & the new inventory
        inventory_delta = buyQuant - sellQuant
        new_inventory=state.inventory+inventory_delta

       
        #Find the new obsvered mid price at the end of the step.
        #Note: to make integer of tick_size // is integer division.
        mid_price_end = (bestbids[-1][0] + bestasks[-1][0]) // 2 // self.tick_size * self.tick_size
           
        #Inventory PnL: 
        InventoryPnL= state.inventory*(mid_price_end-state.mid_price) / self.tick_size 
    
        #Market Making PNL:     
        averageMidprice = ((bestbids[:, 0] + bestasks[:, 0]) // 2).mean() // self.tick_size * self.tick_size
        #jax.debug.print("averageMidprice:{}",averageMidprice)
        #TODO:Real PnL??+weighted inventory PnL
        buyPnL = ((averageMidprice - agent_buys[:, 0]) * jnp.abs(agent_buys[:, 1])).sum() / self.tick_size
        sellPnL = ((agent_sells[:, 0] - averageMidprice) * jnp.abs(agent_sells[:, 1])).sum() / self.tick_size

        #Lamda weighted, non directional#
        # Multiply PnL from inventory with small lambda to dampen the effect
        #reward=buyPnL+sellPnL + self.rewardLambda * InventoryPnL # Symmetrically dampened PnL

       

        # Other versions of reward
        #reward=buyPnL+sellPnL +InventoryPnL # full speculation
        #reward=buyPnL+sellPnL -jnp.abs(state.inventory)
        undamped_reward=buyPnL+sellPnL+InventoryPnL
        #reward=buyPnL+sellPnL-jnp.abs(state.inventory//10)
        #reward=buyPnL+sellPnL + InventoryPnL - (1-self.rewardLambda)*jnp.maximum(0,InventoryPnL//1000) # Asymmetrically dampened PnL
        #jax.debug.print("reward:{}",reward)
        #More complex reward function (should be added as part of the env if we actually use them):
        inventoryPnL_lambda = 0.002
        unrealizedPnL_lambda = 0
        asymmetrically_dampened_lambda = 0.05
        avg_buy_price = jnp.where(buyQuant > 0, (agent_buys[:, 0] / buyQuant * jnp.abs(agent_buys[:, 1])).sum(), 0)  
        avg_sell_price = jnp.where(sellQuant > 0, (agent_sells[:, 0]/ sellQuant * jnp.abs(agent_sells[:, 1])).sum() , 0)
        approx_realized_pnl = jnp.minimum(buyQuant, sellQuant) * (avg_sell_price - avg_buy_price) / self.tick_size
        approx_unrealized_pnl = jnp.where( 
            inventory_delta > 0,
            inventory_delta * (averageMidprice - avg_buy_price) / self.tick_size,  # Excess buys
            jnp.abs(inventory_delta) * (avg_sell_price - averageMidprice) / self.tick_size # Excess sells
        )
  
        #reward = approx_realized_pnl + unrealizedPnL_lambda * approx_unrealized_pnl +  inventoryPnL_lambda * jnp.minimum(InventoryPnL,InventoryPnL*asymmetrically_dampened_lambda) #Last term adds negative inventory PnL without dampening
       
        reward= -jnp.abs(state.inventory)
        

        # Define a penalty if he exceeds a certain inventory
        penalty_threshold = 100.0
        penalty_amount = 500.0 
        penalty = jnp.where(jnp.abs(state.inventory) > penalty_threshold, penalty_amount, 0.0)
        #reward = reward - penalty
        
        
        
        
        
        #Real Revenue calcs: (actual cash flow+actual value of portfolio)
        income=(agent_sells[:, 0]/ self.tick_size* jnp.abs(agent_sells[:, 1])).sum() 
        outgoing=(agent_buys[:, 0] / self.tick_size* jnp.abs(agent_buys[:, 1])).sum() 
             
        PnL=(income-outgoing)/self.tick_size
       
        #calculate a fraction of total market activity attributable to us.
        other_exec_quants = jnp.abs(otherTrades[:, 1]).sum()
        market_share = TradedVolume / (TradedVolume + other_exec_quants)

        # ---------- normalize the reward ----------#
        # reward /= 10_000
        reward_scaled = reward / 1000
        #reward_scaled = jnp.clip(reward_scaled, -0.1, 0.1)
        # reward /= params.avg_twap_list[state.window_index]
        #jax.debug.print("new_inventory:{}",new_inventory)
        return reward, {
            "market_share": market_share,
            "undamped_reward":undamped_reward,
            "PnL": PnL, 
            "end_inventory":new_inventory,
            "mid_price":mid_price_end,
            "agentQuant":inventory_delta,
            "buyQuant":buyQuant,
            "sellQuant":sellQuant,
            "approx_realized_pnl":approx_realized_pnl,
            "approx_unrealized_pnl" : approx_unrealized_pnl,
            "InventoryPnL":InventoryPnL,
            "other_exec_quants":other_exec_quants,
            "averageMidprice": averageMidprice
        }

    def _get_obs(
            self,
            state: EnvState,
            params: EnvParams,
            normalize: bool = True,
            flatten: bool = True,
        ) -> chex.Array:
        """ Return observation from raw state trafo. """
        # NOTE: only uses most recent observation from state
        time = state.time[0] + state.time[1]/1e9
        time_elapsed = time - (state.init_time[0] + state.init_time[1]/1e9)
        obs = {
            "p_bid" : state.best_bids[-1][0],  
            "p_ask": state.best_asks[-1][0], 
            "p_bid_passive" :  state.bid_passive_2,
            "p_ask_passive" :  state.ask_passive_2,
            "spread": jnp.abs(state.best_asks[-1][0] - state.best_bids[-1][0]),
            "q_bid": state.best_bids[-1][1],
            "q_ask": state.best_asks[-1][1],
            "q_bid_passive": state.quant_bid_passive_2,
            "q_ask_passive" : state.quant_ask_passive_2,
            # "q_before2": None, # how much quantity lies above this price level
            "time": time,
            "delta_time": state.delta_time,
            # "episode_time": state.time - state.init_time,
            "time_remaining": params.episode_time - time_elapsed,
            "inventory" : state.inventory,
            "init_price": state.init_price,
            "mid_price":state.mid_price,
            "total_PnL" : state.total_PnL,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
            "remaining_ratio": jnp.where(state.max_steps_in_episode==0, 0., 1. - state.step_counter / state.max_steps_in_episode),
            "prev_action": state.prev_action,  # use quants only
            "prev_executed": state.prev_executed,  # use quants only
            "prev_executed_ratio": jnp.where(state.prev_action==0., 0., state.prev_executed /10)# state.prev_action[:, 1]), Hard code size of normal trade
            
        }

        # TODO: put this into config somewhere?
        #       also check if we can get rid of manual normalization
        #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
        p_mean = 3.5e7
        p_std = 1e6
        means = {
            "p_bid": state.mid_price,
            "p_ask": state.mid_price,
            "p_bid_passive" :  state.mid_price,
            "p_ask_passive" :  state.mid_price,
            "spread": 0,
            "q_bid": 0,
            "q_ask": 0,
            "q_bid_passive": 0,
            "q_ask_passive" : 0,
            "time": 0,
            "delta_time": 0,
            # "episode_time": jnp.array([0, 0]),
            "time_remaining": 0,
            "inventory" : 0,
            "init_price": 0, #p_mean,
            "mid_price":0,
            "total_PnL" : 0,
            #"task_size": 0,
           # "executed_quant": 0,
            #"remaining_quant": 0,
            "step_counter": 0,
            "max_steps": 0,
            "remaining_ratio": 0,
            "prev_action": 0,
            "prev_executed": 0,
           "prev_executed_ratio": 0,
        
        }
        stds = {
            #"is_sell_task": 1,
            "p_bid": 1e5, #p_std,
            "p_ask": 1e5, #p_std,
            "p_bid_passive" :  1e5,
            "p_ask_passive" : 1e5,
            "spread": 1e4,
            "q_bid": 100,
            "q_ask": 100,
            "q_bid_passive": 100,
            "q_ask_passive" : 100,
            "time": 1e5,
            "delta_time": 10,
            # "episode_time": jnp.array([1e3, 1e9]),
            "time_remaining": self.sliceTimeWindow, # 10 minutes = 600 seconds
            "init_price": 1e7, #p_std,
            "mid_price": 1e7, #p_std,
            "inventory" : 100,
            "total_PnL" : 100,
            #"task_size": self.max_task_size,
           # "executed_quant": self.max_task_size,
           # "remaining_quant": self.max_task_size,
            "step_counter": 30,  # TODO: find way to make this dependent on episode length
            "max_steps": 30,
            "remaining_ratio": 1,
            "prev_action": 10,
            "prev_executed": 10,
            "prev_executed_ratio": 1,
        }
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)
        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        return obs


##BELOW HAS NOT BEEN UPDATED FOR MM_ENV
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
            
            return spaces.Discrete(self.n_actions)
            #return spaces.Box(0, 2, (self.n_actions,), dtype=jnp.int32)
    
       

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        #space = spaces.Box(-10,10,(809,),dtype=jnp.float32) 
        # space = spaces.Box(-10, 10, (21,), dtype=jnp.float32) 
        space = spaces.Box(-10, 10, (29,), dtype=jnp.float32) 
        return space

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return NotImplementedError



    
    

# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "buy",
        "MAX_TASK_SIZE": 100,
        "WINDOW_INDEX": 300,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 0.1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 240,  # 
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=MarketMakingEnv(ATFolder,"sell",1)
    env = MarketMakingEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=0.00001,
        #task_size=config["MAX_TASK_SIZE"],
        episode_time=config["EPISODE_TIME"],  # in seconds
    )
    # print(env_params.message_data.shape, env_params.book_data.shape)


    start=time.time()
    obs,state=env.reset(key_reset, env_params)
    print("Time for reset: \n",time.time()-start)

    #print("State after reset: \n",state)
    print("Inventory after reset: \n",state.inventory)
    

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,15):
         # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        #test_action=env.action_space().sample(key_policy)
        test_action = env.action_space().sample(key_policy) 
        jax.debug.print("test_action :{}",test_action)
        #test_action=0
        #env.action_space().sample(key_policy) // 10
        # test_action = jnp.array([100, 10])
        print(f"Sampled {i}th actions are: ", test_action)
        start=time.time()
        
        obs, state, reward, done, info = env.step(
            key_step, state, test_action, env_params)
        
     #   print('revenue',state.total_revenue)
        #print('revenue', state.total_revenue)
        #print('inventory',state.inventory)
        #print('reward',reward)
        #
       # print("Reward: \n",reward)
       # print("Time \n", state.time)
        #print("Intial Time \n", state.init_time)
        #for key, value in info.items():
           #print(key, value)
            
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
            exit()
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================




    # # ####### Testing the vmap abilities ########
    
    enable_vmap=False
    if enable_vmap:
        # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 1024
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)


        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys,
         state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
