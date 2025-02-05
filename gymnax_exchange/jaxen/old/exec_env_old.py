"""
Execution Environment for Limit Order Book

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
ExecutionEnv: Environment class inheriting from BaseLOBEnv, 
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
is_terminal:        Checks if the current state meets the terminal condition, 
                    specific to execution tasks, such as completion of the 
                    execution order or time constraints.
get_obs:            Constructs and returns the current observation for the 
                    execution environment, derived from the state.
name, num_actions:  Inherited methods providing the name of the environment 
                    and the number of possible actions.
action_space:       Defines the action space for execution tasks, including 
                    order types and quantities.
observation_space:  Define the observation space for execution tasks.
state_space:        Describes the state space of the environment, tailored 
                    for execution tasks with components 
                    like bids, asks, and trades.
reset_env:          Resets the environment to a specific state for execution. 
                    It selects a new data window, initializes the order book, 
                    and sets the initial state for execution tasks.
is_terminal:        Checks whether the current state is terminal, based on 
                    the number of steps executed or tasks completed.
getActionMsgs:      Generates action messages based on 
                    the current state and action. 
                    It determines the type, side, quantity, 
                    and price of orders to be executed.
                    including detailed order book information and trade history
hamilton_apportionment_permuted_jax: A utility function using JAX, 
                                     implementing a Hamilton apportionment 
                                     method with randomized seat allocation.
_get_initial_time:  Inherited method to retrieve the 
                    initial time of a data window.
_get_data_messages: Inherited method to fetch market messages for a given 
                    step within a data window.
"""


# from jax import config
# config.update("jax_enable_x64",True)
# ============== testing scripts ===============
import os
import sys
import time 
import timeit
import random
# sys.path.append('/Users/sasrey/AlphaTrade')
# sys.path.append('/homes/80/kang/AlphaTrade')
sys.path.append('.')
import jax
import jax.numpy as jnp


import gymnax
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
import chex

import faulthandler
faulthandler.enable()

# from jax import config
# config.update('jax_platform_name', 'cpu')
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())
chex.assert_gpu_available(backend=None)

# #Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) # use this during training
# config.update("jax_disable_jit", True)
# ============== testing scripts ===============
jax.numpy.set_printoptions(linewidth=183)


from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, flatten_util
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
import pickle
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from AlphaTrade.gymnax_exchange.jaxen.old.base_env_old import BaseLOBEnv
from AlphaTrade.gymnax_exchange.jaxen.old.base_env_old import EnvParams as BaseEnvParams
from AlphaTrade.gymnax_exchange.jaxen.old.base_env_old import EnvState as BaseEnvState
import dataclasses

import jax.tree_util as jtu
def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

def array_index(array,index):
    return array[index]

@jax.jit
def index_tree(tree,index):
    array_index = lambda array,index : array[index]
    indeces=[index]*len(jtu.tree_flatten(tree)[0])
    tree_indeces=jtu.tree_unflatten(jtu.tree_flatten(tree)[1],indeces)
    return jtu.tree_map(array_index,tree,tree_indeces)
    


# import utils

@struct.dataclass
class EnvState(BaseEnvState):
    # Potentially could be moved to base,
    # so long as saving of best ask/bids is base behaviour. 
    best_asks: chex.Array
    best_bids: chex.Array
    # Execution specific stuff
    init_price:int
    task_to_execute:int
    quant_executed:int
    total_revenue:float
    # Execution specific rewards. 
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float


@struct.dataclass
class EnvParams(BaseEnvParams):
    is_sell_task: int
    initstateArray: chex.Array    


class ExecutionEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, task, window_index, action_type,
            task_size = 500, rewardLambda=0.0,data_type="fixed_time"):
        print(alphatradePath,window_index,data_type)
        super().__init__(alphatradePath,window_index,data_type)
        self.n_actions = 4 # [FT, M, NT, PP]
        self.n_ticks_in_book = 2 # Depth of PP actions
        self.task = task # "random", "buy", "sell"
        self.action_type = action_type # 'delta' or 'pure'
        self.rewardLambda = rewardLambda
        self.task_size = task_size # num to sell or buy for the task
                

        print("START:  pre-reset in the initialization")
        pkl_file_name = (alphatradePath
                         + 'stateArray_idx_'+ str(window_index)
                         +'_dtype_"'+data_type
                         +'"_depth_'+str(self.book_depth)
                         +'.pkl')
        print("pre-reset will be saved to ",pkl_file_name)
        try:
            with open(pkl_file_name, 'rb') as f:
                self.initstateArray = pickle.load(f)
            print("LOAD FROM PKL")
        except:
            print("DO COMPUTATION")
            states = [self._get_state_from_data(self.messages[i],
                                                self.books[i],
                                                self.max_steps_in_episode_arr[i]) 
                        for i in range(self.n_windows)]
            self.initstateArray=tree_stack(states)
            with open(pkl_file_name, 'wb') as f:
                pickle.dump(self.initstateArray, f) 
        print("FINISH: pre-reset in the initialization")

        #Theoretically, this entire thing above here could be encapsulated in a function 
        # and moved out of the init function. 
        # Additionally, it should probably be seperated such that the base env calculates
        # those sections of the state that are commmon, and only the state vars unique to
        # the exec are calculated here. 


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        is_sell_task = 0 if self.task == 'buy' else 1 # if self.task == 'random', set defualt as 0
        base_params=super().default_params
        base_vals=jtu.tree_flatten(base_params)[0]
        return EnvParams(*base_vals,is_sell_task,self.initstateArray)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        # '''
        # action = jnp.array([delta,0,0,0],dtype=jnp.int32)
        action = self._reshape_action(action, state, params,key)        
        
        data_messages = self._get_data_messages(params.message_data,state.window_index,state.step_counter)
        
        action_msgs = self._getActionMsgs(action, state, params)
        #Currently just naive cancellation of all agent orders in the book. #TODO avoid being sent to the back of the queue every time. 
        raw_order_side = jax.lax.cond(params.is_sell_task,
                                      lambda: state.ask_raw_orders,
                                      lambda: state.bid_raw_orders)
        cnl_msgs = job.getCancelMsgs(raw_order_side,
                                     job.INITID + 1,
                                     self.n_actions,
                                     1 - params.is_sell_task * 2)
        

        #Add to the top of the data messages
        total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0)
        #Save time of final message to add to state
        time=total_messages[-1, -2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines) 

        
        # ========== get reward and revenue ==========
        # Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = ((job.INITID < executed[:, 2]) & (executed[:, 2] < 0)) | ((job.INITID < executed[:, 3]) & (executed[:, 3] < 0))
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        agentQuant = agentTrades[:,1].sum() # new_execution quants
        # ---------- used for vwap, revenue ----------
        vwapFunc = lambda executed: jnp.nan_to_num((executed[:,0]//self.tick_size* executed[:,1]).sum()/(executed[:,1]).sum(),0.0) # caution: this value can be zero (executed[:,1]).sum()
        vwap = vwapFunc(executed) # average_price of all the tradings, from the varaible executed
        revenue = (agentTrades[:,0]//self.tick_size * agentTrades[:,1]).sum()
        # ---------- used for slippage, price_drift, and RM(rolling mean) ----------
        rollingMeanValueFunc_FLOAT = lambda average_price,new_price:(average_price*state.step_counter+new_price)/(state.step_counter+1)
        vwap_rm = rollingMeanValueFunc_FLOAT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)
        price_adv_rm = rollingMeanValueFunc_FLOAT(state.price_adv_rm,revenue/agentQuant - vwap) # slippage=revenue/agentQuant-vwap, where revenue/agentQuant means agentPrice 
        slippage_rm = rollingMeanValueFunc_FLOAT(state.slippage_rm,revenue - state.init_price//self.tick_size*agentQuant)
        price_drift_rm = rollingMeanValueFunc_FLOAT(state.price_drift_rm,(vwap - state.init_price//self.tick_size)) #price_drift = (vwap - state.init_price//self.tick_size)
        # ---------- used for advantage and drift ----------
        advantage = revenue - vwap * agentQuant # advantage_vwap
        drift = agentQuant * (vwap - state.init_price//self.tick_size)
        # drift = agentQuant * (vwap_rm - state.init_price//self.tick_size)
        # ---------- compute the final reward ----------
        rewardValue = revenue 
        # rewardValue =  advantage
        # rewardValue1 = advantage + self.rewardLambda * drift
        # rewardValue1 = advantage + 1.0 * drift
        # rewardValue2 = revenue - (state.init_price // self.tick_size) * agentQuant
        # rewardValue = rewardValue1 - rewardValue2
        # rewardValue = revenue - vwap_rm * agentQuant # advantage_vwap_rm
        reward = jnp.sign(agentQuant) * rewardValue # if no value agentTrades then the reward is set to be zero
        jax.debug.print("reward {}", reward)
        # ---------- normalize the reward ----------
        reward /= 10000
        # reward /= params.avg_twap_list[state.window_index]
        # ========== get reward and revenue END ==========
        

        (bestasks,
        bestbids) = (self._best_prices_impute(bestasks[-self.stepLines:],
                                                       state.best_asks[-1,0]),
                              self._best_prices_impute(bestbids[-self.stepLines:],
                                                       state.best_bids[-1,0]))
        state = EnvState(ask_raw_orders=asks,
                         bid_raw_orders=bids,
                         trades=trades,
                         init_time=state.init_time,
                         time=time,
                         customIDcounter=state.customIDcounter+self.n_actions,
                         window_index=state.window_index,
                         step_counter=state.step_counter+1,
                         max_steps_in_episode=state.max_steps_in_episode,
                         best_asks=bestasks,
                         best_bids=bestbids,
                         init_price=state.init_price,
                         task_to_execute=state.task_to_execute,
                         quant_executed=state.quant_executed+agentQuant,
                         total_revenue=state.total_revenue+revenue,
                         slippage_rm=slippage_rm,
                         price_adv_rm=price_adv_rm,
                         price_drift_rm=price_drift_rm,
                         vwap_rm=vwap_rm)
        done = self.is_terminal(state, params)
        info={"window_index": state.window_index,
                "total_revenue": state.total_revenue,
                "quant_executed": state.quant_executed,
                "task_to_execute": state.task_to_execute,
                "average_price": jnp.nan_to_num(state.total_revenue 
                                                / state.quant_executed, 0.0),
                "current_step": state.step_counter,
                'done': done,
                'slippage_rm': state.slippage_rm,
                "price_adv_rm": state.price_adv_rm,
                "price_drift_rm": state.price_drift_rm,
                "vwap_rm": state.vwap_rm,
                "advantage_reward": advantage,}
        return self._get_obs(state, params), state, reward, done,info
    

    def reset_env(self,
                  key : chex.PRNGKey,
                  params: EnvParams) -> Tuple[chex.Array, EnvState]:
        #Only sampling of window idx is managed by Base Env.
        _,dummy_state=super().reset_env(key,params)
        #Load state from pre-calced tree of init states for all windows
        state = index_tree(params.initstateArray,
                           dummy_state.window_index)

        key_, key = jax.random.split(key)
        if self.task == 'random':
            direction = jax.random.randint(key_, minval=0, maxval=2, shape=())
            params = dataclasses.replace(params, is_sell_task=direction)
        return self._get_obs(state, params),state
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return((state.max_steps_in_episode - state.step_counter<= 0) |
                (state.task_to_execute - state.quant_executed <= 0))
    
    def _getActionMsgs(self, action: Dict, state: EnvState, params: EnvParams):
        def normal_order_logic(action: jnp.ndarray):
            quants = action.astype(jnp.int32) # from action space
            return quants

        def market_order_logic(state: EnvState):
            quant = state.task_to_execute - state.quant_executed
            quants = jnp.asarray((quant, 0, 0, 0), jnp.int32) 
            return quants

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        # sides = (params.is_buy_task*2 - 1) * jnp.ones((self.n_actions,),jnp.int32)
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
        NT, FT, PP, MKT = jax.lax.cond(
            params.is_sell_task,
            lambda: (best_ask, best_bid, best_ask + self.tick_size*self.n_ticks_in_book, 0),
            lambda: (best_bid, best_ask, best_bid - self.tick_size*self.n_ticks_in_book, job.MAX_INT)
        )
        M = ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size # Mid price
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        # ---------- ifMarketOrder BGN ----------
        # ·········· ifMarketOrder determined by time ··········
        # remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        # marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
        # ifMarketOrder = (remainingTime <= marketOrderTime)
        # ·········· ifMarketOrder determined by steps ··········
        remainingSteps = state.max_steps_in_episode - state.step_counter 
        marketOrderSteps = jnp.array(1, dtype=jnp.int32) # in steps, means the last step was left for market order
        ifMarketOrder = (remainingSteps <= marketOrderSteps)
        # ---------- ifMarketOrder END ----------
        def normal_order_logic(state: EnvState, action: jnp.ndarray):
            quants = action.astype(jnp.int32) # from action space
            prices = jnp.asarray((FT,NT), jnp.int32) if self.n_actions == 2 else jnp.asarray((FT,M,NT,PP), jnp.int32) 
            return quants, prices
        def market_order_logic(state: EnvState):
            quant = state.task_to_execute - state.quant_executed
            quants =  jnp.asarray((quant,0),jnp.int32) if self.n_actions == 2 else jnp.asarray((quant,0,0,0),jnp.int32) 
            prices =  jnp.asarray((MKT,  M),jnp.int32) if self.n_actions == 2 else jnp.asarray((MKT, M,M,M),jnp.int32)
            return quants, prices
        market_quants, market_prices = market_order_logic(state)
        normal_quants, normal_prices = normal_order_logic(state, action)
        quants = jnp.where(ifMarketOrder, market_quants, normal_quants)
        prices = jnp.where(ifMarketOrder, market_prices, normal_prices)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================


    def _get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        print(state.best_asks)
        best_asks, best_bids=state.best_asks[:,0], state.best_bids[:,0]
        best_ask_qtys, best_bid_qtys = state.best_asks[:,1], state.best_bids[:,1]
        
        obs = {
            "is_sell_task": params.is_sell_task,
            "p_aggr": jnp.where(params.is_sell_task, best_bids, best_asks),
            "q_aggr": jnp.where(params.is_sell_task, best_bid_qtys, best_ask_qtys), 
            "p_pass": jnp.where(params.is_sell_task, best_asks, best_bids),
            "q_pass": jnp.where(params.is_sell_task, best_ask_qtys, best_bid_qtys), 
            "p_mid": (best_asks+best_bids)//2//self.tick_size*self.tick_size, 
            "p_pass2": jnp.where(params.is_sell_task, best_asks+self.tick_size*self.n_ticks_in_book, best_bids-self.tick_size*self.n_ticks_in_book), # second_passives
            "spread": best_asks - best_bids,
            "shallow_imbalance": state.best_asks[:,1]- state.best_bids[:,1],
            "time": state.time,
            "episode_time": state.time - state.init_time,
            "init_price": state.init_price,
            "task_size": state.task_to_execute,
            "executed_quant": state.quant_executed,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
        }

        def normalize_obs(obs: Dict[str, jax.Array]):
            """ normalized observation by substracting 'mean' and dividing by 'std'
                (config values don't need to be actual mean and std)
            """
            # TODO: put this into config somewhere?
            #       also check if we can get rid of manual normalization
            #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
            p_mean = 3.5e7
            p_std = 1e6
            means = {
                "is_sell_task": 0,
                "p_aggr": p_mean,
                "q_aggr": 0,
                "p_pass": p_mean,
                "q_pass": 0,
                "p_mid":p_mean,
                "p_pass2":p_mean,
                "spread": 0,
                "shallow_imbalance":0,
                "time": jnp.array([0, 0]),
                "episode_time": jnp.array([0, 0]),
                "init_price": p_mean,
                "task_size": 0,
                "executed_quant": 0,
                "step_counter": 0,
                "max_steps": 0,
            }
            stds = {
                "is_sell_task": 1,
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
                "executed_quant": 500,
                "step_counter": 300,
                "max_steps": 300,
            }
            obs = jax.tree_map(lambda x, m, s: (x - m) / s, obs, means, stds)
            return obs


        obs = normalize_obs(obs)
        # jax.debug.print("obs {}", obs)
        obs, _ = jax.flatten_util.ravel_pytree(obs)
        # jax.debug.breakpoint()
        return obs
    
    def _get_state_from_data(self,message_data,book_data,max_steps_in_episode):
        base_state=super()._get_state_from_data(message_data,book_data,max_steps_in_episode)
        base_vals=jtu.tree_flatten(base_state)[0]
        best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(base_state.ask_raw_orders,base_state.bid_raw_orders)
        M = (best_bid[0] + best_ask[0])//2//self.tick_size*self.tick_size 
        print('best bid: ', best_bid)
        return EnvState(*base_vals,
                        best_asks=jnp.resize(best_ask,(self.stepLines,2)),
                        best_bids=jnp.resize(best_bid,(self.stepLines,2)),
                        init_price=M,
                        task_to_execute=self.task_size,
                        quant_executed=0,
                        total_revenue=0,
                        slippage_rm=0.,
                        price_adv_rm=0.,
                        price_drift_rm=0.,
                        vwap_rm=0.)

    def _reshape_action(self, action : Dict, state: EnvState, params : EnvParams,key : chex.PRNGKey):
            if self.action_type == 'delta':
                def twapV3(state, env_params):
                    # ---------- ifMarketOrder BGN ----------
                    # ·········· ifMarketOrder determined by time ··········
                    # remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                    # marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                    # ifMarketOrder = (remainingTime <= marketOrderTime)
                    # ·········· ifMarketOrder determined by steps ··········
                    remainingSteps = state.max_steps_in_episode - state.step_counter 
                    marketOrderSteps = jnp.array(1, dtype=jnp.int32) 
                    ifMarketOrder = (remainingSteps <= marketOrderSteps)
                    # ---------- ifMarketOrder END ----------
                    # ---------- quants ----------
                    remainedQuant = state.task_to_execute - state.quant_executed
                    remainedStep = state.max_steps_in_episode - state.step_counter
                    stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                    limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True) if self.n_actions == 2 \
                        else jax.random.permutation(key, jnp.array([stepQuant-3*stepQuant//4,stepQuant//4,stepQuant//4,stepQuant//4]), independent=True)
                    market_quants = jnp.array([stepQuant,stepQuant]) if self.n_actions == 2 else jnp.array([stepQuant,stepQuant,stepQuant,stepQuant])
                    quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                    # ---------- quants ----------
                    # jax.debug.breakpoint()
                    return jnp.array(quants) 
                action_space_clipping = lambda action, task_size: jnp.round(action).astype(jnp.int32).clip(-1*task_size//100,task_size//100) 
                action_ = twapV3(state, params) + action_space_clipping(action, state.task_to_execute)
            else:
                action_space_clipping = lambda action, task_size: jnp.round(action).astype(jnp.int32).clip(0,task_size//5)# clippedAction, CAUTION not clipped by task_size, but task_size//5
                action_ = action_space_clipping(action, state.task_to_execute)
            
            def truncate_action(action, remainQuant):
                action = jnp.round(action).astype(jnp.int32).clip(0,remainQuant)
                # NOTE: didn't know this was already implemented? <= posted in channel 2 days(8th Nov) before your commit(10th Nov)
                # NOTE: already add comments and variable names to make it readable
                # NOTE: same thing with Peer's commit, and the logic from our discussion proposed by Sascha
                #       but hamilton_apportionment_permuted_jax add permutation if two poistions shares the same probability
                #       such as [2,1,0,1] and remain 2, in the utils.clip_by_sum_int 
                #       implementation, the second position will always be added by 1
                #       it will turns out to be [3,2,0,1]
                #       but in reality we should permutate between the second and fourth position
                #       it will turns out to be [3,2,0,1] or [3,1,0,2]
                #       the name hamilton_apportionment means our interger split problem, a classical math problem
                #       permuted in the name means randomly choose a position if two or more positions 
                #       shares the same probability.
                #       jax in the name means it is jittable
                # TODO  delete this comment after read
                scaledAction = jnp.where(action.sum() <= remainQuant, action, self._hamilton_apportionment_permuted_jax(action, remainQuant, key)) 
                return scaledAction
            action = truncate_action(action_, state.task_to_execute - state.quant_executed)
            return action.astype(jnp.int32)
    
    def _best_prices_impute(self,bestprices,lastBestPrice):
        def replace_values(prev, curr):
            last_non_999999999_values = jnp.where(curr != 999999999, curr, prev) #non_999999999_mask
            replaced_curr = jnp.where(curr == 999999999, last_non_999999999_values, curr)
            return last_non_999999999_values, replaced_curr
        def forward_fill_999999999_int(arr):
            last_non_999999999_values, replaced = jax.lax.scan(replace_values, arr[0], arr[1:])
            return jnp.concatenate([arr[:1], replaced])
        def forward_fill(arr):
            index = jnp.argmax(arr[:, 0] != 999999999)
            return forward_fill_999999999_int(arr.at[0, 0].set(jnp.where(index == 0, arr[0, 0], arr[index][0])))
        back_fill = lambda arr: jnp.flip(forward_fill(jnp.flip(arr, axis=0)), axis=0)
        mean_forward_back_fill = lambda arr: (forward_fill(arr)+back_fill(arr))//2     
        return jnp.where((bestprices[:,0] == 999999999).all(),jnp.tile(jnp.array([lastBestPrice, 0]), (bestprices.shape[0],1)),mean_forward_back_fill(bestprices))

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""

        return spaces.Box(-5,5,(self.n_actions,),dtype=jnp.int32) if self.action_type=='delta' \
          else spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32)

    
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return NotImplementedError

   
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions
    
    def _hamilton_apportionment_permuted_jax(self, votes, seats, key):
        """
        Compute the Hamilton apportionment method with permutation using JAX.

        Args:
            votes (jax.Array): Array of votes for each party/entity.
            seats (int): Total number of seats to be apportioned.
            key (chex.PRNGKey): JAX key for random number generation.

        Returns:
            jax.Array: Array of allocated seats to each party/entity.
        """
        std_divisor = jnp.sum(votes) / seats # Calculate the standard divisor.
        # Initial allocation of seats based on the standard divisor and compute remainders.
        init_seats, remainders = jnp.divmod(votes, std_divisor)
        # Compute the number of remaining seats to be allocated.
        remaining_seats = jnp.array(seats - init_seats.sum(), dtype=jnp.int32) 
        # Define the scanning function for iterative seat allocation.
        def allocate_remaining_seats(carry,x): # only iterate 4 times, as remaining_seats in {0,1,2,3}
            key,init_seats,remainders = carry
            key, subkey = jax.random.split(key)
            # Create a probability distribution based on the maximum remainder.
            distribution = (remainders == remainders.max())/(remainders == remainders.max()).sum()
            # Randomly choose a party/entity to allocate a seat based on the distribution.
            chosen_index = jax.random.choice(subkey, remainders.size, p=distribution)
            # Update the initial seats and remainders for the chosen party/entity.
            updated_init_seats = init_seats.at[chosen_index].add(jnp.where(x < remaining_seats, 1, 0))
            updated_remainders = remainders.at[chosen_index].set(0)
            return (key, updated_init_seats, updated_remainders), x
            # Iterate over parties/entities to allocate the remaining seats.
        (key, init_seats, remainders), _ = jax.lax.scan(
                                                        allocate_remaining_seats,
                                                        (key, init_seats, remainders), 
                                                        xs=jnp.arange(votes.shape[0])
                                                        )
        return init_seats

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
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay/"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "random", # "sell",
        "TASK_SIZE": 100, # 500,
        "WINDOW_INDEX": -1,
        "ACTION_TYPE": "delta", # "pure",
        "REWARD_LAMBDA": 1.0,
        "DTAT_TYPE":"fixed_time",
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)




    

    # env=ExecutionEnv(ATFolder,"sell",1)
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["WINDOW_INDEX"],config["ACTION_TYPE"],config["TASK_SIZE"],config["REWARD_LAMBDA"],config["DTAT_TYPE"])
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
    print(state)

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ =  jax.random.split(key_policy, 2)
        key_step, _ =  jax.random.split(key_step, 2)
        # test_action=env.action_space().sample(key_policy)
        test_action=env.action_space().sample(key_policy)
        # test_action=env.action_space().sample(key_policy)//10 # CAUTION not real action
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        for key, value in info.items():
            print(key, value)
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================
        
        
        

    # # ####### Testing the vmap abilities ########
    
    enable_vmap=True
    if enable_vmap:
        # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
