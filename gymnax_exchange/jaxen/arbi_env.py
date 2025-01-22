# from jax import config
# config.update("jax_enable_x64",True)

# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import timeit

import faulthandler
faulthandler.enable()

print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

# #Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)

import random
# ============== testing scripts ===============



from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
from AlphaTrade.gymnax_exchange.jaxen.old.base_env_old import BaseLOBEnv
# from gymnax_exchange.test_scripts.comparison import twapV3
import time 

@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index:int
    init_price:int
    task_to_execute:int
    quant_executed:int
    total_revenue:int
    step_counter: int
    max_steps_in_episode: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    stateArray_list: chex.Array
    obs_sell_list: chex.Array
    obs_buy_list: chex.Array
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    # max_steps_in_episode: int = 100 # TODO should be a variable, decied by the data_window
    # messages_per_step: int=1 # TODO never used, should be removed?
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    avg_twap_list=jnp.array([312747.47,
                            312674.06,
                            313180.38,
                            312813.25,
                            312763.78,
                            313094.1,
                            313663.97,
                            313376.72,
                            313533.25,
                            313578.9,
                            314559.1,
                            315201.1,
                            315190.2])
    


class ExecutionEnv(BaseLOBEnv):
    # def __init__(self,alphatradePath,task,task_size = 500, Lambda=0.0, Gamma=0.00):
    def __init__(self,alphatradePath,task,window_index,task_size = 500, Lambda=0.0, Gamma=0.00):
        super().__init__(alphatradePath)
        self.n_actions = 2 # [A, MASKED, P, MASKED] Agressive, MidPrice, Passive, Second Passive
        # self.n_actions = 2 # [MASKED, MASKED, P, PP] Agressive, MidPrice, Passive, Second Passive
        # self.n_actions = 4 # [A, M, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.task = task
        self.window_index =window_index
        self.Lambda = Lambda
        self.Gamma = Gamma
        # self.task_size = 5000 # num to sell or buy for the task
        # self.task_size = 2000 # num to sell or buy for the task
        self.task_size = task_size # num to sell or buy for the task
        # self.task_size = 200 # num to sell or buy for the task
        self.n_fragment_max=2
        self.n_ticks_in_book=20 
        # self.debug : bool = False


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        # return EnvParams(self.messages,self.books)
        return EnvParams(self.messages,self.books,self.stateArray_list,self.obs_sell_list,self.obs_buy_list)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, delta: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        # '''
        def reshape_action(action, state):
            def truncate_action(action, remainQuant):
                action = jnp.round(action).astype(jnp.int32).clip(-100,min(100,self.task_size))
                scaledAction = jnp.where(action.sum() > remainQuant, jnp.round(action * remainQuant / action.sum()).astype(jnp.int32), action)
                return scaledAction
            action = truncate_action(action, state.task_to_execute-state.quant_executed)
            jax.debug.print("action before clip {}",action)
            action = action.clip(-1*state.quant_executed,None).astype(jnp.int32)
            condition = action.sum() < -1 * state.quant_executed
            new_action = jnp.ceil(action * state.quant_executed / jnp.abs(action.sum())).astype(jnp.int32)
            action = jnp.where(condition, new_action, action)
            jax.debug.print("quant for clip {}",-1*state.quant_executed)
            jax.debug.print("action after clip {}",action)
            return action
        action = reshape_action(delta, state)
        
        
        data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        
        action_msgs = self.getActionMsgs(action, state, params)
        # jax.debug.print("action_msgs {}",action_msgs)
        #Currently just naive cancellation of all agent orders in the book. #TODO avoid being sent to the back of the queue every time. 
        cnl_msgs=job.getCancelMsgs(state.ask_raw_orders if self.task=='sell' else state.bid_raw_orders,-8999,self.n_actions,-1 if self.task=='sell' else 1)
        #Add to the top of the data messages
        total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0) # TODO DO NOT FORGET TO ENABLE CANCEL MSG
        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process messages of step (action+data) through the orderbook
        # jax.debug.breakpoint()
        asks,bids,trades,bestasks,bestbids=job.scan_through_entire_array_save_bidask(total_messages,(state.ask_raw_orders,state.bid_raw_orders,trades_reinit),self.stepLines) 
        # jax.debug.print("bestasks {}", bestbids)
        # jax.debug.breakpoint()
        
        # ========== get reward and revenue ==========
        executed = jnp.where((trades[:, 0] > 0)[:, jnp.newaxis], trades, 0)
        mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
        # revenue = TODO determine the side , we should not use "or" here, one side is for the inverse order
        # check this by observe whether the revenue is monotonous increasing or not
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        # jax.debug.breakpoint()
        
        def truncate_agent_trades(agentTrades, remainQuant):
            quantities = agentTrades[:, 1]
            cumsum_quantities = jnp.cumsum(quantities)
            cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
            truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx, jnp.zeros_like(agentTrades[0]), agentTrades.at[:, 1].set(jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, jnp.where(jnp.arange(len(quantities)) == cut_idx, remainQuant - cumsum_quantities[cut_idx - 1], 0))))
            return jnp.where(remainQuant >= jnp.sum(quantities), agentTrades, jnp.where(remainQuant <= quantities[0], jnp.zeros_like(agentTrades).at[0, :].set(agentTrades[0]).at[0, 1].set(remainQuant), truncated_agentTrades))
        agentTrades = truncate_agent_trades(agentTrades, state.task_to_execute-state.quant_executed)
        new_execution = agentTrades[:,1].sum()
        revenue = (agentTrades[:,0]//self.tick_size * agentTrades[:,1]).sum()
        agentQuant = agentTrades[:,1].sum()
        vwap =(executed[:,0]//self.tick_size* executed[:,1]).sum()//(executed[:,1]).sum()
        advantage = revenue - vwap * agentQuant ### (weightedavgtradeprice-vwap)*agentQuant ### revenue = weightedavgtradeprice*agentQuant
        Lambda = self.Lambda 
        drift = agentQuant * (vwap - state.init_price//self.tick_size)
        rewardValue = advantage + Lambda * drift
        reward = jnp.sign(agentTrades[0,0]) * rewardValue # if no value agentTrades then the reward is set to be zero
        # ---------- used for slippage ----------
        agentPrice = revenue/agentQuant
        slippage = agentPrice - vwap
        price_drift = (vwap - state.init_price//self.tick_size)
        drift_reward =  Lambda * drift
        # ========== get reward and revenue END ==========
        
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        def bestPircesImpute(bestprices,lastBestPrice):
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
        bestasks, bestbids = bestPircesImpute(bestasks[-self.stepLines:],state.best_asks[-1,0]),bestPircesImpute(bestbids[-self.stepLines:],state.best_bids[-1,0])
        
        new_submitted = jnp.where((action_msgs[:, 2] < 0)[:, jnp.newaxis], action_msgs, 0)
        new_submitted_cost = (new_submitted[:,3]//self.tick_size*new_submitted[:,2]).sum()
        buy_side_quant = new_submitted[:,2].sum() # CAUTION it should be negative
        revenue += new_submitted_cost # CAUTION, the cost should be negative
        
        jax.debug.print("action {}, buy_side_quant {}",action_msgs[:,2],buy_side_quant)
        state = EnvState(asks,bids,trades,bestasks,bestbids,state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,\
            state.init_price,state.task_to_execute,state.quant_executed+new_execution+buy_side_quant,state.total_revenue+revenue,state.step_counter+1,\
            state.max_steps_in_episode)
        done = self.is_terminal(state,params)
        reward /= 10000
        # reward /= params.avg_twap_list[state.window_index]


        # reward = slippage*new_execution # TODO
        # reward = rewardValue # TODO
        # ---------- used for slippage ----------
        # abs_average_price = params.avg_twap_list[state.window_index]
        # reward = jnp.sign(agentTrades[0,0])*(revenue - abs_average_price*agentQuant)*10000/abs_average_price # TODO
        # '''total revenue: advatange of twap in bp'''
        # ---------- used for slippage ----------
        # jax.debug.print(">>> base {}, delta {}, action {}; truncated {};\n+ reward {};executed {};slippage {};\n+ sign {}; executed*slippage {}",get_base_action(state, params), delta,action_,action,reward,new_execution,slippage,jnp.sign(agentTrades[0,0]),new_execution*slippage)
        return self.get_obs(state,params),state,reward,done,\
            {"window_index":state.window_index,"total_revenue":state.total_revenue,\
            "quant_executed":state.quant_executed,"task_to_execute":state.task_to_execute,\
            "average_price":state.total_revenue/state.quant_executed,\
            "current_step":state.step_counter,\
            'done':done,'slippage':slippage,"price_drift":price_drift,\
            "drift_reward":drift_reward,"advantage_reward":advantage,\
            "current_executed":new_execution,
            }
        # TODO episodic slippage


    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        # all windows can be reached
        
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()) if self.window_index == -1 else jnp.array(self.window_index,dtype=jnp.int32)
        # idx_data_window = jnp.array(self.window_index,dtype=jnp.int32)
        # one window can be reached
        
        # jax.debug.print("window_size {}",self.max_steps_in_episode_arr[0])
        
        # task_size,content_size,array_size = self.task_size,self.max_steps_in_episode_arr[idx_data_window],self.max_steps_in_episode_arr.max().astype(jnp.int32) 
        # task_size,content_size,array_size = self.task_size,self.max_steps_in_episode_arr[idx_data_window],1000
        # base_allocation = task_size // content_size
        # remaining_tasks = task_size % content_size
        # array = jnp.full(array_size, 0, dtype=jnp.int32)
        # array = array.at[:remaining_tasks].set(base_allocation+1)
        # twap_quant_arr = array.at[remaining_tasks:content_size].set(base_allocation)
        
        def stateArray2state(stateArray):
            state0 = stateArray[:,0:6];state1 = stateArray[:,6:12];state2 = stateArray[:,12:18];state3 = stateArray[:,18:20];state4 = stateArray[:,20:22]
            state5 = stateArray[0:2,22:23].squeeze(axis=-1);state6 = stateArray[2:4,22:23].squeeze(axis=-1);state9= stateArray[4:5,22:23][0].squeeze(axis=-1)
            return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window])
            # return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window],0,twap_quant_arr)
        stateArray = params.stateArray_list[idx_data_window]
        state_ = stateArray2state(stateArray)
        # print(self.max_steps_in_episode_arr[idx_data_window])
        # jax.debug.breakpoint()
        obs_sell = params.obs_sell_list[idx_data_window]
        obs_buy = params.obs_buy_list[idx_data_window]
        state = EnvState(*state_)
        # jax.debug.print("state after reset {}", state)
        obs = obs_sell if self.task == "sell" else obs_buy
        
        return obs,state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (state.time-state.init_time)[0]>params.episode_time
    
    def getActionMsgs(self, action: Dict, state: EnvState, params: EnvParams):
        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides_=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        sides=jnp.sign(action)*sides_
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
        A = best_bid if self.task=='sell' else best_ask # aggressive would be at bids
        # M = (best_bid + best_ask)//2//self.tick_size*self.tick_size 
        P = best_ask if self.task=='sell' else best_bid
        PP= best_ask+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bid-self.tick_size*self.n_ticks_in_book
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
        ifMarketOrder = (remainingTime <= marketOrderTime)
        def market_order_logic(state: EnvState):
            # quant = state.task_to_execute - state.quant_executed
            quant =  -1*state.quant_executed
            # price = A + (-1 if self.task == 'sell' else 1) * (self.tick_size * 100) * 100
            #FIXME not very clean way to implement, but works:
            quants = jnp.asarray((quant - quant//2, quant//2),jnp.int32) 
            prices = jnp.asarray((A, A),jnp.int32)
            # (self.tick_size * 100) : one dollar
            # (self.tick_size * 100) * 100: choose your own number here(the second 100)
            return quants, prices
        def normal_order_logic(state: EnvState, action: jnp.ndarray):
            quants = action.astype(jnp.int32) # from action space
            prices = jnp.asarray((A, P), jnp.int32)
            return quants, prices
        market_quants, market_prices = market_order_logic(state)
        normal_quants, normal_prices = normal_order_logic(state, action)
        quants = jnp.where(ifMarketOrder, market_quants, normal_quants)
        prices = jnp.where(ifMarketOrder, market_prices, normal_prices)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================


    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # ========= self.get_obs(state,params) =============
        # -----------------------1--------------------------
        best_asks=state.best_asks[:,0]
        best_bids =state.best_bids[:,0]
        mid_prices=(best_asks+best_bids)//2//self.tick_size*self.tick_size 
        second_passives = best_asks+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bids-self.tick_size*self.n_ticks_in_book
        spreads = best_asks - best_bids
        # -----------------------2--------------------------
        timeOfDay = state.time
        deltaT = state.time - state.init_time
        # -----------------------3--------------------------
        initPrice = state.init_price
        priceDrift = mid_prices[-1] - state.init_price
        # -----------------------4--------------------------
        taskSize = state.task_to_execute
        executed_quant=state.quant_executed
        # -----------------------5--------------------------
        shallowImbalance = state.best_asks[:,1]- state.best_bids[:,1]
        # ========= self.get_obs(state,params) =============
        # jax.debug.breakpoint()
        # [item for item in map(type,[best_bids,best_asks,mid_prices,second_passives,spreads,timeOfDay,deltaT,shallowImbalance])]
        
        obs = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives,spreads,timeOfDay,deltaT,jnp.array([initPrice]),jnp.array([priceDrift]),\
            jnp.array([taskSize]),jnp.array([executed_quant]),shallowImbalance,jnp.array([state.step_counter]),jnp.array([state.max_steps_in_episode])))
        # jax.debug.breakpoint()
        def obsNorm(obs):
            return jnp.concatenate((
                obs[:400]/3.5e7, # best_bids,best_asks,mid_prices,second_passives  TODO CHANGE THIS
                obs[400:500]/100000, # spreads
                obs[500:501]/100000, # timeOfDay
                obs[501:502]/1000000000, # timeOfDay
                obs[502:503]/10,# deltaT
                obs[503:504]/1000000000,# deltaT
                obs[504:505]/3.5e7,# initPrice  TODO CHANGE THIS
                obs[505:506]/100000,# priceDrift
                obs[506:507]/500, # taskSize TODO CHANGE THIS
                obs[507:508]/500, # executed_quant TODO CHANGE THIS
                obs[508:608]/100, # shallowImbalance 
                obs[608:609]/300, # step_counter TODO CHANGE THIS
                obs[609:610]/300, # max_steps_in_episode TODO CHANGE THIS
            ))
        obsNorm_=obsNorm(obs)
        # jax.debug.breakpoint()
        return obsNorm_


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        # return spaces.Box(-1,1,(self.n_actions,),dtype=jnp.int32)
        # return spaces.Box(-2,2,(self.n_actions,),dtype=jnp.int32)
        # return spaces.Box(-5,5,(self.n_actions,),dtype=jnp.int32)
        return spaces.Box(-100,100,(self.n_actions,),dtype=jnp.int32)
    
    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        space = spaces.Box(-10,10,(610,),dtype=jnp.float32) 
        return space

    #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

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
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell",1)
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
    
    print("Defualt side: sell(+); buy(-)")
   
    '''
    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ =  jax.random.split(key_policy, 2)
        test_action=env.action_space().sample(key_policy)
        # test_action=env.action_space().sample(key_policy) * random.randint(1, 10) # CAUTION not real action
        # test_action=env.action_space().sample(key_policy)//10 # CAUTION not real action
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        # for key, value in info.items():
        #     print(key, value)
        print("quant_executed",info['quant_executed'])
        print("current_step",info['current_step'])
        print("current_executed",info['current_executed'])
        print("total_revenue",info['total_revenue'])
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================
    '''
        
        
        

    # print(env_params.message_data.shape, env_params.book_data.shape)
    testAction=jnp.array([[ 0, 0],
                          [10,10],
                          [-10,-10],
                          [10,10],
                          [-10,-10],
                          [10,10],
                          [-10,-10],
                          [10,10],
                          [-10,-10],
                          [10,10],
                          [-10,-10],
                          [10,10],
                          [-10,-10],
                          [10,10],
                          [-10,-10],
                          ])
    for i in range(1,10):
    # for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ =  jax.random.split(key_policy, 2)
        test_action=testAction[i,:]
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        # for key, value in info.items():
        #     print(key, value)
        print("quant_executed",info['quant_executed'])
        print("current_step",info['current_step'])
        print("current_executed",info['current_executed'])
        print("total_revenue",info['total_revenue'])
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        if done:
            print("==="*20)
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================
        
        

    # # ####### Testing the vmap abilities ########
    
    enable_vmap=False
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
