"""
Base Environment 

University of Oxford
Corresponding Author: 
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
Kang Li     (kang.li@keble.ox.ac.uk)
Peer Nagy   (peer.nagy@reuben.ox.ac.uk)
V1.0

Module Description
This module offers an advanced simulation environment for limit order books 
 using JAX for high-performance computations. It is designed for reinforcement
 learning applications in financial markets.

Key Components
EnvState:   Dataclass to manage the state of the environment, 
            including order book states, trade records, and timing information.
EnvParams:  Configuration class for environment parameters, 
            including message data, book data, and episode timing.
BaseLOBEnv: Main environment class inheriting from Gymnax's base environment, 
            providing methods for environment initialization, 
            stepping through time steps, and resetting the environment. 

Functionality Overview
__init__:           Initializes the environment. Sets up paths for data, 
                    time windows, order book depth, and other parameters. 
                    It also loads and preprocesses the data from LOBSTER.
default_params:     Returns the default environment parameters, 
                    including the preprocessed message and book data.
step_env:           Advances the environment by one step. It processes both the
                    action messages and data messages through the order book, 
                    updates the state, and determines the reward 
                    and termination condition.
reset_env:          Resets the environment to an initial state. 
                    It selects a new data window, initializes the order book, 
                    and sets the initial state.
is_terminal:        Checks whether the current state is terminal, 
                    based on the elapsed time since the episode's start.
get_obs:            Returns the current observation from environment's state.
name:               Provides the name of the environment.
num_actions:        Returns the number of possible actions in the environment.
action_space:       Defines the action space of the environment, including 
                    sides, quantities, and prices of actions.
observation_space:  (Not implemented) Intended to define 
                    the observation space of the environment.
state_space:        Defines the state space of the environment, 
                    including bids, asks, trades, and time.
_get_initial_time:  Retrieves the initial time of a data window.
_get_data_messages: Fetches an array of messages for a given step 
                    within a data window.
"""

# from jax import config
# config.update("jax_enable_x64",True)
import sys
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
import itertools
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER
from gymnax_exchange.utils import *
import pickle


@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index:int
    step_counter: int
    max_steps_in_episode: int

@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    #episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    time_delay_obs_act: chex.Array 
    init_base_state_array: chex.Array


class BaseLOBEnv(environment.Environment):
    """The basic RL environment for the limit order book (LOB) using
    JAX-LOB functions for manipulating the orderbook.

    Inherits from gymnax base environment. 

    ...
    Attributes
    ----------
    sliceTimeWindow : int
        first name of the person
    stepLines : int
        family name of the person
    messagePath : int
        age of the person

        ... #TODO Complete the class docstring once refactored. 

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def _get_state_from_data(self,message_data,book_data,max_steps_in_episode,window_index)->EnvState:
        time=jnp.array(message_data[0,0,-2:])
        #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
        def get_initial_orders(book_data,time):
            orderbookLevels=10
            initid=job.INITID
            data=jnp.array(book_data).reshape(int(10*2),2)
            newarr = jnp.zeros((int(orderbookLevels*2),8),dtype=jnp.int32)
            initOB = newarr \
                .at[:,3].set(data[:,0]) \
                .at[:,2].set(data[:,1]) \
                .at[:,0].set(1) \
                .at[0:orderbookLevels*4:2,1].set(-1) \
                .at[1:orderbookLevels*4:2,1].set(1) \
                .at[:,4].set(initid) \
                .at[:,5].set(initid-jnp.arange(0,orderbookLevels*2)) \
                .at[:,6].set(time[0]) \
                .at[:,7].set(time[1])
            return initOB
        init_orders=get_initial_orders(book_data,time)
        #Initialise both sides of the book as being empty
        asks_raw=job.init_orderside(self.nOrdersPerSide)
        bids_raw=job.init_orderside(self.nOrdersPerSide)
        trades_init=(jnp.ones((self.nTradesLogged,8))*-1).astype(jnp.int32)
        #Process the initial messages through the orderbook
        ordersides=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw,trades_init))
        return EnvState(ask_raw_orders=ordersides[0],
                        bid_raw_orders=ordersides[1],
                        trades=ordersides[2],
                        init_time=time,
                        time=time,
                        customIDcounter=0,
                        window_index=window_index,
                        step_counter=0,
                        max_steps_in_episode=max_steps_in_episode)
    
    def _init_states(self,alphatradePath):
        print("START:  pre-reset in the initialization")
        pkl_file_name = (alphatradePath
                         + 'stateArray_idx_'+ str(self.window_index)
                         +'_dtype_"'+self.data_type
                         +'"_depth_'+str(self.book_depth)
                         +'.pkl')
        print("pre-reset will be saved to ",pkl_file_name)
        try:
            with open(pkl_file_name, 'rb') as f:
                self.init_base_states_array = pickle.load(f)
            print("LOAD FROM PKL")
        except:
            print("DO COMPUTATION")
            states = [self._get_state_from_data(self.messages[i],
                                                self.books[i],
                                                self.max_steps_in_episode_arr[i],
                                                i) 
                        for i in range(self.n_windows)]
            self.init_base_states_array=tree_stack(states)
            with open(pkl_file_name, 'wb') as f:
                pickle.dump(self.init_base_states_array, f) 
        print("FINISH: pre-reset in the initialization")


    def __init__(self, alphatradePath,window_index,data_type="fixed_time"):
        super().__init__()
        self.window_index = window_index
        self.data_type = data_type # fixed_steps, fixed_time
        self.sliceTimeWindow = 1800 # counted by seconds, 1800s=0.5h
        self.stepLines = 100
        self.messagePath = alphatradePath+"/data/Flow_10/"
        self.orderbookPath = alphatradePath+"/data/Book_10/"
        self.start_time = 34200  # 09:30
        self.end_time = 57600  # 16:00
        self.nOrdersPerSide=100
        self.nTradesLogged=100
        self.book_depth=10
        self.n_actions=3
        self.customIDCounter=0
        self.trader_unique_id=job.INITID+1
        self.tick_size=100
        self.tradeVolumePercentage = 0.01
        self.data_type = data_type
        loader=LoadLOBSTER(alphatradePath,10,"fixed_time",self.sliceTimeWindow,self.stepLines)
        msgs,books,window_lengths,n_windows=loader.run_loading()
        self.max_steps_in_episode_arr = window_lengths 
        self.messages=msgs
        self.books=books
        self.n_windows = n_windows

        self._init_states(alphatradePath)


    
    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(self.messages, self.books,jnp.array([0, 0]),self.init_base_states_array)


    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        data_messages=self._get_data_messages(params.message_data,state.window_index,state.step_counter)

        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides=((action["sides"]+1)/2).astype(jnp.int32)      #from action space
        prices=action["prices"]     #from action space
        quants=action["quantities"] #from action space
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)

        #Add to the top of the data messages 
        total_messages=jnp.concatenate([action_msgs,data_messages],axis=0)
        #jax.debug.print("Step messages to process are: \n {}", total_messages)

        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]

        #Process messages of step (action+data) through the orderbook
        ordersides=job.scan_through_entire_array(total_messages,(state.ask_raw_orders,state.bid_raw_orders,state.trades))

        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter)
        state = EnvState(ordersides[0],ordersides[1],ordersides[2],state.init_time,time,state.customIDcounter+self.n_actions,\
            state.window_index,state.step_counter+1,state.max_steps_in_episode)
        done = self.is_terminal(state,params)
        reward=0
        #jax.debug.print("Final state after step: \n {}", state)
        return self.get_obs(state,params),state,reward,done,{"info":0}



    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        idx_data_window = jnp.where(
            self.window_index == -1,
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()),  
            jnp.array(self.window_index, dtype=jnp.int32))

        #Dummy state for all but window index.
        base_state = index_tree(params.init_base_state_array,
                           idx_data_window)
        
        obs=self._get_obs(base_state,params=params)
        return obs,base_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (state.time-state.init_time)[0]>params.episode_time

    def _get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return dummy observation."""
        return 0

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeBase-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Dict(
            {
                "sides":spaces.Box(0,2,(self.n_actions,),dtype=jnp.int32),
                "quantities":spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32),
                "prices":spaces.Box(0,99999999,(self.n_actions,),dtype=jnp.int32)
            }
        )

    #TODO: define obs space (4xnDepth) array of quants&prices. Not that important right now. 
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return NotImplementedError

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment. #FIXME Samples absolute
          nonsense, don't use.
        """
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,999999999,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,999999999,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,999999999,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
    
    def _get_initial_time(self,messageData,idx_window):
        """Obtain the arrival time of the first message in a given
        data_window. Data window: pre-arranged 
            Parameters:
                    messageData (Array): 4D array with dimensions: windows,
                                            steps, messages, features. 
                    idx_window (int): Index of the window to consider.
            Returns:
                    Time (Array): Timestamp of first message [s, ns]
        """
        return messageData[idx_window,0,0,-2:]

    def _get_data_messages(self,messageData,idx_window,step_counter):
        """Returns an array of messages for a given step. 
            Parameters:
                    messageData (Array): 4D array with dimensions: windows,
                                            steps, messages, features. 
                    idx_window (int): Index of the window to consider.
                    step_counter (int): desired step to consider. 
            Returns:
                    Time (Array): Timestamp of first message [s, ns]
        """
        messages=messageData[idx_window,step_counter,:,:]
        return messages
    
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
        "ACTION_TYPE": "pure", # "pure",
        "REWARD_LAMBDA": 1.0,
        "DTAT_TYPE":"fixed_time",
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    print(config["ATFOLDER"])
    env= BaseLOBEnv(config["ATFOLDER"],config["WINDOW_INDEX"],config["DTAT_TYPE"])
    env_params=env.default_params

    obs,state=env.reset(key_reset,env_params)

    print(state)
    print(obs)
    print(env.messages.shape)