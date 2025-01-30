"""
Base Environment with variable start time for episodes. 

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
__init__:           Sets up initial values and paths. Loads data from 
                    LOBSTER and pre-calculates all initial states for 
                    reset.
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
from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER_resample
from gymnax_exchange.jaxlobster.gen_loader import GenLoader
from gymnax_exchange.utils.utils import *
import pickle
from jax.experimental import checkify


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
    start_index: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    episode_time: int
    time_delay_obs_act: chex.Array
    window_selector : int
    init_states_array: chex.Array


class BaseLOBEnv(environment.Environment):

    """The basic RL environment for the limit order book (LOB) using
    JAX-LOB functions for manipulating the orderbook. 
    Inherits from gymnax base environment. 
    Attributes
    ----------
    window_selector : int
        -1 to randomly choose start times from all available.
        int in range(0,n_starts) to choose a specific window for debug. 
    data_type : str
        "fixed_steps" and "fixed_time" to defn episode end crit.
    sliceTimeWindow : int
        Length of episode in steps or seconds based on above.
    stepLines : int
        number of messages to process per step. 
    day_start : int
        Beginning time of day in seconds
    day_end : int
        End time of day in seconds
    nOrdersPerSide : int
        Maximum capacity of orders for JAXLOB
    nTradesLogged : int
        Maximum number of trades logged (in a step)
    book_depth : int
        Depth considered for LOBSTER data retrieval
    n_actions : int
        Number of actions. Dimension of act space
    n_ticks_in_book : int
        Depth of passive order in act space in ticks.
    customIDCounter : int
        Ensures unique IDs for orders submitted by agent.
    trader_unique_id : int
        Offset of unique ID that can be used.
    tick_size : int
        Price tick size. Lobster counts in hundreths of cents.
    start_resolution: int
        Interval, in seconds, at which episodes may start.
    loader : LoadLOBSTER 
        Object that deals with data-loading.
    max_messages_in_episode_arr : jnp.Array 
        Total messages for each possible window.
    messages : jnp.Array  
        Loaded message data.
    books : jnp.Array  
        Loaded book data for start-points
    n_windows : int 
        Number of start points
    start_indeces : jnp.Array  
        Ineces for start points
    end_indeces : jnp.Array  
        Indeces for ep end for each start-point. 
    init_states_array : jnp.Array  
        Initial state for each start point: for reset func. 

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, alphatradePath, window_selector, sliceTimeWindow, ep_type="fixed_time"):
        super().__init__()
        self.window_selector = window_selector
        self.ep_type = ep_type # fixed_steps, fixed_time
        self.sliceTimeWindow = sliceTimeWindow # counted by seconds, 1800s=0.5h
        self.stepLines = 100
        self.day_start = 34200  # 09:30
        self.day_end = 57600  # 16:00
        self.nOrdersPerSide=100
        self.nTradesLogged=100
        self.book_depth=10
        self.n_actions=2
        self.n_ticks_in_book = 2 # Depth of PP actions
        self.customIDCounter=0
        self.trader_unique_id=10
        self.tick_size=100
        self.start_resolution=60 #Interval in seconds at which eps start
        loader=LoadLOBSTER_resample(alphatradePath,
                                    self.book_depth,
                                    ep_type,
                                    window_length=self.sliceTimeWindow,
                                    n_msg_per_step=self.stepLines,
                                    window_resolution=self.start_resolution) 
        msgs,starts,ends,books,max_messages_arr=loader.run_loading()
        self.max_messages_in_episode_arr = max_messages_arr
        self.messages=msgs #Is different to trad. base: all msgs concat. 
        self.books=books
        self.n_windows = starts.shape[0]
        self.start_indeces=starts
        self.end_indeces=ends
        self._init_states(alphatradePath,self.start_indeces)
    
    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            message_data=self.messages,
            book_data=self.books,
            episode_time=self.sliceTimeWindow,
            time_delay_obs_act=jnp.array([0, 0]),
            window_selector=self.window_selector,
            init_states_array=self.init_states_array
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        #data_messages=#self._get_data_messages(params.message_data,
                       #                       state.start_index,
                        #                      state.step_counter,
                         #                     state.init_time[0]+params.episode_time)
        data_messages=self._get_generative_messages(params.message_data,n_messages)
        
        #Note: Action of the base environment should consistently be "DO NOTHING"

        total_messages=data_messages

        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]
        #Process messages of step (action+data) through the orderbook
        
        ordersides=job.scan_through_entire_array(total_messages,(state.ask_raw_orders,state.bid_raw_orders,state.trades))

        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter)
        state = EnvState(ordersides[0],ordersides[1],ordersides[2],state.init_time,time,state.customIDcounter+self.n_actions,\
            state.window_index,state.step_counter+1,state.max_steps_in_episode,state.start_index)
        done = self.is_terminal(state,params)
        reward=0
        #jax.debug.print("Final state after step: \n {}", state)
        return self._get_obs(state,params),state,reward,done,{"info":0}

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        idx_data_window = jnp.where(
            params.window_selector == -1,
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()),  
            jnp.array(params.window_selector, dtype=jnp.int32))
        first_state = index_tree(params.init_states_array, idx_data_window)
        obs = self._get_obs(first_state, params=params)
        return obs, first_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        #jax.debug.print("Time: {} , Init time: {}, Difference: {}",state.time, state.init_time,(state.time-state.init_time)[0])
        return (state.time-state.init_time)[0]>=params.episode_time

    def _get_state_from_data(self,first_message,book_data,max_steps_in_episode,window_index,start_index)->EnvState:
        time=jnp.array(first_message[-2:])
        #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
        def get_initial_orders(book_data,time):
            orderbookLevels=10
            #initid=25
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
        #jax.debug.print("init_orders {}",init_orders)
        #Initialise both sides of the book as being empty
        asks_raw=job.init_orderside(self.nOrdersPerSide)
        bids_raw=job.init_orderside(self.nOrdersPerSide)
        trades_init=(jnp.ones((self.nTradesLogged,8))*-1).astype(jnp.int32)
        #Process the initial messages through the orderbook
        ordersides=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw,trades_init))
        #jax.debug.print("trades init {}",ordersides[2])
        return EnvState(ask_raw_orders=ordersides[0],
                        bid_raw_orders=ordersides[1],
                        trades=ordersides[2],
                        init_time=jnp.array([(window_index*self.start_resolution)
                                                        %(self.day_end-self.day_start-self.sliceTimeWindow+self.start_resolution)
                                                        +self.day_start,0])
                                    if self.ep_type=="fixed_time" else time,
                        time=time,
                        customIDcounter=0,
                        window_index=window_index,
                        step_counter=0,
                        max_steps_in_episode=max_steps_in_episode,
                        start_index=start_index)

    def _init_states(self,alphatradePath,starts):
        print("START:  pre-reset in the initialization")
        pkl_file_name = (alphatradePath
                         + '_' + type(self).__name__
                         + '_stateArray_idx_' + str(self.window_selector)
                         + '_dtype_"' + self.ep_type
                         + '"_depth_' + str(self.book_depth)
                         + '.pkl')
        print("pre-reset will be saved to ", pkl_file_name)
        try:
            with open(pkl_file_name, 'rb') as f:
                self.init_states_array = pickle.load(f)
                print("LOADING STATES FROM PKL...")
        except:
            print("COMPUTING INIT STATES...")
            states = [self._get_state_from_data(self.messages[starts[i]],
                                                self.books[i],
                                                self.max_messages_in_episode_arr[i]
                                                    //self.stepLines+1,
                                                    i,
                                                    starts[i]) 
                        for i in range(self.n_windows)]
            
            self.init_states_array=tree_stack(states)
            print("SAVING STATES TO PKL...")
            with open(pkl_file_name, 'wb') as f:
                pickle.dump(self.init_states_array, f)
        print("DONE: pre-reset in the initialization")

    def _get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return dummy observation."""
        return 0
    
    def _get_data_messages(self,messageData,start,step_counter,end_time_s):
        """Returns an array of messages for a given step. 
            Parameters:
                    messageData (Array): 2D array of all msgs with
                                        dimensions: messages, features.
                    start (int): Index of first message to in episode
                    step_counter (int): desired step to consider
                    end_time_s (int): End time of ep in seconds
            Returns:
                    Messages (Array): 2D array of messages for step 
        """
        index_offset=start+self.stepLines*step_counter
        
        messages=jax.lax.dynamic_slice_in_dim(messageData,index_offset,self.stepLines,axis=0)
        #jax.debug.print("{}",messages)
        #jax.debug.print("End time: {}",end_time_s)
        #messages=messageData[index_offset:(index_offset+self.stepLines),:]
        #Replace messages after the cutoff time with padded 0s (except time)
        #jax.debug.print("m_wout_time {}",jnp.transpose(jnp.resize(messages[:,-2]>=end_time_s,messages[:,:-2].shape[::-1])))
        m_wout_time=jnp.where(jnp.transpose(jnp.resize(
                                messages[:,-2]>=end_time_s,
                                messages[:,:-2].shape[::-1])),
                              jnp.zeros_like(messages[:,:-2]),
                              messages[:,:-2])
        #jax.debug.print("m_wout_time {}",m_wout_time)

        messages=jnp.concatenate((m_wout_time,messages[:,-2:]),axis=1,dtype=jnp.int32)
        return messages
    def _get_generative_messages(self,previous_messages,n_messages):
        """Draft Generative Loader:
        Inputs:
        
        Outputs:
        
        
        """
        loader = GenLoader(
        model=GenLoader.dummy_model,
        initial_context=previous_messages,
        initial_ob_state=jnp.zeros((2, 10)),
        n_messages=100
        )
        messages, _ = loader.generate_step()
        

        return messages
    
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
                "trades": spaces.Box(-1,999999999,shape=(8,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
    


    
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

    env= BaseLOBEnv(alphatradePath=config["ATFOLDER"],
                    window_selector=config["WINDOW_INDEX"],
                    ep_type=config["DTAT_TYPE"])
    env_params=env.default_params

    obs,state=env.reset(key_reset,env_params)
    done=False

    while not done :
        obs,state,rewards,done,info=env.step_env(key_step,state,{},env_params)
        print(done)



    print(state)
    print(obs)
