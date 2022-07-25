# =============================================================================
import random
import numpy as np
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod
# ----------------------------------------------------------------------------
from gym import Env
from gym import spaces
# ----------------------------------------------------------------------------
from gym_trading.utils import * 
from gym_trading.envs.match_engine import Core
from gym_trading.envs.match_engine import Broker, Utils
# =============================================================================



class BaseEnv(Env, ABC):
    num_steps = 1024 # size of a flow
    high = 1024
    max_quantity = 6000
    max_price = 31620700
    min_price = 31120200
    num2liuquidate = 300
    cost_parameter = 0.01

    def __init__(self, Flow) -> None:
        super().__init__()
        self.Flow = Flow
        self.core = None
        self.price_list = None
        self.action_space = spaces.Box(0, BaseEnv.high,shape =(1,),dtype = np.float32)
        self.observation_space = spaces.Dict({
            'price':spaces.MultiDiscrete([BaseEnv.max_price for _ in range(10)], dtype=np.int64),
            'quantity':spaces.MultiDiscrete([BaseEnv.max_quantity for _ in range(10)], dtype=np.int64)
            })
        self.done = False
        self.running_reward = 0
        self.init_reward = 0
        self.max_order_size = 0
        self.info = {}
        
    def setp(self, action: float = 0):
        # return observation, reward, done, info
        observation = self._get_obs(acion)
        reward = self._get_reward(acion)
        done = self._get_done(acion)
        info = self._get_info(acion)
        return  observation, reward, done, info
    
    def reset(self):
        '''return the observation of the initial condition'''
        index_random = random.randint(0, self.Flow.shape[0]-BaseEnv.num_steps-1)
        flow = self.Flow.iloc[index_random:index_random+BaseEnv.num_steps,:]
        self.core = Core(flow)
        stream =  flow.iloc[0,:]
        self._set_init_reward(stream)
        init_price = np.array(get_price_from_stream(stream)).astype(np.int64)
        init_quant = np.array(get_quantity_from_stream(stream)).astype(np.int64)
        self.running_reward += self.calculate_reward()
        self.reset_states()
        self.max_order_size = sum(init_obs)
        init_obs= {
            'price' : init_price,
            'quantity' : init_quant
            }
        return init_obs
    def reset_states(self):
        self.running_reward = 0
        self.done = False


    def _get_obs(self):
        pass
    def _get_reward(self,acion):
        if not self.done:
            return 0
        else:
            return self.running_reward
    def _get_done(self,acion):
        return self.done
    def _get_info(self,acion):
        return self.info
    def calculate_reward(self):
        return 1 # TODO
    def _set_init_reward(self, stream):
        num = BaseEnv.num2liuquidate
        obs = Utils.from_series2pair(stream)
        level = Broker._level_market_order_liquidating(num, obs)
        reward,consumed = 0,0
        for i in range(level-1):
            reward += obs[i][0] * obs[i][1]
            consumed += obs[i][1]
        if level == 0:
            self.init_reward = 0 
        elif level == -999:
            consumed += obs[level-1][1]
            reward += obs[level-1][0] * obs[level-1][1]
            inventory_cost=(num-consumed)*(num-consumed)*BaseEnv.cost_parameter
            self.init_reward = reward - inventory_cost
        else:
            reward += obs[level-1][0] * (num - consumed)
            self.init_reward = reward
    
if __name__=="__main__":
    from gym_trading.data.data_pipeline import ExternalData
    Flow = ExternalData.get_sample_order_book_data()
    env = BaseEnv(Flow)
    obs = env.reset()