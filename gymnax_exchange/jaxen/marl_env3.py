from flax import struct
import chex
from typing import Tuple, Dict, Optional
from gymnax.environments import environment, spaces
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import dataclasses



@struct.dataclass
class MARLEnvState:
    mm_states: Dict[int, MarketMakingEnv.EnvState]
    exec_states: Dict[int, ExecutionEnv.EnvState]
    num_mm_agents: int
    num_exec_agents: int



@struct.dataclass
class MARLEnvParams:
    mm_params: Dict[int, MarketMakingEnv.EnvParams]
    exec_params: Dict[int, ExecutionEnv.EnvParams]


class MARLEnv(BaseLOBEnv):
    def __init__(self, alphatradePath, window_selector, sliceTimeWindow, ep_type="fixed_time"):
        super().__init__()
        self.mm_envs = {}
        self.exec_envs = {}
        self.num_mm_agents = 0
        self.num_exec_agents = 0

    def add_mm_agent(self, agent_id, task, action_type, max_task_size=500, rewardLambda=0.0001):
        self.mm_envs[agent_id] = MarketMakingEnv(
            alphatradePath=self.base_env.alphatradePath,
            task=task,
            window_index=self.base_env.window_selector,
            action_type=action_type,
            episode_time=self.base_env.sliceTimeWindow,
            max_task_size=max_task_size,
            rewardLambda=rewardLambda,
            ep_type=self.base_env.ep_type
        )
        self.num_mm_agents += 1

    def add_exec_agent(self, agent_id, task, action_type, max_task_size=500, rewardLambda=1.0):
        self.exec_envs[agent_id] = ExecutionEnv(
            alphatradePath=self.base_env.alphatradePath,
            task=task,
            window_index=self.base_env.window_selector,
            action_type=action_type,
            episode_time=self.base_env.sliceTimeWindow,
            max_task_size=max_task_size,
            rewardLambda=rewardLambda,
            ep_type=self.base_env.ep_type
        )
        self.num_exec_agents += 1



    def is_terminal(self, state: MARLEnvState, params: MARLEnvParams) -> bool:
        base_done = self.base_env.is_terminal(state.base_state, params.base_params)
        mm_done = any(
            env.is_terminal(state.mm_states[agent_id], params.mm_params[agent_id])
            for agent_id, env in self.mm_envs.items()
        )
        exec_done = any(
            env.is_terminal(state.exec_states[agent_id], params.exec_params[agent_id])
            for agent_id, env in self.exec_envs.items()
        )
        return base_done or mm_done or exec_done

    @jit
    def step_env(self, key: chex.PRNGKey, state: MARLEnvState, actions: Dict[int, jax.Array], params: MARLEnvParams):
        key, subkey = jax.random.split(key)

        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )

        mm_action_msgs = [
            self.mm_envs[agent_id]._getActionMsgsV2(actions[agent_id], state.mm_states[agent_id], params.mm_params[agent_id])
            for agent_id in self.mm_envs.keys()
        ]
        exec_action_msgs = [
            self.exec_envs[agent_id]._getActionMsgs(actions[agent_id], state.exec_states[agent_id], params.exec_params[agent_id])
            for agent_id in self.exec_envs.keys()
        ]
        
        #Need to do cancel messages
        cnl_msg_bid = job.getCancelMsgs(state.bid_raw_orders,  self.num_exec_agents, self.num_mm_agents, 1)
        cnl_msg_ask = job.getCancelMsgs(state.ask_raw_orders, self.num_mm_agents + self.num_exec_agents, self.num_mm_agents, -1)
        
        all_action_msgs = jnp.concatenate(mm_action_msgs + exec_action_msgs, axis=0)
        all_cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)
        
        trades_reinit = jnp.full((self.base_env.nTradesLogged, 8), -1, dtype=jnp.int32)
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            jnp.concatenate([all_cnl_msgs, all_action_msgs,data_messages], axis=0),
            (state.bid_raw_orders, state.ask_raw_orders, trades_reinit),
            self.base_env.stepLines
        )
        #Update base state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter)
       
        new_base_state = dataclasses.replace( asks, bids,trades,state.init_time,time, state.base_state.step_counter + 1,state.max_steps_in_episode,state.start_index)
       
        
        rewards = {agent_id: self.mm_envs[agent_id]._get_reward(new_mm_states[agent_id], params.mm_params[agent_id], trades, bestasks, bestbids) for agent_id in self.mm_envs.keys()}
        rewards.update({agent_id: self.exec_envs[agent_id]._get_reward(new_exec_states[agent_id], params.exec_params[agent_id], trades) for agent_id in self.exec_envs.keys()})
        
        new_mm_states = {
            agent_id: self.mm_envs[agent_id]._update_state(
                state.mm_states[agent_id], asks, bids, trades, bestasks, bestbids, params.mm_params[agent_id]
            ) for agent_id in self.mm_envs.keys()
        }
        
        new_exec_states = {
            agent_id: self.exec_envs[agent_id]._update_state(
                state.exec_states[agent_id], asks, bids, trades, bestasks, bestbids, params.exec_params[agent_id]
            ) for agent_id in self.exec_envs.keys()
        }

        infos = {agent_id: self.mm_envs[agent_id]._get_info(new_mm_states[agent_id]) for agent_id in self.mm_envs.keys()}
        infos.update({agent_id: self.exec_envs[agent_id]._get_info(new_exec_states[agent_id]) for agent_id in self.exec_envs.keys()})
        
        done = self.is_terminal(new_base_state, params)

        new_state = MARLEnvState(new_base_state, new_mm_states, new_exec_states, self.num_mm_agents, self.num_exec_agents)
        obs = {agent_id: self.mm_envs[agent_id]._get_obs(new_mm_states[agent_id], params.mm_params[agent_id]) for agent_id in self.mm_envs.keys()}
        obs.update({agent_id: self.exec_envs[agent_id]._get_obs(new_exec_states[agent_id], params.exec_params[agent_id]) for agent_id in self.exec_envs.keys()})
        
        return obs, new_state, rewards, done, infos

    def reset_env(self, key: chex.PRNGKey, params: MARLEnvParams):
        key, subkey = jax.random.split(key)
        base_obs, base_state = self.reset_env(subkey, params.base_params)
        obs = {}
        
        mm_states = {agent_id: base_state for agent_id in self.mm_envs.keys()}
        exec_states = {agent_id: base_state for agent_id in self.exec_envs.keys()}
        
        for agent_id in self.mm_envs.keys():
            obs[agent_id] = self.mm_envs[agent_id]._get_obs(mm_states[agent_id], params.mm_params[agent_id])
        for agent_id in self.exec_envs.keys():
            obs[agent_id] = self.exec_envs[agent_id]._get_obs(exec_states[agent_id], params.exec_params[agent_id])
        
        marl_state = MARLEnvState(base_state, mm_states, exec_states,)#no of each state... , )
        #Need to make obs a dict with this and base obs
        return obs, marl_state


    def action_space(self, params: Optional[MARLEnvParams] = None) -> Dict[int, spaces.Box]:
        action_spaces = {}
        for agent_id, env in self.mm_envs.items():
            action_spaces[agent_id] = env.action_space(params.mm_params[agent_id])
        for agent_id, env in self.exec_envs.items():
            action_spaces[agent_id] = env.action_space(params.exec_params[agent_id])
        return action_spaces

    def observation_space(self, params: MARLEnvParams) -> Dict[int, spaces.Box]:
        obs_spaces = {}
        for agent_id, env in self.mm_envs.items():
            obs_spaces[agent_id] = env.observation_space(params.mm_params[agent_id])
        for agent_id, env in self.exec_envs.items():
            obs_spaces[agent_id] = env.observation_space(params.exec_params[agent_id])
        return obs_spaces

    def state_space(self, params: MARLEnvParams) -> spaces.Dict:
        return NotImplementedError
    

