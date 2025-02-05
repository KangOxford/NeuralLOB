# ============== testing scripts ===============
# from jax import config
# config.update("jax_enable_x64",True)
import jax
import jax.numpy as jnp
import sys
from re import L
import time 
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
import chex
import faulthandler; faulthandler.enable()
chex.assert_gpu_available(backend=None)
from jax import config # Code snippet to disable all jitting.
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
from AlphaTrade.gymnax_exchange.jaxen.old.exec_env_old import *
import json
# ============== testing scripts ===============

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = "/homes/80/kang/AlphaTrade/testing_small"
        # ATFolder = '/homes/80/kang/AlphaTrade'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    # print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)

    i = 1
    for i in range(1,10000):
        # ==================== ACTION ====================
        # ---------- acion from given strategy  ----------
        print("---"*20)
        print("window_index ",state.window_index)
        key_policy, _ = jax.random.split(key_policy,2)
        test_action=env.action_space().sample(key_policy)
        
        # env_state, last_obs, last_done, rng = runner_state
        # rng, _rng = jax.random.split(rng)
        # rng_action=jax.random.split(_rng, config["NUM_ENVS"])
        # action = jax.vmap(env.action_space().sample, in_axes=(0))(rng_action)
        # rng, _rng = jax.random.split(rng)
        # rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        # obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
        #     env.step, in_axes=(0, 0, 0, None)
        # )(rng_step, env_state, action, env_params)
        # if i == 277:
        #     jax.debug.breakpoint()
        def twap(state, env_params):
            # ---------- ifMarketOrder ----------
            remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
            marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
            ifMarketOrder = (remainingTime <= marketOrderTime)
            print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
            # ---------- ifMarketOrder ----------
            # ---------- quants ----------
            remainedQuant = state.task_to_execute - state.quant_executed
            remainedStep = state.max_steps_in_episode - state.step_counter
            stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
            # quants = [stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4, stepQuant//4]
            # quants = jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4, stepQuant//4]), independent=True)
            # limit_quants_agressive = jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4,stepQuant//4]), independent=True)
            # limit_quants_agressive = jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4, stepQuant//4]), independent=True).at[-1].set(max(stepQuant - 3*stepQuant//4,stepQuant//4))
            limit_quants_agressive = jnp.append(jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4]), independent=True),max(stepQuant - 3*stepQuant//4,stepQuant//4))
            # limit_quants_passive = jax.random.permutation(key_policy, jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4,remainedQuant//4]), independent=True)
            limit_quants_passive = jnp.append(jax.random.permutation(key_policy, jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4]), independent=True),remainedQuant//4)
            limit_quants = jnp.where(limit_quants_agressive.sum() <= remainedQuant,limit_quants_agressive,limit_quants_passive)
            # assert limit_quants.sum() <= remainedQuant, f"{limit_quants}, {limit_quants.sum()} should less_equal than {remainedQuant}"
            # if not limit_quants_agressive.sum() <= remainedQuant: print("+++")
            market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
            quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
            # ---------- quants ----------
            return jnp.array(quants)
            
        twap_action = twap(state, env_params)
        
        
        # print(f"Sampled {i}th actions are: ",test_action)
        print(f"Sampled {i}th actions are: ",twap_action)
        start=time.time()
        # obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        obs,state,reward,done,info=env.step(key_step, state,twap_action, env_params)
        print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        print(f"Time for {i} step: \n",time.time()-start)
        print("excuted ",info["quant_executed"])
        if done:
            break
    print(info["total_revenue"])
    print(info["total_revenue"]/info["task_to_execute"])

    # ####### Testing the vmap abilities ########
    
    # enable_vmap=True
    # if enable_vmap:
    #     vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    #     vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    #     vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

    #     num_envs = 10
    #     vmap_keys = jax.random.split(rng, num_envs)

    #     test_actions=vmap_act_sample(vmap_keys)
    #     print(test_actions)

    #     start=time.time()
    #     obs, state = vmap_reset(vmap_keys, env_params)
    #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

    #     start=time.time()
    #     n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
    #     print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)