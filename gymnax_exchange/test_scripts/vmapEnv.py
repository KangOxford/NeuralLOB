# ============== testing scripts ===============
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
from gymnax_exchange.jaxen.exec_env import *
import json
# ============== testing scripts ===============

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)
    assert env.task_size == 500    
    
    import time;timestamp = str(int(time.time()))
    rngInitNum = 0;rng = jax.random.PRNGKey(rngInitNum)
    
    
    
    enable_vmap=True
    # if enable_vmap:
    #     # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
    #     vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
    #     vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    #     vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))
    #     # print(test_actions)

    #     num_envs = 10
    #     rng, resetKey, actionKey, stepKey = jax.random.split(rng, 4)


    #     start=time.time()
    #     resetKeys =jax.random.split(resetKey, num_envs)
    #     actionKeys =jax.random.split(actionKey, num_envs)
    #     stepKeys =jax.random.split(stepKey, num_envs)
    #     obs, state = vmap_reset(resetKeys, env_params)
    #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)
    #     total_steps = 1000*10
    #     for i in range(total_steps//num_envs):
    #         start=time.time()
    #         actionKeys = jax.random.split(actionKeys[0], num_envs)
    #         stepKeys = jax.random.split(stepKeys[0], num_envs)
    #         test_actions=vmap_act_sample(actionKeys)
    #         n_obs, n_state, n_reward, n_done, n_info = vmap_step(stepKeys, state, test_actions, env_params)
    #         print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
    #         print(f"\n-------- {i} --------")
    #         print(test_actions)
    #         print(n_done)
    #         print("stepKeys",stepKeys)
    #         print("current_step",n_info["current_step"])
    #         print("window_index",n_info["window_index"])
    #         if any(n_done):
    #             break
    #         # if done:
    #         #     vmap_keys = jax.random.split(vmap_keys, num_envs)
    #         #     start=time.time()
    #         #     obs, state = vmap_reset(vmap_keys, env_params)
    #         #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)
                
    if enable_vmap:
        vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        vmap_act_sample=jax.jit(jax.vmap(env.action_space().sample, in_axes=(0)))

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

        def step_wrap_vmap(runner_state,unused):
            jax.debug.print('step')
            env_state, obsv, done, rng=runner_state
            rng,_rng=jax.random.split(rng)
            vmap_keys = jax.random.split(_rng, num_envs)
            test_actions=vmap_act_sample(vmap_keys).astype(jnp.float32)
            obsv,env_state,reward,done,info=vmap_step(vmap_keys, env_state, test_actions, env_params)
            runner_state = (env_state, obsv, done, rng)
            jax.debug.print("---done {}",done)
            return runner_state,None
    
        r_state=(n_state, n_obs, done, rng)


        n_steps=500
        # n_steps=globalSteps/num_envs


        def scan_func_vmap(r_state,n_steps):
            r_state,_=jax.lax.scan(step_wrap_vmap,r_state,None,n_steps)
            return r_state

        start=time.time()
        r_state=jax.jit(scan_func_vmap,static_argnums=(1,))(r_state,n_steps)
        print("Time for vmap step with,",num_envs, " environments and",n_steps," steps: \n",time.time()-start)                
                
                
                
                
                
                
    # rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    # start=time.time()
    # obs,state=env.reset(key_reset,env_params)
    # print("Time for reset: \n",time.time()-start)
    # print(env_params.message_data.shape, env_params.book_data.shape)
    # erlist = []
    # excuted_list = []
    # er = 0
    # for i in range(1,int(1e7)):
    #     # ==================== ACTION ====================
    #     # ---------- acion from given strategy  ----------
    #     print("---"*20)
    #     print("window_index ",state.window_index)
    #     key_policy, _ = jax.random.split(key_policy,2)
    #     def twap(state, env_params):
    #         # ---------- ifMarketOrder ----------
    #         remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
    #         marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
    #         ifMarketOrder = (remainingTime <= marketOrderTime)
    #         print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
    #         # ---------- ifMarketOrder ----------
    #         # ---------- quants ----------
    #         remainedQuant = state.task_to_execute - state.quant_executed
    #         remainedStep = state.max_steps_in_episode - state.step_counter
    #         stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
    #         limit_quants_agressive = jnp.append(jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4]), independent=True),max(stepQuant - 3*stepQuant//4,stepQuant//4))
    #         limit_quants_passive = jnp.append(jax.random.permutation(key_policy, jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4]), independent=True),remainedQuant//4)
    #         limit_quants = jnp.where(limit_quants_agressive.sum() <= remainedQuant,limit_quants_agressive,limit_quants_passive)
    #         market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
    #         quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
    #         # ---------- quants ----------
    #         return jnp.array(quants)
    #     def twapV3(state, env_params):
    #         # ---------- ifMarketOrder ----------
    #         remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
    #         marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
    #         ifMarketOrder = (remainingTime <= marketOrderTime)
    #         # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
    #         # ---------- ifMarketOrder ----------
    #         # ---------- quants ----------
    #         remainedQuant = state.task_to_execute - state.quant_executed
    #         remainedStep = state.max_steps_in_episode - state.step_counter
    #         stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
    #         limit_quants = jax.random.permutation(key_policy, jnp.array([stepQuant//2,stepQuant-stepQuant//2,stepQuant//2,stepQuant-stepQuant//2]), independent=True)
    #         market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
    #         quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
    #         # ---------- quants ----------
    #         return jnp.array(quants) 
            
    #     twap_action = twapV3(state, env_params)
    #     print(f"Sampled {i}th actions are: ",twap_action)
    #     start=time.time()
    #     obs,state,reward,done,info=env.step(key_step, state,twap_action, env_params)
    #     er+=reward
    #     print(f"Time for {i} step: \n",time.time()-start)
    #     print("excuted ",info["quant_executed"])
    #     excuted_list.append(info["quant_executed"])
    #     if done:
    #         erlist.append((i, er))
    #         print(f"global step {i:<10} , episodic return {er:^20} , ",\
    #             file=open('twap_'+ timestamp +"_OneDay_train_"+'.txt','a'))
    #         key_reset, _ = jax.random.split(key_reset)
    #         obs,state=env.reset(key_reset,env_params)
    #         er =0