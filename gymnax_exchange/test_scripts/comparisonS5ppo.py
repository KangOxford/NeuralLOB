# from jax import config
# config.update("jax_enable_x64",True)
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
from AlphaTrade.gymnax_exchange.jaxen.old.exec_env_old import *
import json
# ============== testing scripts ===============

paramsFile = '/homes/80/kang/AlphaTrade/params_file_smooth-paper-19_09-23_04-30'
# paramsFile = '/homes/80/kang/AlphaTrade/params_file_dutiful-thunder-5_07-21_18-48'

# def twapV3(state, env_params):
#     # ---------- ifMarketOrder ----------
#     remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
#     marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
#     ifMarketOrder = (remainingTime <= marketOrderTime)
#     # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
#     # ---------- ifMarketOrder ----------
#     # ---------- quants ----------
#     remainedQuant = state.task_to_execute - state.quant_executed
#     remainedStep = state.max_steps_in_episode - state.step_counter
#     stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
#     limit_quants = jax.random.permutation(key_policy, jnp.array([stepQuant//2,stepQuant-stepQuant//2,stepQuant//2,stepQuant-stepQuant//2]), independent=True)
#     market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
#     quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
#     # ---------- quants ----------
#     return jnp.array(quants) 

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        ATFolder = '/homes/80/kang/AlphaTrade/testing_oneDay'
        
    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)
    assert env.task_size == 500
    import time; timestamp = str(int(time.time()))
    
    def get_ppo_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        obs,state=env.reset(key_reset,env_params)
        ppo_config = {
                "LR": 2.5e-4,
                "NUM_ENVS": 1, # CAUTION !!!
                "NUM_STEPS": 1,
                "TOTAL_TIMESTEPS": 1e8,
                "UPDATE_EPOCHS": 1,
                "NUM_MINIBATCHES": 1,
                "GAMMA": 0.99,
                "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2,
                "ENT_COEF": 0.01,
                "VF_COEF": 0.5,
                "MAX_GRAD_NORM": 0.5,
                "ANNEAL_LR": True,
                "DEBUG": True,
                
                "ENV_NAME": "alphatradeExec-v0",
                "NORMALIZE_ENV": False,
                "ENV_LENGTH": "oneWindow",
                "ATFOLDER": ATFolder,
                "TASKSIDE":'sell',
                "LAMBDA":0.0,
                "TASK_SIZE":500,
            }
        import flax
        from gymnax_exchange.jaxrl.ppoS5ExecCont import ActorCriticS5
        from gymnax_exchange.jaxrl.ppoS5ExecCont import StackedEncoderModel, ssm_size, n_layers
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=ppo_config)
        init_hstate = StackedEncoderModel.initialize_carry(ppo_config["NUM_ENVS"], ssm_size, n_layers)
            
        # ===================================================
        # CHOICE ONE
        with open(paramsFile, 'rb') as f:
            restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        # ---------------------------------------------------
        # init_x = (
        #     jnp.zeros(
        #         (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
        #     ),
        #     jnp.zeros((1, ppo_config["NUM_ENVS"])),
        # )
        # network_params = network.init(key_policy, init_hstate, init_x)
        # restored_params = network_params
        # CHOICE OTWO
        # ===================================================
        
        
        init_done = jnp.array([False]*ppo_config["NUM_ENVS"])
        ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
        assert len(ac_in[0].shape) == 3
        hstate, pi, value = network.apply(restored_params, init_hstate, ac_in)
        print("Network Carry Initialized")
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
        print(f"-------------\nPPO 0th actions are: {action} with sum {action.sum()}")
        obs,state,reward,done,info=env.step(key_step, state, action, env_params)
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
        excuted_list = []
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from trained network ----------
            ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
            assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
            assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
            hstate, pi, value = network.apply(restored_params, hstate, ac_in) 
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
            # ---------- acion from trained network ----------
            # ==================== ACTION ====================    
            print(f"-------------\nPPO {i}th actions are: {action} with sum {action.sum()}")
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], excuted_list
    
    def get_twap_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        print("Time for reset: \n",time.time()-start)
        print(env_params.message_data.shape, env_params.book_data.shape)
        excuted_list = []
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from given strategy  ----------
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
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
                limit_quants_agressive = jnp.append(jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4]), independent=True),max(stepQuant - 3*stepQuant//4,stepQuant//4))
                limit_quants_passive = jnp.append(jax.random.permutation(key_policy, jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4]), independent=True),remainedQuant//4)
                limit_quants = jnp.where(limit_quants_agressive.sum() <= remainedQuant,limit_quants_agressive,limit_quants_passive)
                market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
                quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                # ---------- quants ----------
                return jnp.array(quants)
            def twapV3(state, env_params):
                # ---------- ifMarketOrder ----------
                remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                ifMarketOrder = (remainingTime <= marketOrderTime)
                # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
                # ---------- ifMarketOrder ----------
                # ---------- quants ----------
                remainedQuant = state.task_to_execute - state.quant_executed
                remainedStep = state.max_steps_in_episode - state.step_counter
                stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                limit_quants = jax.random.permutation(key_policy, jnp.array([stepQuant//2,stepQuant-stepQuant//2,stepQuant//2,stepQuant-stepQuant//2]), independent=True)
                market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
                quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                # ---------- quants ----------
                return jnp.array(quants) 
               
            twap_action = twapV3(state, env_params)
            print(f"Sampled {i}th actions are: ",twap_action)
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,twap_action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'], info['average_price'], excuted_list
    def get_random_average_price(rngInitNum):
        # ---------- init probabilities ----------
        import numpy as np
        p_0_1 = 0.9 # Define the probabilities for the numbers 0 and 1
        p_2_10 = 0.1 # Define the remaining probability for the numbers 2 through 10
        numbers_2_10 = np.arange(2, 200) # Define the numbers 2 through 10
        pareto_distribution = (1 / numbers_2_10)**4 # Generate a Pareto distribution for the numbers 2 through 200
        pareto_distribution /= pareto_distribution.sum()
        pareto_distribution *= p_2_10
        probabilities = np.array([p_0_1 / 2, p_0_1 / 2] + list(pareto_distribution)) # Combine the probabilities for all numbers from 0 to 10
        assert np.isclose(probabilities.sum(), 1.0) # Verify that the probabilities sum to 1
        # ---------- init probabilities ----------
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        print("Time for reset: \n",time.time()-start)
        excuted_list = []
        for i in range(1,10000):
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
            def randomV1(state, env_params):
                quants = np.random.choice(np.arange(0, 200), size=4, p=probabilities) # Generate random data from the custom distribution
                return jnp.array(quants)                
            random_action = randomV1(state, env_params)
            print(f"Sampled {i}th actions are: ",random_action)
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,random_action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'], info['average_price'], excuted_list
    def get_hush_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        print("Time for reset: \n",time.time()-start)
        excuted_list = []
        for i in range(1,10000):
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
            def rushV1(state, env_params):
                quants = np.random.choice(np.arange(0, 200), size=4) # Generate random data from the custom distribution
                return jnp.array(quants)                
            random_action = rushV1(state, env_params)
            print(f"Sampled {i}th actions are: ",random_action)
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,random_action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'], info['average_price'], excuted_list
    # def get_best_price_average_price(rngInitNum):
    #     rng = jax.random.PRNGKey(rngInitNum)
    #     rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    #     start=time.time()
    #     obs,state=env.reset(key_reset,env_params)
    #     print("Time for reset: \n",time.time()-start)
    #     excuted_list = []
    #     for i in range(1,10000):
    #         print("---"*20)
    #         print("window_index ",state.window_index)
    #         key_policy, _ = jax.random.split(key_policy,2)
    #         def bestPrice(state, env_params):
    #             quants = jnp.array([min(remaining Q, size at best price), 0, 0, 0])
    #             return jnp.array(quants)                
    #         random_action = bestPrice(state, env_params)
    #         print(f"Sampled {i}th actions are: ",random_action)
    #         start=time.time()
    #         obs,state,reward,done,info=env.step(key_step, state,random_action, env_params)
    #         print(f"Time for {i} step: \n",time.time()-start)
    #         print("excuted ",info["quant_executed"])
    #         excuted_list.append(info["quant_executed"])
    #         if done:
    #             break
    #     return info['window_index'], info['average_price'], excuted_list
    def get_advantage(rngInitNum):
        window_index2,twap,executed_list2=get_twap_average_price(rngInitNum)
        window_index1,ppo,executed_list1=get_ppo_average_price(rngInitNum)
        window_index3,random,executed_list3=get_random_average_price(rngInitNum)
        window_index4,rush,executed_list4=get_hush_average_price(rngInitNum)
        assert window_index1 == window_index2
        assert window_index1 == window_index3
        assert window_index1 == window_index4
        return window_index1, (ppo-twap)/twap*10000, (ppo-random)/random*10000, (ppo-rush)/rush*10000, ppo, twap, random, rush, executed_list1, executed_list2, executed_list3, executed_list4
    # result_list = []
    for rngInitNum in range(100,1000):
        print(f"++++ rngInitNum {rngInitNum}")
        result_tuple = get_advantage(rngInitNum) 
        # result_list.append(result_tuple[0]) # window index
        print(f"window_index {result_tuple[0]:<4} , advantageTWAP {result_tuple[1]:^20} , advantageRANDOM {result_tuple[2]:^20} , advantageRUSH {result_tuple[3]:^20} , ppoAP {result_tuple[4]:<20} , twapAP {result_tuple[5]:<20} , randomAP {result_tuple[6]:<20} , rushAP {result_tuple[7]:<20} , ppoExecuted { [int(x) for x in result_tuple[8]]} , twapExecuted { [int(x) for x in result_tuple[9]]} , randomExecuted { [int(x) for x in result_tuple[10]]} , rushExecuted { [int(x) for x in result_tuple[11]]}",\
            file=open('comparison'+ paramsFile.split('_')[-3] +"_OneMonth_"+timestamp+'.txt','a'))
        
        
        
        


