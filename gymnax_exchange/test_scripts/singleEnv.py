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
    
    # -----------------------------------        
    import time
    timestamp = str(int(time.time()))
    rngInitNum = 1    
    rng = jax.random.PRNGKey(rngInitNum)
    

    
    
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)
    erlist = []
    excuted_list = []
    er = 0
    for i in range(1,10000):
        # ==================== ACTION ====================
        # ---------- acion from trained network ----------
        ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
        assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
        assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
        hstate, pi, value = network.apply(network_params, hstate, ac_in) 
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
        # ---------- acion from trained network ----------
        # ==================== ACTION ====================    
        print(f"Sampled {i}th actions are: ",action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,action, env_params)
        er+=reward
        print(f"Time for {i} step: \n",time.time()-start)
        print("excuted ",info["quant_executed"])
        excuted_list.append(info["quant_executed"])
        if done:
            erlist.append((i, er))
            # print(f"windowIndex {:<10} , global step {i:<10} , episodic return {er:^20} , ",\
            print(f" global step {i:<10} , episodic return {er:^20} , ",\
                file=open('twap_'+ timestamp +"_OneDay_train_"+'.txt','a'))
            network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
            init_hstate = ScannedRNN.initialize_carry(ppo_config["NUM_ENVS"], 128)
            init_x = (
                    jnp.zeros(
                        (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
                    ),
                    jnp.zeros((1, ppo_config["NUM_ENVS"])),
                )
            _rng,_= jax.random.split(_rng,2)
            network_params = network.init(_rng, init_hstate, init_x)
            key_reset, _ = jax.random.split(key_reset,2)
            obs,state=env.reset(key_reset,env_params)
            er =0
