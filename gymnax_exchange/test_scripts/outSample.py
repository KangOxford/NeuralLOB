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

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing'
        ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    # print(env_params.message_data.shape, env_params.book_data.shape)
    
    
    ppo_config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 1,
            "NUM_STEPS": 1,
            "TOTAL_TIMESTEPS": 5e5,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ENV_NAME": "alphatradeExec-v0",
            "ANNEAL_LR": True,
            "DEBUG": True,
            "NORMALIZE_ENV": False,
            "ATFOLDER": ATFolder,
            "TASKSIDE":'buy'
        }
    import flax
    from gymnax_exchange.jaxrl.ppo_mm import ActorCriticRNN
    from gymnax_exchange.jaxrl.ppo_mm import ScannedRNN
    with open('/homes/80/kang/AlphaTrade/params_file_smooth-paper-19_09-23_04-30', 'rb') as f:
    # with open('/homes/80/kang/AlphaTrade/params_file_prime-armadillo-72_07-17_11-02', 'rb') as f:
    # with open('/homes/80/kang/AlphaTrade/params_file_firm-fire-68_07-17_09-53', 'rb') as f:
    # with open('/homes/80/kang/AlphaTrade/params_file_2023-07-10_12-34-24', 'rb') as f:
    # with open('/homes/80/kang/AlphaTrade/params_file_2023-07-08_15-22-20', 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"pramas restored")
    network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
    init_hstate = ScannedRNN.initialize_carry(ppo_config["NUM_ENVS"], 128)
    init_done = jnp.array([False]*ppo_config["NUM_ENVS"])
    ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
    assert len(ac_in[0].shape) == 3
    hstate, pi, value = network.apply(restored_params, init_hstate, ac_in)
    print("Network Carry Initialized")
    action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
    print(f"-------------\nPPO 0th actions are: {action} with sum {action.sum()}")
    obs,state,reward,done,info=env.step(key_step, state, action, env_params)
    # done, state.quant_executed
    print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")

    
    i = 1
    for i in range(1,10000):
        # ==================== ACTION ====================
        # ---------- acion from trained network ----------
        ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
        assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
        assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
        hstate, pi, value = network.apply(restored_params, hstate, ac_in) 
        # hstate, pi, value = network.apply(restored_params, hstate, ac_in) # TODO does hstate need to be from the out?
        # hstate, pi, value = network.apply(runner_state.train_state.params, hstate, ac_in)
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
        # ---------- acion from trained network ----------
        # ==================== ACTION ====================    
        print(f"-------------\nPPO {i}th actions are: {action} with sum {action.sum()}")
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,action, env_params)
        print(f"Time for {i} step: \n",time.time()-start)
        # done, state.quant_executed
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
        if done:
            break
    # print(info)
