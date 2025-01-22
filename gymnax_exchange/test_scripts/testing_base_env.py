import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
from AlphaTrade.gymnax_exchange.jaxen.old.base_env_old import BaseLOBEnv
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
import chex
import time

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

#Code snippet to disable all jitting.
#from jax import config
#config.update("jax_disable_jit", True)


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'

    # enable_vmap=True 
    # enable_2nd_singles=True
    enable_vmap=False 
    enable_2nd_singles=False

    rng = jax.random.PRNGKey(234)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=BaseLOBEnv(ATFolder) 
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    
    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)

    if enable_2nd_singles:
        #jax.profiler.start_trace("/tmp/tensorboard")
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        obs.block_until_ready()
        #jax.profiler.stop_trace()
        print("State after reset: \n",state)
        print("Time for 2nd reset: \n",time.time()-start)
        print(env_params.message_data.shape, env_params.book_data.shape)



    """test_action={"sides":jnp.array([1,1,1]),
                 "quantities":jnp.array([10,10,10]),
                 "prices":jnp.array([2154900,2154000,2153900]),
                 }"""
    
    test_action=env.action_space().sample(key_policy)
    print("Sampled actions are: ",test_action)

    start=time.time()
    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    print("State after one step: \n",state,done)
    print("Time for one step: \n",time.time()-start)

    if enable_2nd_singles:
        test_action=env.action_space().sample(key_policy)
        print("Sampled actions are: \n",test_action)
        
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        print("State after 2 steps: \n",state,done)
        print("Time for 2nd step: \n",time.time()-start)
        #comment


    ####### Testing the vmap abilities ########
    
    if enable_vmap:
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 100000
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)
        
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)

        start=time.time()
        for i in range(10):
            n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
