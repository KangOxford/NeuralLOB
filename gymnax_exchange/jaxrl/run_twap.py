import datetime
import time
import sys
import jax
import jax.numpy as jnp
import dataclasses
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxrl.ppo_mm import Transition
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper, ClipAction, VecEnv, NormalizeVecObservation, NormalizeVecReward


wandbOn = False # False
if wandbOn:
    import wandb

####### HEURISTIC POLICY FUNCTIONS #######

def random_policy(env, obs, rng):
    rng, _rng = jax.random.split(rng)
    return env.action_space().sample(_rng)

def twap_aggr(env, obs, rng):
    frontloading = 0.1
    steps_left = calc_steps_left(env, obs, frontloading)
    step_quant = obs['remaining_quant'] // steps_left
    action = jnp.zeros((env.n_actions,)).at[0].set(step_quant)
    return action

def twap_pass(env, obs, rng):
    frontloading = 0.1
    overstuff_factor = 2
    steps_left = calc_steps_left(env, obs, frontloading)
    step_quant = (obs['remaining_quant'] // steps_left) * overstuff_factor
    action = jnp.zeros((env.n_actions,)).at[-1].set(step_quant)
    return action

def all_passive(env, obs, rng):
    return jnp.zeros((env.n_actions,)).at[-1].set(obs['remaining_quant'])

def calc_steps_left(env, obs, frontloading):
    if env.ep_type == "fixed_steps":
        steps_left = obs["max_steps"] - obs["step_counter"]
    elif env.ep_type == "fixed_time":
        steps_left = jax.lax.cond(
            obs['delta_time'] == 0,
            lambda: obs["max_steps"] - obs["step_counter"],
            lambda: (obs["time_remaining"] // obs['delta_time']).astype(jnp.int32),
        )
    steps_left = jnp.clip((steps_left * (1-frontloading)).astype(jnp.int32), 1, None)
    return steps_left

##########################################

def make_run(config, rng):
    env = ExecutionEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        rewardLambda=config["REWARD_LAMBDA"],
        ep_type=config["DATA_TYPE"],
    )
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=config["REWARD_LAMBDA"],
        task_size=config["TASK_SIZE"],
        episode_time=config["EPISODE_TIME"],
    )

    if config["POLICY"] == "random":
        policy_fn = lambda obs, rng: random_policy(env, obs, rng)
    elif config["POLICY"] == "twap_aggr":
        policy_fn = lambda obs, rng: twap_aggr(env, obs, rng)
    elif config["POLICY"] == "twap_pass":
        policy_fn = lambda obs, rng: twap_pass(env, obs, rng)
    elif config["POLICY"] == "all_passive":
        policy_fn = lambda obs, rng: all_passive(env, obs, rng)
    else:
        raise ValueError("Invalid policy function " + str(config["POLICY"]))

    env = LogWrapper(env)
    
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        # NOTE: don't normalize reward for now
        # env = NormalizeVecReward(env, config["GAMMA"])

    def run(runner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            env_state, last_obs, last_done, rng = runner_state
            rng, _rng = jax.random.split(rng)
            rng_actions = jax.random.split(_rng, config["NUM_ENVS"])

            # TODO: SELECT ACTION
            # hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

            # action = env.action_space().sample(_rng)
            # action = jax.vmap(env.action_space().sample)(rng_actions)
            action = jax.vmap(policy_fn)(
                jax.vmap(env._get_obs, in_axes=(0, None, None, None))(
                    env_state.env_state, env_params, False, False
                ),
                rng_actions
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])

            obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)
            value = jnp.nan
            log_prob = jnp.nan
            transition = Transition(
                done_step, action, value, reward_step, log_prob, last_obs, info_step
            )
            runner_state = (env_state_step, obsv_step, done_step, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, xs=None, length=config["NUM_STEPS"]
        )

        if config["DEBUG"]:

            def callback(info):
                
                # update_step, info, trainstate_logs, trainstate_params = metric
                
                return_values = info["returned_episode_returns"][
                    info["returned_episode"]
                ]
                timesteps = (
                    info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                )
                
                revenues = info["total_revenue"][info["returned_episode"]]
                quant_executed = info["quant_executed"][info["returned_episode"]]
                average_price = info["average_price"][info["returned_episode"]]
                
                # slippage_rm = info["slippage_rm"][info["returned_episode"]]
                # price_drift_rm = info["price_drift_rm"][info["returned_episode"]]
                # price_adv_rm = info["price_adv_rm"][info["returned_episode"]]
                # vwap_rm = info["vwap_rm"][info["returned_episode"]]
                
                current_step = info["current_step"][info["returned_episode"]]
                mkt_forced_quant = info["mkt_forced_quant"][info["returned_episode"]]
                doom_quant = info["doom_quant"][info["returned_episode"]]
                trade_duration = info["trade_duration"][info["returned_episode"]]
                advantage_reward = info["advantage_reward"][info["returned_episode"]]
                drift_reward = info["drift_reward"][info["returned_episode"]]
                
                '''
                print(info["current_step"][0,0],info["total_revenue"][0,0],info["average_price"][0,0],info['quant_executed'][0,0],info['action'][0,0])  
                if info['done']: print("==="*10 + str(info["window_index"]) +"==="*10 + '\n')      
                # if info['done']: print("==="*10 + "==="*10 + '\n')      
                # if info['done']: print("==="*10 + str(info["window_index"])[0,0] + "==="*10 + '\n')      
                # print(info["total_revenue"])  
                # print(info["quant_executed"])   
                # print(info["average_price"])   
                # print(info["returned_episode_returns"])
                '''
                
                # '''
                for t in range(0, len(timesteps)):
                # for t in range(len(timesteps)):
                    if wandbOn:
                        wandb.log(
                            data={
                                "global_step": timesteps[t] * 100,  # NOTE: increase x-axis (since the policy is fixed)
                                "episodic_return": return_values[t],
                                "episodic_revenue": revenues[t],
                                "quant_executed": quant_executed[t],
                                "average_price": average_price[t],
                                # "slippage_rm":slippage_rm[t],
                                # "price_adv_rm":price_adv_rm[t],
                                # "price_drift_rm":price_drift_rm[t],
                                # "vwap_rm":vwap_rm[t],
                                "current_step": current_step[t],
                                "advantage_reward": advantage_reward[t],
                                "drift_reward": drift_reward[t],
                                "mkt_forced_quant": mkt_forced_quant[t],
                                "doom_quant": doom_quant[t],
                                "trade_duration": trade_duration[t],
                            },
                            commit=True
                        )        
                    else:
                        print(
                            # f"global step={timesteps[t]:<11} | episodic return={return_values[t]:.10f<15} | episodic revenue={revenues[t]:.10f<15} | average_price={average_price[t]:<15}"
                            f"global step={timesteps[t]:<11} | episodic return={return_values[t]:<20} | episodic revenue={revenues[t]:<20} | average_price={average_price[t]:<11}"
                        )     
                        # print("==="*20)      
                        # print(info["current_step"])  
                        # print(info["total_revenue"])  
                        # print(info["quant_executed"])   
                        # print(info["average_price"])   
                        # print(info["returned_episode_returns"])
                # '''


            jax.debug.callback(callback, traj_batch.info)
        
        return runner_state, None
    
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    rng, _rng = jax.random.split(rng)

    runner_state = (
        env_state,
        obsv,
        jnp.zeros((config["NUM_ENVS"]), dtype=bool),
        _rng,
    )

    # loop run until total timesteps
    return lambda: jax.lax.scan(
        run, init=runner_state, xs=None, length=config["NUM_UPDATES"]
    )
    # return run

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")

    # get command line arg
    policy_name = sys.argv[1]

    config = {
        "POLICY": policy_name, # "random", "twap_aggr", "twap_pass", "all_passive",
        "NUM_ENVS": 256, #512, 1024, #128, #64, 1000,
        "TOTAL_TIMESTEPS": 5e5,  # 
        "NUM_STEPS": 20,
        
        "GAMMA": 0.999,
        "NORMALIZE_ENV": False,  # only norms observations (not reward)
        
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1, # 2 fix random episode #-1,
        "DEBUG": True,
        
        "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 0.1, #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "MAX_TASK_SIZE": 100,
        "TASK_SIZE": 100, # 500,
        "EPISODE_TIME": 60 * 5, # time in seconds
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "REDUCE_ACTION_SPACE_BY": 10,
      
        "ATFOLDER": "./training_oneDay/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        # "ATFOLDER": "./training_oneMonth/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        "RESULTS_FILE": "training_runs/results_file_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR": "training_runs/checkpoints_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/checkpoints_"+f"{timestamp}",
    }
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
        )

    # +++++ Single GPU +++++
    rng = jax.random.PRNGKey(0)
    # rng = jax.random.PRNGKey(30)
    run_jit = jax.jit(make_run(config, rng))
    start = time.time()
    out = run_jit()
    print("Time: ", time.time() - start)
    # +++++ Single GPU +++++

    if wandbOn:
        run.finish()
