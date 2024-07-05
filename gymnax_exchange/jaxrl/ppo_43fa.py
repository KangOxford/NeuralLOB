# from jax import config
# config.update("jax_enable_x64",True)

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import datetime
import dataclasses

#Code snippet to disable all jitting.

from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.




wandbOn = True
# wandbOn = False
if wandbOn:
    import wandb



class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        #pi = distrax.Categorical(logits=actor_mean)
        #Old version^^

        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        #New version ^^

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
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
    env = LogWrapper(env)
    
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        # NOTE: don't normalize reward for now
        # env = NormalizeVecReward(env, config["GAMMA"])
    

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # jax.debug.breakpoint()
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            """
            Pseudocode
            if i%50 ==0:
                envparam.message_data.reshuffled()
                envparam.book_data.reshuffled()
            
            reshuffled():
                0-30,30-60.... --> 5-35,35-65
                                    ...or 40-70,70-100
            
            """

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                # jax.debug.print('Step')
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng) # 4*1, should be (4*4: 4actions * 4envs)
                # Guess to be 4 actions. caused by ppo_rnn is continuous. But our action space is discrete
                log_prob = pi.log_prob(action)

                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done_step, action, value, reward_step, log_prob, last_obs, info_step
                )
                runner_state = (train_state, env_state_step, obsv_step, done_step, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    
                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]
                    average_price = info["average_price"][info["returned_episode"]]
                    current_step = info["current_step"][info["returned_episode"]]
                    benchmarkTotalRevenue = info["benchmarkTotalRevenue"][info["returned_episode"]]
                    advatange_in_bp = info["advatange_in_bp"][info["returned_episode"]]
                    
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
                    for t in range(len(timesteps)):  
                        if wandbOn:
                            wandb.log(
                                {
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed":quant_executed[t],
                                    "average_price":average_price[t],
                                    "current_step":current_step[t],
                                    "benchmarkTotalRevenue":benchmarkTotalRevenue[t],
                                    "advatange_in_bp":advatange_in_bp[t],
                                }
                            )        
                        else:
                            print(
                                f"global step={timesteps[t]:<11} | episodic return={return_values[t]:<11} | episodic revenue={revenues[t]:<11} | average_price={average_price[t]:<11}"
                            )     
                            # print("==="*20)      
                            # print(info["current_step"])  
                            # print(info["total_revenue"])  
                            # print(info["quant_executed"])   
                            # print(info["average_price"])   
                            # print(info["returned_episode_returns"])
                    # '''


                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    ATFolder = '/homes/80/kang/AlphaTrade/AMZN_data/'
    print("AlphaTrade folder:",ATFolder)

    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")

    ppo_config = {
        "LR": 1e-4, # 1e-4, 5e-4, #5e-5, #1e-4,#2.5e-5,
        "LR_COS_CYCLES": 8,  # only relevant if ANNEAL_LR == "cosine"
        "ENT_COEF": 0., # 0., 0.001, 0, 0.1, 0.01, 0.001
        "NUM_ENVS": 256, #512, 1024, #128, #64, 1000,
        "TOTAL_TIMESTEPS": 2e6,  # 1e8, 5e7, # 50MIL for single data window convergence #,1e8,  # 6.9h
        "NUM_MINIBATCHES": 4, #8, 4, 2,
        "UPDATE_EPOCHS": 10, #10, 30, 5,
        "NUM_STEPS": 10, #20, 512, 500,
        "CLIP_EPS": 0.2,  # TODO: should we change this to a different value? 
        
        "GAMMA": 0.999,
        "GAE_LAMBDA": 0.99, #0.95,
        "VF_COEF": 1.0, #1., 0.01, 0.001, 1.0, 0.5,
        "MAX_GRAD_NORM": 5, # 0.5, 2.0,
        "ANNEAL_LR": 'cosine', # 'linear', 'cosine', False
        "NORMALIZE_ENV": False,  # only norms observations (not reward)
        
        "RNN_TYPE": "GRU",  # "GRU", "S5"
        "HIDDEN_SIZE": 64,  # 128
        "ACTIVATION_FN": "relu", # "tanh", "relu", "leaky_relu", "sigmoid", "swish"
        "ACTION_NOISE_COLOR": 2.,  # 2  # only relevant if CONT_ACTIONS == True

        "RESET_ADAM_COUNT": True,  # resets Adam's t (count) every update
        "ADAM_B1": 0.99,  # 0.9
        "ADAM_B2": 0.99,
        "ADAM_EPS": 1e-5,  # 1e-4, 1e-6
        
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1, # 2 fix random episode #-1,
        # "WINDOW_INDEX": 2, # 2 fix random episode #-1,
        "DEBUG": True,
        
        "TASKSIDE": "sell", # "random", "buy", "sell"
        # "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 1., #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "MAX_TASK_SIZE": 100,
        "TASK_SIZE": 100, # 500,
        "EPISODE_TIME": 60 * 5, # time in seconds
        "DATA_TYPE": "fixed_steps", # "fixed_time", "fixed_steps"
        # "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "CONT_ACTIONS": False,  # True
        "JOINT_ACTOR_CRITIC_NET": True,  # True, False
        "ACTOR_STD": "state_dependent",  # 'state_dependent', 'param', 'fixed'
        "REDUCE_ACTION_SPACE_BY": 10,
      
        "ATFOLDER": "/homes/80/kang/AlphaTrade/AMZN_data/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        # "ATFOLDER": "./training_oneDay/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        # "ATFOLDER": "./training_oneMonth/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        "RESULTS_FILE": "training_runs/results_file_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR": "training_runs/checkpoints_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/checkpoints_"+f"{timestamp}",
    }
    
    
    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
        )
        import datetime;params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        print(f"Results would be saved to {params_file_name}")
    else:
        import datetime;params_file_name = f'params_file_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        print(f"Results would be saved to {params_file_name}")
        

    
    # +++++ Single GPU +++++
    rng = jax.random.PRNGKey(0)
    # rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(ppo_config))
    start=time.time()
    out = train_jit(rng)
    print("Time: ", time.time()-start)
    # +++++ Single GPU +++++

    # # +++++ Multiple GPUs +++++
    # num_devices = 4
    # rng = jax.random.PRNGKey(30)
    # rngs = jax.random.split(rng, num_devices)
    # train_fn = lambda rng: make_train(ppo_config)(rng)
    # start=time.time()
    # out = jax.pmap(train_fn)(rngs)
    # print("Time: ", time.time()-start)
    # # +++++ Multiple GPUs +++++
    
    

    # '''
    # # ---------- Save Output ----------
    import flax

    train_state = out['runner_state'][0] # runner_state.train_state
    params = train_state.params
    


    import datetime;params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'

    # Save the params to a file using flax.serialization.to_bytes
    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"pramas saved")

    # Load the params from the file using flax.serialization.from_bytes
    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"pramas restored")
        
    # jax.debug.breakpoint()
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, restored_params))
    # print(">>>")
    # '''

    if wandbOn:
        run.finish()

