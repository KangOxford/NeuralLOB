import dataclasses
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import time
from typing import Optional, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex

sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper, ClipAction, VecEnv, NormalizeVecObservation, NormalizeVecReward
from purejaxrl.experimental.s5.s5 import StackedEncoderModel
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxrl.actorCritic import ActorCriticRNN, ScannedRNN
from gymnax_exchange.jaxrl import actorCriticS5
from gymnax_exchange.jaxrl.ActorCriticMoE import ActorCriticMoE
import flax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
from jax import config
config.update("jax_disable_jit", False)
config.update("jax_check_tracer_leaks", False)
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)

wandbOn = False
if wandbOn:
    import wandb

def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def reset_adam(train_state, reset_type="count"):
    inner_state = train_state.opt_state[1].inner_state
    if reset_type == "count":
        inner_state = (inner_state[0]._replace(count=0), inner_state[1])
    elif reset_type == "all":
        inner_state = jax.tree_map(jnp.zeros_like, inner_state)
    opt_state = (
        train_state.opt_state[0],
        train_state.opt_state[1]._replace(inner_state=inner_state),
    )
    return train_state.replace(opt_state=opt_state)

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
    
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    @jax.jit
    def cosine_lr(count):
        count = count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        n_cycles = config["LR_COS_CYCLES"]
        T_i = config["NUM_UPDATES"] / n_cycles / 2
        lr_min = config["LR"] / 10
        lr_max = config["LR"]
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * count / T_i))

    @jax.jit
    def lin_cos_lr(step):
        cos = cosine_lr(step)
        frac = (
            1.0 - (step // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        )
        return cos * frac

    def train(rng):
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )

        network = ActorCriticMoE(config=config, action_dim=env.action_space(env_params).shape[0], num_experts=config['NUM_EXPERTS'], k=config['K'])
        network_params = network.init(_rng, init_x, key=_rng)
        
        if config["ANNEAL_LR"] == "linear":
            lr = linear_schedule
        elif config["ANNEAL_LR"] == "cosine":
            lr = lin_cos_lr
        else:
            lr = config["LR"]
        
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(learning_rate=lr, b1=config["ADAM_B1"], b2=config["ADAM_B2"], eps=config["ADAM_EPS"]),
        )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, action_noise):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, ac_in, key=_rng)

                if config["CONT_ACTIONS"]:
                    a_mean = pi._loc
                    a_std = pi._scale_diag
                    action = action_noise * a_std + a_mean
                else:
                    action = pi.sample(seed=_rng) * config["REDUCE_ACTION_SPACE_BY"]

                log_prob = pi.log_prob(action // config["REDUCE_ACTION_SPACE_BY"])
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

            update_step = runner_state[-1]
            initial_hstate = runner_state[-3]
            rng, rng_ = jax.random.split(runner_state[-2])
            runner_state = runner_state[:-2] + (rng,) + runner_state[-1:]
            if config["CONT_ACTIONS"]:
                col_noise = cnoise.powerlaw_psd_gaussian(config["ACTION_NOISE_COLOR"], (network.action_dim, len(runner_state[2]), config["NUM_STEPS"]), _rng, 0.).T
            else:
                col_noise = None
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state[:-1], col_noise, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
            _, _, last_val = network.apply(train_state.params, ac_in, key=_rng)
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
                    train_state, _ = train_state
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        def debug_log(metric):
                            (update_step, info, dead_ratio, obs_norm, action_mean, action_std) = metric
                            data = {
                                "update_step": update_step,
                                "global_step": info["timestep"][info["returned_episode"]] * config["NUM_ENVS"],
                                "dead_neurons": dead_ratio,
                                "obs_norm": obs_norm,
                            }

                            action_mean = action_mean.mean(axis=0)
                            if (config["ACTOR_STD"] == "state_dependent") or (not config["CONT_ACTIONS"]):
                                action_std = action_std.mean(axis=0)
                                for i in range(action_mean.shape[0]):
                                    data[f"action_mean_{i}"] = action_mean[i]
                                    data[f"action_std_{i}"] = action_std[i]
                            else:
                                data[f"action_std_0"] = action_std
                                for i in range(action_mean.shape[0]):
                                    data[f"action_mean_{i}"] = action_mean[i]
                                    
                            wandb.log(
                                data=data,
                                commit=False
                            )
                        
                        filter_neurons = lambda mdl, method_name: isinstance(mdl, nn.LayerNorm) or isinstance(mdl, StackedEncoderModel)

                        if config['RNN_TYPE'] == "GRU":
                            if config['JOINT_ACTOR_CRITIC_NET']:
                                init_hstate = init_hstate[0]
                            else:
                                init_hstate = (init_hstate[0][0], init_hstate[1][0])
                        (_, pi, value), network_state = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done), key=_rng,
                            capture_intermediates=filter_neurons, mutable=["intermediates"]
                        )
                        log_prob = pi.log_prob(traj_batch.action // config["REDUCE_ACTION_SPACE_BY"])
                        activations = network_state["intermediates"]
                        dead_ratio = _dead_neuron_ratio(activations)
                        obs_norm = jnp.sqrt((traj_batch.obs**2).sum(axis=-1)).mean()
                        
                        if config["CONT_ACTIONS"]:
                            action_mean = pi._loc.squeeze()[-1]
                            action_std = pi._scale_diag.squeeze()[-1]
                        else:
                            action_mean = pi.distribution.probs[-1] @ jnp.arange(0, config["MAX_TASK_SIZE"] + 1, config["REDUCE_ACTION_SPACE_BY"])
                            action_std = jnp.sqrt(
                                pi.distribution.probs[-1].T \
                                @ (
                                    jnp.tile(
                                        jnp.arange(0, config["MAX_TASK_SIZE"] + 1, config["REDUCE_ACTION_SPACE_BY"]),
                                        pi.event_shape + (action_mean.shape[0], 1),
                                    ).T - action_mean
                                ) ** 2
                            )
                            action_std = jnp.diagonal(action_std.T)

                        metric = (
                            update_step,
                            traj_batch.info,
                            dead_ratio,
                            obs_norm,
                            action_mean,
                            action_std,
                        )
                        if wandbOn:
                            jax.debug.callback(debug_log, metric)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

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

                        losses = (
                            config["VF_COEF"] * value_loss,
                            loss_actor,
                            - config["ENT_COEF"] * entropy
                        )
                        losses = (losses[0] + losses[1] + losses[2], *losses)
                        return losses, (value_loss, loss_actor, entropy)

                    total_loss = _loss_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    total_loss = total_loss[0][0], total_loss[1]
                    jac = jax.jacrev(_loss_fn, has_aux=True)(train_state.params, init_hstate, traj_batch, advantages, targets)
                    grads = jac[0][0]
                    train_state = train_state.apply_gradients(grads=grads)
                    grad_parts_norm = jnp.array([
                        optax.global_norm(g) for g in jac[0]
                    ])

                    return (train_state, grad_parts_norm), total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    _
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

                (train_state, grad_norm), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, jnp.zeros(4,)), minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    grad_norm
                )
                return update_state, total_loss

            if config["RNN_TYPE"] == "GRU":
                if config["JOINT_ACTOR_CRITIC_NET"]:
                    init_hstate = initial_hstate[None, :]
                else:
                    init_hstate = (
                        initial_hstate[0][None, :],
                        initial_hstate[1][None, :]
                    )
            else:
                init_hstate = initial_hstate

            if config["RESET_ADAM_COUNT"]:
                train_state = reset_adam(train_state)

            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
                jnp.zeros((4,))
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            trainstate_logs = {
                "learning_rate": train_state.opt_state[1].hyperparams["learning_rate"],
                "grad_norm": update_state[6][0],
                "v_grad_norm": update_state[6][1],
                "a_grad_norm": update_state[6][2],
                "ent_grad_norm": update_state[6][3],
                "mean_loss": jnp.mean(loss_info[0]),
                "mean_value_loss": jnp.mean(loss_info[1][0]),
                "mean_actor_loss": jnp.mean(loss_info[1][1]),
                "mean_entropy_loss": jnp.mean(loss_info[1][2]),
            }
            metric = (update_step, traj_batch.info, trainstate_logs, train_state.params)
            rng = update_state[5]

            if config.get("DEBUG"):
                def callback(metric):
                    update_step, info, trainstate_logs, trainstate_params = metric
                    
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    
                    def evaluation():
                        if not os.path.exists(config['CHECKPOINT_DIR']): os.makedirs(config['CHECKPOINT_DIR'])
                        if any(timesteps % int(1e3) == 0) and len(timesteps) > 0:
                            start = time.time()
                            jax.debug.print(">>> checkpoint saving {}", round(timesteps[0], -3))
                            checkpoint_filename = os.path.join(config['CHECKPOINT_DIR'], f"checkpoint_{round(timesteps[0], -3)}.ckpt")
                            save_checkpoint(trainstate_params, checkpoint_filename)
                            jax.debug.print("+++ checkpoint saved  {}", round(timesteps[0], -3))
                            jax.debug.print("+++ time taken        {}", time.time()-start)
                    evaluation()
                    
                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]
                    average_price = info["average_price"][info["returned_episode"]]
                    current_step = info["current_step"][info["returned_episode"]]
                    mkt_forced_quant = info["mkt_forced_quant"][info["returned_episode"]]
                    doom_quant = info["doom_quant"][info["returned_episode"]]
                    trade_duration = info["trade_duration"][info["returned_episode"]]
                    advantage_reward = info["advantage_reward"][info["returned_episode"]]
                    drift_reward = info["drift_reward"][info["returned_episode"]]
                    
                    if wandbOn:
                        wandb.log(
                            data={
                                "update_step": update_step,
                                "global_step": jnp.max(timesteps),
                                "episodic_return": jnp.mean(return_values),
                                "episodic_revenue": jnp.mean(revenues),
                                "quant_executed": jnp.mean(quant_executed),
                                "average_price": jnp.mean(average_price),
                                "current_step": jnp.mean(current_step),
                                "advantage_reward": jnp.mean(advantage_reward),
                                "drift_reward": jnp.mean(drift_reward),
                                "mkt_forced_quant": jnp.mean(mkt_forced_quant),
                                "doom_quant": jnp.mean(doom_quant),
                                "trade_duration": jnp.mean(trade_duration),
                                **trainstate_logs,
                            },
                            commit=True
                        )        
                    else:
                        print(
                            f"global step={jnp.max(timesteps):<11} | episodic return={jnp.mean(return_values):<20} | episodic revenue={jnp.mean(revenues):<20} | average_price={jnp.mean(average_price):<11}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, update_step + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_x,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")

    ppo_config = {
        "LR": 1e-4,
        "LR_COS_CYCLES": 8,
        "ENT_COEF": 0.,
        "NUM_ENVS": 256,
        "TOTAL_TIMESTEPS": 2e6,
        "NUM_MINIBATCHES": 4,
        "UPDATE_EPOCHS": 10,
        "NUM_STEPS": 10,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.999,
        "GAE_LAMBDA": 0.99,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 5,
        "ANNEAL_LR": 'cosine',
        "NORMALIZE_ENV": False,
        "RNN_TYPE": "S5",
        "HIDDEN_SIZE": 64,
        "ACTIVATION_FN": "relu",
        "ACTION_NOISE_COLOR": 2.,
        "RESET_ADAM_COUNT": True,
        "ADAM_B1": 0.99,
        "ADAM_B2": 0.99,
        "ADAM_EPS": 1e-5,
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1,
        "DEBUG": True,
        "TASKSIDE": "sell",
        "REWARD_LAMBDA": 1.,
        "ACTION_TYPE": "pure",
        "MAX_TASK_SIZE": 100,
        "TASK_SIZE": 100,
        "EPISODE_TIME": 60 * 5,
        "DATA_TYPE": "fixed_steps",
        "CONT_ACTIONS": False,
        "JOINT_ACTOR_CRITIC_NET": True,
        "ACTOR_STD": "state_dependent",
        "REDUCE_ACTION_SPACE_BY": 10,
        "ATFOLDER": "/homes/80/kang/AlphaTrade/AMZN_data/",
        "RESULTS_FILE": "training_runs/results_file_" + f"{timestamp}",
        "CHECKPOINT_DIR": "training_runs/checkpoints_" + f"{timestamp}",
        "NUM_EXPERTS": 3,
        "K": 2,
    }

    assert ppo_config["NUM_ENVS"] % ppo_config["NUM_MINIBATCHES"] == 0, "NUM_ENVS must be divisible by NUM_MINIBATCHES"
    assert ppo_config["NUM_ENVS"] > ppo_config["NUM_MINIBATCHES"], "NUM_ENVS must be a multiple of NUM_MINIBATCHES"

    ppo_config["NUM_UPDATES"] = (
        ppo_config["TOTAL_TIMESTEPS"] // ppo_config["NUM_STEPS"] // ppo_config["NUM_ENVS"]
    )
    ppo_config["MINIBATCH_SIZE"] = (
        ppo_config["NUM_ENVS"] // ppo_config["NUM_MINIBATCHES"]
    )

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            save_code=True,
        )
        import datetime; params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
    else:
        import datetime; params_file_name = f'params_file_{timestamp}'

    print(f"Results will be saved to {params_file_name}")

    rng = jax.random.PRNGKey(0)
    train_jit = jax.jit(make_train(ppo_config))
    start = time.time()
    out = train_jit(rng)
    print("Time: ", time.time() - start)

    train_state = out['runner_state'][0]
    params = train_state.params

    import datetime; params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'

    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"params saved")

    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"params restored")

    if wandbOn:
        run.finish()
