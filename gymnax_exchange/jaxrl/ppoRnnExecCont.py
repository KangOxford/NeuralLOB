# from jax import config
# config.update("jax_enable_x64",True)
import dataclasses
import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import flax.linen as nn
# import numpy as np
import optax
import time
# from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxrl.actorCritic import ActorCriticRNN, ScannedRNN
import os
import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)



wandbOn = True # False
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
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )

        if config['JOINT_ACTOR_CRITIC_NET']:
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        else:
            init_hstate = (
                ScannedRNN.initialize_carry(config["NUM_ENVS"], 128),
                ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
            )

        network_params = network.init(_rng, init_hstate, init_x)
        
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
        
        # jax.debug.breakpoint()
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        if config['JOINT_ACTOR_CRITIC_NET']:
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"])
        else:
            init_hstate = (
                ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"]),
                ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"])
            )

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
            def _env_step(runner_state, action_noise):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                if config["CONT_ACTIONS"]:
                    # use pre-computed colored noise sample instead of sampling here
                    a_mean = pi._loc
                    a_std = pi._scale_diag
                    action = action_noise * a_std + a_mean
                    # jax.debug.print('a_mean {}, a_std {}, action_noise{}, action {}', a_mean, a_std, action_noise, action)
                else:
                    action = pi.sample(seed=_rng)

                log_prob = pi.log_prob(action)

                # print('action {}, log_prob {}', action.shape, log_prob.shape)
                # jax.debug.print('action {}, log_prob {}', action, log_prob)

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
            # generate colored noise sequence for correlated actions (lenght of NUM_STEPS)
            rng, rng_ = jax.random.split(runner_state[-2])
            # include new rng in runner_state
            runner_state = runner_state[:-2] + (rng,) + runner_state[-1:]
            # args: exponent, size, rng, fmin.  transpose to have first dimension correlated
            col_noise = cnoise.powerlaw_psd_gaussian(config["ACTION_NOISE_COLOR"], (network.action_dim, len(runner_state[2]), config["NUM_STEPS"]), _rng, 0.).T
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state[:-1], col_noise, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
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
                    train_state, _ = train_state
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _dead_neuron_ratio(activations):
                        # jax.tree_util.tree_map(lambda x: jax.debug.print('{}', x.shape), activations)
                        # num_activations = jax.tree_util.tree_reduce(
                        #     lambda x, y: x + y,
                        #     jax.tree_util.tree_map(jnp.size, activations)
                        # )
                        num_activations = len(jax.tree_util.tree_leaves(activations))
                        # num_dead = jax.tree_util.tree_reduce(
                        #     lambda x, y: x + y,
                        #     jax.tree_util.tree_map(lambda x: (x<=0).sum(), activations)
                        # )
                        num_dead = jax.tree_util.tree_reduce(
                            lambda x, y: x + y,
                            jax.tree_util.tree_map(lambda x: (x < 0).all().astype(int), activations)
                        )
                        dead_ratio = num_dead / num_activations
                        # jax.debug.print('size: {}, num_dead {}, dead_ratio: {}', num_activations, num_dead, dead_ratio)
                        # jax.debug.breakpoint()
                        return dead_ratio
                    
                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        def debug_log(metric):
                            (update_step, info, dead_ratio, obs_norm, action_mean, action_std) = metric
                            data = {
                                "update_step": update_step,
                                "global_step": info["timestep"][info["returned_episode"]] * config["NUM_ENVS"],
                                "dead_neurons": dead_ratio,
                                "obs_norm": obs_norm,
                            }

                            # mean over batch dimension --> shape (4,)
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
                        
                        # RERUN NETWORK
                        filter_neurons = lambda mdl, method_name: isinstance(mdl, nn.LayerNorm)

                        if config['JOINT_ACTOR_CRITIC_NET']:
                            init_hstate = init_hstate[0]
                        else:
                            init_hstate = (init_hstate[0][0], init_hstate[1][0])
                        (_, pi, value), network_state = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done),
                            capture_intermediates=filter_neurons, mutable=["intermediates"]
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        activations = network_state["intermediates"]
                        dead_ratio = _dead_neuron_ratio(activations)
                        # norm of trajectory batch observation
                        obs_norm = jnp.sqrt((traj_batch.obs**2).sum(axis=-1)).mean()
                        # jax.debug.print('_scale_diag {}', pi._scale_diag.squeeze()[-1].shape)
                        # jax.debug.print('obs_norm: {}', obs_norm)
                        
                        # action_mean = (
                        #     pi._loc.squeeze()[-1] if config["CONT_ACTIONS"]
                        #     else pi.distribution.probs @ jnp.arange(config["MAX_TASK_SIZE"] + 1))
                        
                        # action_std = (
                        #     pi._scale_diag.squeeze()[-1] if config["CONT_ACTIONS"]
                        #     else jnp.sqrt(
                        #         pi.distribution.probs @ (jnp.arange(config["MAX_TASK_SIZE"] + 1) - action_mean[-1, -1])**2)
                        #     )

                        # continuous actions (mean variance mutlivar. normal):
                        if config["CONT_ACTIONS"]:
                            # shape: (bsz, action_dim)
                            action_mean = pi._loc.squeeze()[-1]
                            action_std = pi._scale_diag.squeeze()[-1]
                        # discrete actions:
                        else:

                            action_mean = pi.distribution.probs[-1] @ jnp.arange(config["MAX_TASK_SIZE"] + 1)
                            # print('action_mean: ', action_mean.shape)

                            # (seq_len, bsz, num_actions, action_dim) @ ((action_dim, num_actions) - (num_actions,))
                            action_std = jnp.sqrt(
                                jnp.dot(
                                    pi.distribution.probs[-1],
                                    (
                                        jnp.tile(
                                            jnp.arange(config["MAX_TASK_SIZE"] + 1),
                                            pi.event_shape + (1,),
                                        ).T - action_mean[-1]
                                    ) ** 2
                                )
                            )
                            action_std = jnp.diagonal(action_std.T)

                            # TODO: benchmark this alternative way to calc std
                            #       reduced number of matmul operations but for loop over num_actions
                            # action_std = jnp.sqrt(jnp.array([
                            #     # (bsz, action_dim) @ (action_dim)  e.g. (128, 501,) @ (501,)
                            #     pi.distribution.probs[-1, :, i] @ (jnp.arange(config["MAX_TASK_SIZE"] + 1) - action_mean[-1, i]) ** 2
                            #     for i in range(pi.event_shape[0])
                            # ])).T  # (bsz, num_actions)
                        
                            # print('action_std: ', action_std.shape)
                            # jax.debug.print('std arrays equal {} {} {}', jnp.array_equal(action_std[-1], action_std_1[-1]), action_std[-1], action_std_1[-1])

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
                        # jax.debug.print('obs_norm: {}', obs_norm)
                        # jax.debug.breakpoint()

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

                        # total_loss = (
                        #     loss_actor
                        #     + config["VF_COEF"] * value_loss
                        #     - config["ENT_COEF"] * entropy
                        # )
                        # return total_loss, (value_loss, loss_actor, entropy)
                        losses = (
                            config["VF_COEF"] * value_loss,
                            loss_actor,
                            - config["ENT_COEF"] * entropy
                        )
                        losses = (losses[0] + losses[1] + losses[2], *losses)
                        return losses, (value_loss, loss_actor, entropy)

                    # grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    # total_loss, grads = grad_fn(
                    #     train_state.params, init_hstate, traj_batch, advantages, targets
                    # )

                    # NOTE: only for debugging. might have negative impact on performance
                    # to do forward pass and jacobian separately
                    total_loss = _loss_fn(train_state.params, init_hstate, traj_batch, advantages, targets)[0][0]
                    jac = jax.jacrev(_loss_fn, has_aux=True)(train_state.params, init_hstate, traj_batch, advantages, targets)
                    # (jac, (value_loss, loss_actor, entropy)): [0][0] for gradient
                    grads = jac[0][0]

                    # HERE TODO: figure out why this isn't working

                    # jax.debug.print("grads: {}", grads['params']['log_std'])
                    # jax.debug.print("grads: {}", grads)
                    # print(jax.tree_util.tree_structure(grads))
                    train_state = train_state.apply_gradients(grads=grads)
                    # make sure the action std doesn't grow too large
                    # train_state.params = train_state.params.copy({
                    #     'log_std': jnp.min(
                    #         train_state.params["log_std"],
                    #         -1.6 * jnp.ones_like(train_state.params["log_std"]),
                    #     )
                    # })

                    # grad_norm = optax.global_norm(grads)
                    # jax.debug.print("grad_norm: {}", grad_norm)
                    # jax.debug.print("grads: {}", grads)

                    # TODO: calculate gradient norm for each loss component
                    #       handle this in the scan function (4 instead of 1 gradient norm)
                    grad_parts_norm = jnp.array([
                        optax.global_norm(g) for g in jac[0]
                    ])

                    # return (train_state, grad_norm), total_loss
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
                # jax.debug.print('traj_batch {}', traj_batch.obs.shape)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                # jax.debug.print('shuffled_batch {}', shuffled_batch[1].obs.shape)

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
                # jax.debug.print('minibatches {}', minibatches[1].obs.shape)

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

            if config["JOINT_ACTOR_CRITIC_NET"]:
                init_hstate = initial_hstate[None, :]  # TBH
            else:
                init_hstate = (
                    initial_hstate[0][None, :],
                    initial_hstate[1][None, :]
                )

            if config["RESET_ADAM_COUNT"]:
                train_state = reset_adam(train_state)

            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
                jnp.zeros((4,))  # grad_norm
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
            # jax.debug.breakpoint()
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
                        # Inside your loop or function where you save the checkpoint
                        if any(timesteps % int(1e3) == 0) and len(timesteps) > 0:  # +1 since global_step is 0-indexed
                            start = time.time()
                            jax.debug.print(">>> checkpoint saving {}",round(timesteps[0], -3))
                            # Save the checkpoint to the specific directory
                            checkpoint_filename = os.path.join(config['CHECKPOINT_DIR'], f"checkpoint_{round(timesteps[0], -3)}.ckpt")
                            save_checkpoint(trainstate_params, checkpoint_filename)  # Assuming trainstate_params contains your model's state
                            jax.debug.print("+++ checkpoint saved  {}",round(timesteps[0], -3))
                            jax.debug.print("+++ time taken        {}",time.time()-start)        
                    evaluation()
                    
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
                    # advantage_reward = info["advantage_reward"][info["returned_episode"]]
                    
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
                    # NOTE: only log every 100th timestep
                    for t in range(0, len(timesteps), 100):
                    # for t in range(len(timesteps)):
                        if wandbOn:
                            wandb.log(
                                data={
                                    "update_step": update_step,
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed": quant_executed[t],
                                    "average_price": average_price[t],
                                    # "slippage_rm":slippage_rm[t],
                                    # "price_adv_rm":price_adv_rm[t],
                                    # "price_drift_rm":price_drift_rm[t],
                                    # "vwap_rm":vwap_rm[t],
                                    "current_step": current_step[t],
                                    # "advantage_reward":advantage_reward[t],
                                    "mkt_forced_quant": mkt_forced_quant[t],
                                    "doom_quant": doom_quant[t],
                                    "trade_duration": trade_duration[t],
                                    **trainstate_logs,
                                    # "learning_rate": trainstate_info['learning_rate'],
                                    # "grad_norm": trainstate_info['grad_norm'],
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


                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, update_step + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")

    ppo_config = {
        "LR": 1e-5, # 1e-4, 5e-4, #5e-5, #1e-4,#2.5e-5,
        "LR_COS_CYCLES": 8,  # only relevant if ANNEAL_LR == "cosine"
        "ENT_COEF": 0.001, # 0., 0.001, 0, 0.1, 0.01, 0.001
        "NUM_ENVS": 1024, #1024, #128, #64, 1000,
        "TOTAL_TIMESTEPS": 5e7,  # 1e8, 5e7, # 50MIL for single data window convergence #,1e8,  # 6.9h
        "NUM_MINIBATCHES": 8, #8, 4, 2,
        "UPDATE_EPOCHS": 10, #30, 5,
        "NUM_STEPS": 20, #20, 512, 500,
        "CLIP_EPS": 0.2,  # TODO: should we change this to a different value? 
        
        "GAMMA": 0.999,
        "GAE_LAMBDA": 0.99, #0.95,
        "VF_COEF": 0.1, #1., 0.01, 0.001, 1.0, 0.5,
        "MAX_GRAD_NORM": 10, # 0.5, 2.0,
        "ANNEAL_LR": 'cosine', # 'linear', 'cosine', False
        "NORMALIZE_ENV": False,  # only norms observations (not reward)
        
        "ACTOR_TYPE": "RNN",
        "HIDDEN_SIZE": 256,  # 128
        "ACTION_NOISE_COLOR": 2.,  # 2  # only relevant if CONT_ACTIONS == True

        "RESET_ADAM_COUNT": True,  # resets Adam's t (count) every update
        "ADAM_B1": 0.99,
        "ADAM_B2": 0.99,
        "ADAM_EPS": 1e-5,  # 1e-4, 1e-6
        
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1, # 2 fix random episode #-1,
        "DEBUG": True,
        
        "TASKSIDE": "sell", # "random", "buy", "sell"
        "REWARD_LAMBDA": 1., #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "MAX_TASK_SIZE": 500,
        "TASK_SIZE": 500, # 500,
        "EPISODE_TIME": 60 * 1, # 60 * 1 --> 1 minute
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "CONT_ACTIONS": False,  # True
        "JOINT_ACTOR_CRITIC_NET": False,  # True
        "ACTOR_STD": "state_dependent",  # 'state_dependent', 'param', 'fixed'
      
        # "ATFOLDER": "./training_oneDay/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        "ATFOLDER": "./training_oneMonth/", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        "RESULTS_FILE": "training_runs/results_file_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR": "training_runs/checkpoints_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/checkpoints_"+f"{timestamp}",
    }

    assert ppo_config["NUM_ENVS"] % ppo_config["NUM_MINIBATCHES"] == 0, "NUM_ENVS must be divisible by NUM_MINIBATCHES"
    assert ppo_config["NUM_ENVS"] > ppo_config["NUM_MINIBATCHES"], "NUM_ENVS must be a multiple of NUM_MINIBATCHES"

    # CAVE: DEBUG VALUES:
    # ppo_config['TOTAL_TIMESTEPS'] = 1e6
    # ppo_config['NUM_ENVS'] = 4
    # ppo_config['NUM_STEPS'] = 100

    ppo_config["NUM_UPDATES"] = (
        ppo_config["TOTAL_TIMESTEPS"] // ppo_config["NUM_STEPS"] // ppo_config["NUM_ENVS"]
    )
    ppo_config["MINIBATCH_SIZE"] = (
        #ppo_config["NUM_ENVS"] * ppo_config["NUM_STEPS"] // ppo_config["NUM_MINIBATCHES"]
        # sequences are kept together as one sample 
        ppo_config["NUM_ENVS"] // ppo_config["NUM_MINIBATCHES"]
    )

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
        )
        import datetime;params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
    else:
        import datetime;params_file_name = f'params_file_{timestamp}'

    print(f"Results will be saved to {params_file_name}")
    
    # +++++ Single GPU +++++
    rng = jax.random.PRNGKey(0)
    # rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(ppo_config))
    start=time.time()
    out = train_jit(rng)
    print("Time: ", time.time() - start)
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
        print(f"params saved")

    # Load the params from the file using flax.serialization.from_bytes
    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"params restored")
        
    # jax.debug.breakpoint()
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, restored_params))
    # print(">>>")
    # '''

    if wandbOn:
        run.finish()
