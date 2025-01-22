# from jax import config
# config.update("jax_enable_x64",True)

import sys
import time

import chex
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Any, Dict, NamedTuple, Sequence
import distrax
from gymnax.environments import spaces

sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
#Code snippet to disable all jitting.
from jax import config

from AlphaTrade.gymnax_exchange.jaxen.old.exec_env_old import ExecutionEnv

config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.

import datetime
# wandbOn = True
wandbOn = False
if wandbOn:
    import wandb
    

from purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from purejaxrl.experimental.s5.wrappers import FlattenObservationWrapper, LogWrapper


def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")


d_model = 256
ssm_size = 256
C_init = "lecun_normal"
discretization="zoh"
dt_min=0.001
dt_max=0.1
n_layers = 4
conj_sym=True
clip_eigs=False
bidirectional=False

blocks = 1
block_size = int(ssm_size / blocks)

Lambda, _, B, V, B_orig = make_DPLR_HiPPO(ssm_size)

block_size = block_size // 2
ssm_size = ssm_size // 2

Lambda = Lambda[:block_size]
V = V[:, :block_size]

Vinv = V.conj().T


ssm_init_fn = init_S5SSM(H=d_model,
                            P=ssm_size,
                            Lambda_re_init=Lambda.real,
                            Lambda_im_init=Lambda.imag,
                            V=V,
                            Vinv=Vinv,
                            C_init=C_init,
                            discretization=discretization,
                            dt_min=dt_min,
                            dt_max=dt_max,
                            conj_sym=conj_sym,
                            clip_eigs=clip_eigs,
                            bidirectional=bidirectional)
from dataclasses import field
class ActorCriticS5(nn.Module):
    action_dim: Sequence[int] = (11, 11)  # Two dimensions, each with 11 choices
    config: Dict = field(default_factory=dict)  # Use default_factory for mutable types

    def setup(self):
        self.encoder_0 = nn.Dense(128, kernel_init=orthogonal(1.41421356237), bias_init=constant(0.0))
        self.encoder_1 = nn.Dense(256, kernel_init=orthogonal(1.41421356237), bias_init=constant(0.0))

        self.action_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_decoder_0 = nn.Dense(11, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.action_decoder_1 = nn.Dense(11, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

        self.value_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.s5 = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=128,
            n_layers=6,
            activation="half_glu1",
        )
    
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = self.encoder_0(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = self.encoder_1(embedding)
        embedding = nn.leaky_relu(embedding)

        # jax.debug.print("+++hidden{} embedding{} dones{}",type(hidden), type(embedding), type(dones))
        jax.debug.print("+++hidden{} embedding{}",hidden, embedding)
        hidden, embedding = self.s5(hidden, embedding, dones)
        jax.debug.print("---hidden{} embedding{}",hidden, embedding)
        # jax.debug.print("---hidden{} embedding{} ",type(hidden), type(embedding))

        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_body_1(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)

        actor_logits_0 = self.action_decoder_0(actor_mean)
        actor_logits_1 = self.action_decoder_1(actor_mean)

        pi_0 = distrax.Categorical(logits=actor_logits_0)
        pi_1 = distrax.Categorical(logits=actor_logits_1)

        critic = self.value_body_0(embedding)
        critic = nn.leaky_relu(critic)
        critic = self.value_body_1(critic)
        critic = nn.leaky_relu(critic)
        critic = self.value_decoder(critic)

        return hidden, (pi_0, pi_1), jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # # old version
    # env, env_params = gymnax.make(config["ENV_NAME"])
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)
    # # old version
    
    # new version
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["TASK_SIZE"],config["LAMBDA"],config["GAMMA"])
    env_params = env.default_params
    env = LogWrapper(env)
    
    # #FIXME : Uncomment normalisation.
    # if config["NORMALIZE_ENV"]:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config["GAMMA"])
    # new version

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        
        # network = ActorCriticS5(env.action_space(env_params).n, config=config)
        # old version 
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=config)
        # new version
        
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # # SELECT ACTION
                # ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                # hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                # action = jnp.clip(pi.sample(seed=_rng), -5, 5)
                # log_prob = pi.log_prob(action)
                # value, action, log_prob = (
                #     value.squeeze(0),
                #     action.squeeze(0),
                #     log_prob.squeeze(0),
                # )
                
                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, (pi_0, pi_1), value = network.apply(train_state.params, hstate, ac_in)
                # Sampling from both distributions and converting index to action
                action_0 = pi_0.sample(seed=_rng) - 5  # Convert index to action for dimension 0
                action_1 = pi_1.sample(seed=_rng) - 5  # Convert index to action for dimension 1
                # Combine actions
                action = jnp.stack([action_0, action_1], axis=1)  # Assume axis=1 is the correct axis for your setup
                # Compute log probabilities for both actions
                log_prob_0 = pi_0.log_prob(action_0 + 5)  # Convert action back to index for log_prob
                log_prob_1 = pi_1.log_prob(action_1 + 5)
                # Combine log probabilities
                log_prob = log_prob_0 + log_prob_1
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                
                
                

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
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
            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward 
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, (pi1, pi2), value = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
                        )
                        
                        log_prob1 = pi1.log_prob(traj_batch.action1)  # Assuming traj_batch.action1 is for pi1
                        log_prob2 = pi2.log_prob(traj_batch.action2)  # Assuming traj_batch.action2 is for pi2


                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS FOR PI1
                        ratio1 = jnp.exp(log_prob1 - traj_batch.log_prob1)  # Assuming traj_batch.log_prob1 is for pi1
                        gae1 = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1_1 = ratio1 * gae1
                        loss_actor1_2 = jnp.clip(ratio1, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae1
                        loss_actor1 = -jnp.minimum(loss_actor1_1, loss_actor1_2)
                        loss_actor1 = loss_actor1.mean()
                        entropy1 = pi1.entropy().mean()

                        # CALCULATE ACTOR LOSS FOR PI2
                        ratio2 = jnp.exp(log_prob2 - traj_batch.log_prob2)  # Assuming traj_batch.log_prob2 is for pi2
                        gae2 = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor2_1 = ratio2 * gae2
                        loss_actor2_2 = jnp.clip(ratio2, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae2
                        loss_actor2 = -jnp.minimum(loss_actor2_1, loss_actor2_2)
                        loss_actor2 = loss_actor2.mean()
                        entropy2 = pi2.entropy().mean()

                        # COMBINE THE ACTOR LOSSES AND ENTROPIES
                        loss_actor_combined = loss_actor1 + loss_actor2
                        entropy_combined = entropy1 + entropy2

                        # CALCULATE TOTAL LOSS
                        total_loss = (
                            loss_actor_combined
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy_combined
                        )

                        return total_loss, (value_loss, loss_actor_combined, entropy_combined)

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

            init_hstate = initial_hstate # TBH
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
                
                # # old version
                # def callback(info):
                #     return_values = info["returned_episode_returns"][info["returned_episode"]]
                #     timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                #     for t in range(len(timesteps)):
                #         print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                        

                # new version
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    
                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]
                    average_price = info["average_price"][info["returned_episode"]]
                    slippage = info["slippage"][info["returned_episode"]]
                    price_drift = info["price_drift"][info["returned_episode"]]
                    current_step = info["current_step"][info["returned_episode"]]
                    step_reward = info["step_reward"][info["returned_episode"]]
                    drift_reward = info["drift_reward"][info["returned_episode"]]
                    advantage_reward = info["advantage_reward"][info["returned_episode"]]
                    
                    # if len(timesteps) >0:
                    #     if any(timesteps % int(1e5) == 0):  # +1 since global_step is 0-indexed
                    #         checkpoint_filename = f"checkpoint_{round(timesteps[0],-5)}.ckpt"
                    #         save_checkpoint(trainstate, checkpoint_filename)  # Assuming runner_state[0] contains your model's state
                    
                    for t in range(len(timesteps)):  
                        if wandbOn:
                            wandb.log(
                                {
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed":quant_executed[t],
                                    "average_price":average_price[t],
                                    "slippage":slippage[t],
                                    "price_drift":price_drift[t],
                                    "current_step":current_step[t],
                                    "step_reward":step_reward[t],
                                    "drift_reward":drift_reward[t],
                                    "advantage_reward":advantage_reward[t],
                                    # "grad_norm":grad_norm,
                                }
                            ) 
                            
                            print(
                                f"global step={timesteps[t]:<11} | episodic return={return_values[t]:<11} | episodic revenue={revenues[t]:<11} | average_price={average_price[t]:<11}",\
                                file=open(config['RESULTS_FILE'],'a')
                            )       
                        else:
                            print(
                                f"global step={timesteps[t]:<11} | episodic return={return_values[t]:<11} | episodic revenue={revenues[t]:<11} | average_price={average_price[t]:<11}"
                            )
                            # print(grad_norm)     
                # new version
                
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
    try:
        ATFolder = sys.argv[1] 
    except:
        #ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = '/homes/80/kang/AlphaTrade'
        ATFolder = '/home/duser/AlphaTrade/training_oneDay'
    print("AlphaTrade folder:",ATFolder)

    ppo_config = {
        "LR": 2.5e-4,
        "ENT_COEF": 0.1,
        "NUM_ENVS": 1000,
        "TOTAL_TIMESTEPS": 3e7,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 5,
        "NUM_STEPS": 5,
        "CLIP_EPS": 0.2,
        
        # "LR": 2.5e-6,
        # "NUM_ENVS": 1,
        # "NUM_STEPS": 1,
        # "NUM_MINIBATCHES": 1,
        # "NUM_ENVS": 1000,
        # "NUM_STEPS": 10,
        # "NUM_MINIBATCHES": 4,
        # "TOTAL_TIMESTEPS": 1e7,
        # "UPDATE_EPOCHS": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        # "CLIP_EPS": 0.2,
        # "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": True,
        
        "ENV_NAME": "alphatradeExec-v0",
        "ENV_LENGTH": "oneWindow",
        "DEBUG": True,
        "ATFOLDER": ATFolder,
        "TASKSIDE":'sell',
        "LAMBDA":0.1,
        "GAMMA":10.0,
        "TASK_SIZE":500,
        "RESULTS_FILE":"/home/duser/AlphaTrade/results_file_"+f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
    }

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_ParamSearch_Small",
            config=ppo_config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
        )
        import datetime;params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        print(f"Results would be saved to {params_file_name}")
    else:
        import datetime;params_file_name = f'params_file_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        print(f"Results would be saved to {params_file_name}")
        



    device = jax.devices()[0]
    rng = jax.device_put(jax.random.PRNGKey(0), device)
    train_jit = jax.jit(make_train(ppo_config), device=device)
    out = train_jit(rng)

    # if jax.device_count() == 1:
    #     # +++++ Single GPU +++++
    #     rng = jax.random.PRNGKey(0)
    #     # rng = jax.random.PRNGKey(30)
    #     train_jit = jax.jit(make_train(ppo_config))
    #     start=time.time()
    #     out = train_jit(rng)
    #     print("Time: ", time.time()-start)
    #     # +++++ Single GPU +++++
    # else:
    #     # +++++ Multiple GPUs +++++
    #     num_devices = int(jax.device_count())
    #     rng = jax.random.PRNGKey(30)
    #     rngs = jax.random.split(rng, num_devices)
    #     train_fn = lambda rng: make_train(ppo_config)(rng)
    #     start=time.time()
    #     out = jax.pmap(train_fn)(rngs)
    #     print("Time: ", time.time()-start)
    #     # +++++ Multiple GPUs +++++
    
    

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
        