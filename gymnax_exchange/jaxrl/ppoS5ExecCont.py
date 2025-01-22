# from jax import config
# config.update("jax_enable_x64",True)

import os
import sys
import time
import socket

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
from jax import config
# config.update('jax_platform_name', 'cpu')
# print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

from AlphaTrade.gymnax_exchange.jaxen.old.exec_env_old import ExecutionEnv

#Code snippet to disable all jitting.
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.

import os; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
# %env XLA_PYTHON_CLIENT_PREALLOCATE=false

import datetime
wandbOn = True 
# wandbOn = False
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

class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.encoder_0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.encoder_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
    
        self.action_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        # self.action_decoder = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.action_decoder = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5))

        self.value_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.s5 = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=d_model,
            n_layers=n_layers,
            activation="half_glu1",
        )
        self.actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        # self.actor_logtstd = self.param("log_std", nn.initializers.constant(-0.7), (self.action_dim,))

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = self.encoder_0(obs)
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.leaky_relu(embedding)
        embedding = self.encoder_1(embedding)
        embedding = nn.LayerNorm()(embedding)
        # jax.debug.print("embedding shape {}", embedding.shape)
        embedding = nn.leaky_relu(embedding)

        hidden, embedding = self.s5(hidden, embedding, dones)
        embedding = nn.LayerNorm()(embedding)

        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_body_1(actor_mean)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        
        actor_mean = self.action_decoder(actor_mean)


        #pi = distrax.Categorical(logits=actor_mean)
        #Old version ^^
        # actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.actor_logtstd))
        #New version ^^

        critic = self.value_body_0(embedding)
        critic = nn.LayerNorm()(critic)
        critic = nn.leaky_relu(critic)
        critic = self.value_body_1(critic)
        critic = nn.LayerNorm()(critic)
        critic = nn.leaky_relu(critic)
        critic = self.value_decoder(critic)

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
    # env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["TASK_SIZE"],config["LAMBDA"],config["GAMMA"])
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["WINDOW_INDEX"],config["ACTION_TYPE"],config["TASK_SIZE"],config["REWARD_LAMBDA"])
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
        
        # old version 
        # network = ActorCriticS5(env.action_space(env_params).n, config=config)
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

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                # sampled = pi.sample(seed=_rng)
                # action = jnp.clip(sampled, -5, 5)
                action = pi.sample(seed=_rng)
                # jax.debug.print("sampled \n {}", sampled)
                log_prob = pi.log_prob(action)
                # jax.debug.print("log_prob \n {}", log_prob)
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
                        _, pi, value = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
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
            # grad_norm=jnp.mean(loss_info[1])
            train_state = update_state[0]
            # metric = (traj_batch.info,train_state.params,grad_norm)
            metric = (traj_batch.info,train_state.params)
            rng = update_state[-1]
            if config.get("DEBUG"):
                
                # # old version
                # def callback(info):
                #     return_values = info["returned_episode_returns"][info["returned_episode"]]
                #     timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                #     for t in range(len(timesteps)):
                #         print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                        

                # new version
                def callback(metric):
                    
                    info,trainstate_params=metric
                    # info,trainstate,grad_norm=metric
                    
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    
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
                    
                    slippage_rm = info["slippage_rm"][info["returned_episode"]]
                    price_drift_rm = info["price_drift_rm"][info["returned_episode"]]
                    price_adv_rm = info["price_adv_rm"][info["returned_episode"]]
                    vwap_rm = info["vwap_rm"][info["returned_episode"]]
                    
                    current_step = info["current_step"][info["returned_episode"]]
                    advantage_reward = info["advantage_reward"][info["returned_episode"]]
                    
                    for t in range(len(timesteps)):  
                        if wandbOn:
                            wandb.log(
                                {
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed":quant_executed[t],
                                    "average_price":average_price[t],
                                    "slippage_rm":slippage_rm[t],
                                    "price_adv_rm":price_adv_rm[t],
                                    "price_drift_rm":price_drift_rm[t],
                                    "vwap_rm":vwap_rm[t],
                                    "current_step":current_step[t],
                                    "advantage_reward":advantage_reward[t],
                                    # "grad_norm":grad_norm,
                                }
                            ) 
                            print(
                                f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}",\
                                file=open(config['RESULTS_FILE'],'a')
                            )                                 
                        else:
                            print(
                                f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}"
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
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    ppo_config = {
        "LR": 2.5e-3,
        "ENT_COEF": 0.1,
        "NUM_ENVS": 100, # 500,
        "TOTAL_TIMESTEPS": 1e8,
        "NUM_MINIBATCHES": 2,
        # "NUM_MINIBATCHES": 4,
        "UPDATE_EPOCHS": 30,
        "NUM_STEPS": 200, #455,
        "CLIP_EPS": 0.2,
        # "CLIP_EPS": 0.2,
        
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": False,
        
        "ACTOR_TYPE":"S5",
        
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": 0,
        # "WINDOW_INDEX": -1,
        "DEBUG": True,
        # "ATFOLDER": ".",
        "ATFOLDER": "/homes/80/kang/AlphaTrade/training_oneDay/",
        "TASKSIDE":'sell',
        "REWARD_LAMBDA":0,
        # "REWARD_LAMBDA":1,
        # "ACTION_TYPE":"pure",
        "ACTION_TYPE":"delta",
        "TASK_SIZE": 100, # 500,
        "RESULTS_FILE":"./results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR":"./checkpoints_"+f"{timestamp}",
        "DOCKER_CONTAINER_NAME": socket.gethostname(), # container_id
    }

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
            # settings=wandb.Settings(code_dir="~/AlphaTrade/gymnax_exchange/")
        )
        wandb.save("/homes/80/kang/AlphaTrade/gymnax_exchange/jaxen/base_env.py")
        wandb.save("/homes/80/kang/AlphaTrade/gymnax_exchange/jaxen/exec_env.py")
        wandb.save("/homes/80/kang/AlphaTrade/gymnax_exchange/jaxrl/ppoS5ExecCont.py")
        wandb.save("/homes/80/kang/AlphaTrade/gymnax_exchange/jaxrl/ppoRnnExecCont.py")
        import datetime;params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
        print(f"Results would be saved to {params_file_name}")
    else:
        import datetime;params_file_name = f'params_file_{timestamp}'
        print(f"Results would be saved to {params_file_name}")
        


    # device = jax.devices()[0]
    # device = jax.devices()[1]
    device = jax.devices()[-1]
    from rich.console import Console
    console = Console()
    console.print(f"Training on device: [bold cyan]{device}[/bold cyan]")
    
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
    


    import datetime;params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
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
        
        
        
