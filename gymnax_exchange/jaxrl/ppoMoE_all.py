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
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
# from gymnax_exchange.jaxrl.router import TopKRouter
import chex
import flax

from absl import logging
import chex
import flax.linen as nn
# import gin
import jax
from gymnax_exchange.jaxrl import types


from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) 


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

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)




# @gin.configurable
class TopKRouter(nn.Module):
  """A simple router that linearly projects assignments."""

  k: int
  num_experts: int | None = None
  noise_std: float = 1.0

  def setup(self):
    logging.info("Creating a %s", self.__class__.__name__)

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      *,
      num_experts: int | None = None,
      k: int | None = None,
      **kwargs
  ) -> types.RouterReturn:
    chex.assert_rank(x, 2)

    num_experts = nn.merge_param("num_experts", num_experts, self.num_experts)
    k = nn.merge_param("k", k, self.k)
    sequence_length = x.shape[0]

    x = nn.Dense(num_experts, use_bias=False)(x)
    chex.assert_shape(x, (sequence_length, num_experts))

    probs = jax.nn.softmax(x, axis=-1)

    top_expert_weights, top_experts = jax.lax.top_k(probs, k=k)

    return types.RouterReturn(
        output=x,
        probabilities=probs,
        top_expert_weights=top_expert_weights,
        top_experts=top_experts,
    )

class ActorCriticMoE(nn.Module):
    action_dim: Sequence[int]
    num_experts: int 
    k: int
    config: Dict
    
    def setup(self):
        self.num_experts = self.config['num_experts']
        self.k = self.config['top_k']
        self.router = TopKRouter(num_experts=self.num_experts, k=self.k)
        self.actor_critics = [ActorCriticRNN(name=f'actorCritic{i}_freeze', action_dim=self.action_dim, config=self.config) for i in range(self.num_experts)]
    
    def __call__(self, hiddens, x, *, key):
        chex.assert_rank(x, 2)
        assert self.config['JOINT_ACTOR_CRITIC_NET']
        
        # obs, dones = x
        
        # Get routing information
        routing_info = self.router(x, key=key)
        
        # Get the index of the top expert
        top_expert_idx = routing_info.top_experts[0]  # Assuming k=1, we take the first (and only) top expert
        
        # Use only the selected expert
        selected_actor_critic = self.actor_critics[top_expert_idx]
        hidden_state, pi, value = selected_actor_critic(hiddens[top_expert_idx], x)
        
        # Update only the hidden state for the selected expert
        new_hiddens = hiddens.at[top_expert_idx].set(hidden_state)
        
        return new_hiddens, pi, value
    

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
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticMoE(env.action_space(env_params).n, num_experts=config['num_experts'], k=config['top_k'], config=config)
        
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        
        # init_hstates = [ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        #                 for _ in range(config['num_experts'])]
        # # TODO do I need to remvoe the for loop and use vmap?
        indices = jnp.arange(config['num_experts'])
        initialize_vmap = jax.vmap(ScannedRNN.initialize_carry, in_axes=(None, None))
        init_hstates = initialize_vmap(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstates, init_x)
                
        # │   223 │   │   # # TODO do I need to remvoe the for loop and use vmap?                            │
        # │   224 │   │   indices = jnp.arange(config['num_experts'])                                        │
        # │   225 │   │   initialize_vmap = jax.vmap(ScannedRNN.initialize_carry, in_axes=(None, None))      │
        # │ ❱ 226 │   │   init_hstates = initialize_vmap(config["NUM_ENVS"], 128)                            │
        # │   227 │   │   network_params = network.init(_rng, init_hstates, init_x)                          │
        # │   228                                                                                            │
        # │   229                                                                                            │
        # ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
        # ValueError: vmap must have at least one non-None value in in_axes
        # I0000 00:00:1720172384.958425  152546 tfrt_cpu_pjrt_client.cc:352] TfrtCpuClient destroyed.
                
        
        # # Load pretrained expert model params
        # expert_params = []
        # for i in range(config['num_experts']):
        #     params_file = f"expert_{i}_params.pkl"
        #     with open(params_file, 'rb') as f:
        #         expert_param = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        #     expert_params.append(expert_param)
        def load_expert_params(i):
            params_file = f"expert_{i}_params.pkl"
            with open(params_file, 'rb') as f:
                return flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        vmap_load_expert_params = jax.vmap(load_expert_params)
        # Generate an array of expert indices
        expert_indices = jnp.arange(config['num_experts'])
        # Use vmap to load all expert parameters in parallel
        expert_params = vmap_load_expert_params(expert_indices)
        
        
        
        # # Replace the model's parameters with the pretrained expert parameters
        # network_params = network_params.unfreeze()
        # for i in range(config['num_experts']):
        #     network_params['params'][f'actorCritic{i}_freeze'] = flax.core.frozen_dict.freeze(expert_params[i]['params'])
        # # network_params = flax.core.frozen_dict.freeze(network_params)
        network_params = network_params.unfreeze()
        def assign_params(i, network_params, expert_params):
            # Create the key dynamically
            key = f'actorCritic{i}_freeze'
            # Perform the assignment for this specific expert
            return network_params.at['params'].set(key, flax.core.frozen_dict.freeze(expert_params[i]['params']))
        indices = jnp.arange(config['num_experts'])
        assign_vmap = jax.vmap(assign_params, in_axes=(0, None, None))
        network_params = assign_vmap(indices, network_params, expert_params)

        
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
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
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
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

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
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
        "DEBUG": True,
        
        "num_experts":2,
        'top_k':1,
    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)