import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import chex
import flax
from typing import Sequence, Dict, Any
from flax.training.train_state import TrainState

from gymnax_exchange.jaxrl.actorCriticS5 import ActorCriticS5, EncoderS5, ActorDiscS5, CriticS5
from gymnax_exchange.jaxrl.router import TopKRouter

class ActorCriticMoE(nn.Module):
    config: Dict
    action_dim: Sequence[int]
    num_experts: int
    k: int
    
    def setup(self):
        self.router = TopKRouter(num_experts=self.num_experts, k=self.k)
        self.actor_critics = [ActorCriticS5(name=f'actorCritic{i}_freeze', action_dim=self.action_dim, config=self.config) for i in range(self.num_experts)]
    
    def __call__(self, x, *, key):
        chex.assert_rank(x, 2)

        # Initialize the router
        routing_info = self.router(x, key=key)

        hidden_states = []
        pis = []
        values = []

        for idx, actor_critic in enumerate(self.actor_critics):
            hidden_all = actor_critic.initialize_carry(x.shape[0], self.config["HIDDEN_SIZE"], actor_critic.config['n_layers'])
            hidden_all, pi, value = actor_critic(hidden_all, x)
            hidden_states.append(hidden_all)
            pis.append(pi)
            values.append(value)

        pis = jnp.stack(pis, axis=-1)  # Shape: (batch_size, action_dim, num_experts)
        values = jnp.stack(values, axis=-1)  # Shape: (batch_size, value_dim, num_experts)

        # Weighted outputs
        top_expert_weights_expanded = routing_info.top_expert_weights[..., None]  # Shape: (batch_size, k, 1)
        pi = jnp.sum(top_expert_weights_expanded * pis, axis=-1)  # Shape: (batch_size, action_dim)
        value = jnp.sum(top_expert_weights_expanded * values, axis=-1)  # Shape: (batch_size, value_dim)

        return pi, value, routing_info
    
    
def create_train_state(config: Dict, model: nn.Module, rng: jax.random.PRNGKey, learning_rate_fn) -> TrainState:
    params = model.init(rng, jnp.ones([1, config['obs_dim']]))['params']
    
    # Freeze parameters that contain 'freeze' in their names
    def filter_fn(path, _):
        if 'freeze' in path:
            return 'frozen'
        return 'trainable'

    partition_optimizers = {'trainable': optax.adam(learning_rate_fn), 'frozen': optax.set_to_zero()}
    param_partitions = flax.traverse_util.path_aware_map(filter_fn, params)
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

# Example of how to use the new ActorCriticMoE and create a train state
if __name__ == "__main__":
    config = {
        "HIDDEN_SIZE": 64,
        "obs_dim": 4,
        "MAX_GRAD_NORM": 1.0,
        "ADAM_B1": 0.9,
        "ADAM_B2": 0.999,
        "ADAM_EPS": 1e-8,
        "n_layers": (2, 2, 2),  # Example layer configuration for ActorCriticS5
    }
    
    
    action_dim = [2]
    num_experts = 3
    k = 2
    rng = jax.random.PRNGKey(0)
    
    model = ActorCriticMoE(config=config, action_dim=action_dim, num_experts=num_experts, k=k)
    learning_rate_fn = optax.linear_schedule(init_value=3e-4, end_value=1e-5, transition_steps=10000)
    state = create_train_state(config, model, rng, learning_rate_fn)
    
    print(state)
