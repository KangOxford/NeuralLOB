import functools
from typing import Callable, Optional, Sequence, NamedTuple, Any, Dict
import jax
import jax.numpy as jnp
from jax._src import dtypes
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax


# custom weight initializer
def biased_constant(
        value: jax.Array,
        first_value: jax.Array,
        dtype = jnp.float_
    ):
    def init(key, shape, dtype = dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        a = jnp.full(shape, value, dtype=dtype)
        a = a.at[..., 0].set(first_value)
        return a
    return init


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    activation_fn_str: str = "relu"

    def setup(self):
        if self.activation_fn_str == "relu":
            self.act_fn = nn.relu
        elif self.activation_fn_str == "leaky_relu":
            self.act_fn = nn.leaky_relu
        elif self.activation_fn_str == "swish":
            self.act_fn = nn.swish
        elif self.activation_fn_str == "tanh":
            self.act_fn = nn.tanh
        elif self.activation_fn_str == "sigmoid":
            self.act_fn = nn.sigmoid
        else:
            raise ValueError(f"Invalid activation_fn_str: {self.activation_fn_str}")

        self.encoder = Encoder(name='embedding', config=self.config, act_fn=self.act_fn)
        self.critic = Critic(self.config, act_fn=self.act_fn)
        if self.config['CONT_ACTIONS']:
            self.actor = ActorCont(self.action_dim, self.config, act_fn=self.act_fn)
        else:
            self.actor = ActorDisc(self.action_dim, self.config, act_fn=self.act_fn)
        if not self.config['JOINT_ACTOR_CRITIC_NET']:
            self.actor_embedding = Encoder(name='actor_embedding', config=self.config, act_fn=self.act_fn)

    def __call__(self, hidden, x):
        obs, dones = x
        if not self.config['JOINT_ACTOR_CRITIC_NET']:
            hidden, hidden_actor = hidden

        # jax.debug.print('obs {}', obs)
        # jax.debug.print('hidden {}', hidden)

        hidden, embedding = self.encoder(hidden, obs, dones)
        self.sow("intermediates", "embedding_rnn", embedding)

        ## ACTOR
        if self.config['JOINT_ACTOR_CRITIC_NET']:
            actor_embedding = embedding
        else:
            hidden_actor, actor_embedding = self.actor_embedding(hidden_actor, obs, dones)
            # manually save intermediate values as the layer norm inside the module isn't recorded automatically
            self.sow("intermediates", "actor_embedding_rnn", actor_embedding)

        pi = self.actor(actor_embedding)

        ## CRITIC
        value = self.critic(embedding)

        if not self.config['JOINT_ACTOR_CRITIC_NET']:
            hidden = (hidden, hidden_actor)

        # jax.debug.print('pi {}', pi.sample(seed=1))

        return hidden, pi, value


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
            resets[:, jnp.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(gate_fn=lambda x: nn.sigmoid(nn.LayerNorm()(x)))(rnn_state, ins)
        # new_rnn_state, y = nn.LayerNorm()(new_rnn_state), nn.LayerNorm()(y)
        new_rnn_state, y = nn.GRUCell(gate_fn=lambda x: nn.sigmoid(nn.LayerNorm()(x)))(new_rnn_state, y)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )

class Encoder(nn.Module):
    name: str = "encoder"
    config: Dict
    act_fn: Callable

    def setup(self):
        self.dense_0 = nn.Dense(
            self.config["HIDDEN_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )
        self.ln_0 = nn.LayerNorm()
        self.rnn = ScannedRNN(name=self.name + "_rnn")

    def __call__(self, hidden, obs, dones):
        # jax.debug.print('encoder obs: {}, hidden {}', obs, hidden)
        x = self.dense_0(obs)
        x = self.ln_0(x)
        # jax.debug.print('bef {}', x)
        x = self.act_fn(x)
        # jax.debug.print('aft {}', x)
        rnn_in = (x, dones)
        hidden, embedding = self.rnn(hidden, rnn_in)
        print('hidden', hidden.shape, 'embedding', embedding.shape)
        return hidden, embedding
    

class ActorCont(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    act_fn: Callable

    def setup(self):
        self.dense_0 = nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.ln_0 = nn.LayerNorm()
        self.dense_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(-2.0)  # CAVE: init as -2.0 important (low range for execution)
            # self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5)
        )
        # max_action_logstd = -1.6  # exp -1.6 = 0.2
        # init_action_logstd = -4.0  # exp -4.0 ~= 0.018
        self.init_action_logstd = -6.0  # exp -6.0 ~= 0.0025
        if self.config['ACTOR_STD'] == "state_dependent":
            self.dense_logstd = nn.Dense(
                # self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(max_action_logstd), name="log_std"
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(self.init_action_logstd), name="log_std"
            )

    def __call__(self, actor_embedding):
        actor_net = self.dense_0(actor_embedding)
        actor_net = self.ln_0(actor_net)
        actor_net = self.act_fn(actor_net)
        
        actor_mean = self.dense_mean(actor_net)
        # make sure the action mean is within the bounds (but allow slightly negative values)
        actor_mean = (jnp.tanh(actor_mean) * 0.5 + 0.5) * 1.01 - 0.01
        
        if self.config['ACTOR_STD'] == "state_dependent":
            actor_logtstd = self.dense_logstd(actor_net)
        elif self.config['ACTOR_STD'] == "param":
            init_action_logstd = self.init_action_logstd + jnp.log(2)  # more variance for exploration
            actor_logtstd = self.param("log_std", nn.initializers.constant(init_action_logstd), (self.action_dim,))
        elif self.config['ACTOR_STD'] == "fixed":
            init_action_logstd = self.init_action_logstd + jnp.log(2)  # more variance for exploration
            actor_logtstd = init_action_logstd
        else:
            raise ValueError(f"Invalid ACTOR_STD: {self.config['ACTOR_STD']}")

        # actor_logtstd = self.param("log_std", nn.initializers.constant(-1.6), (self.action_dim,))
        #Trying to get an initial std_dev of 0.2 (log(0.2)~=-0.7)
        # pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        pi = MultiVariateNormalDiagClipped(
            actor_mean * self.config['MAX_TASK_SIZE'],  # mean
            jnp.exp(actor_logtstd) * self.config['MAX_TASK_SIZE'],  # std
            self.config['MAX_TASK_SIZE'] / 100,  # max std
        )
        return pi
    

class ActorDisc(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    act_fn: Callable

    def setup(self):
        self.dense_0 = nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.ln_0 = nn.LayerNorm()

        self.action_outs = [
            nn.Dense(
                int(self.config['MAX_TASK_SIZE'] // self.config['REDUCE_ACTION_SPACE_BY']) + 1,  # +1 for the 0 action
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0)
                # bias_init=biased_constant(0.0, 10.0)  # bias for 0 action is 10
            ) for _ in range(self.action_dim)
        ]

    def __call__(self, actor_embedding):
        actor_net = self.dense_0(actor_embedding)
        actor_net = self.ln_0(actor_net)
        actor_net = self.act_fn(actor_net)
        print('actor_net', actor_net.shape, actor_net.dtype)

        action_logits = jnp.moveaxis(
            jnp.array([out(actor_net) for out in self.action_outs]),
            0, -2
        )
        print('action_logits', action_logits.shape, action_logits.dtype)
        pi = distrax.Independent(
            distrax.Categorical(logits=action_logits),
            1
        )
        return pi


class Critic(nn.Module):
    config: Dict
    act_fn: Callable

    def setup(self):
        self.dense_0 = nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.ln_0 = nn.LayerNorm()
        # self.dense_1 = nn.Dense(
        #     (self.config["HIDDEN_SIZE"] // 2).astype(jnp.int32),
        #     kernel_init=orthogonal(jnp.sqrt(2)), 
        #     bias_init=constant(0.0)
        # )
        # self.ln_1 = nn.LayerNorm()
        self.dense_2 = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def __call__(self, embedding):
        critic = self.dense_0(embedding)
        critic = self.ln_0(critic)
        critic = self.act_fn(critic)

        # critic = self.dense_1(critic)
        # critic = self.ln_1(critic)
        # critic = nn.relu(critic)

        critic = self.dense_2(critic)
        return jnp.squeeze(critic, axis=-1)



    # @nn.compact
    # def __call__(self, hidden, x):
    #     obs, dones = x
    #     if not self.config['JOINT_ACTOR_CRITIC_NET']:
    #         hidden, actor_hidden = hidden

    #     embedding = nn.Dense(
    #         128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
    #     )(obs)
    #     embedding = nn.LayerNorm()(embedding)
    #     embedding = nn.relu(embedding)
    #     rnn_in = (embedding, dones)
    #     hidden, embedding = ScannedRNN(name="rnn")(hidden, rnn_in)
    #     # embedding = nn.LayerNorm()(embedding)
    #     self.sow("intermediates", "embedding", embedding)

    #     ## ACTOR
    #     if self.config['JOINT_ACTOR_CRITIC_NET']:
    #         actor_embedding = embedding
    #     else:
    #         actor_embedding = nn.Dense(
    #             128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
    #         )(obs)
    #         actor_embedding = nn.LayerNorm()(actor_embedding)
    #         actor_embedding = nn.relu(actor_embedding)
    #         rnn_in = (actor_embedding, dones)
    #         hidden_actor, actor_embedding = ScannedRNN(name="actor_rnn")(actor_hidden, rnn_in)
    #         # embedding = nn.LayerNorm()(embedding)
    #         self.sow("intermediates", "actor_embedding", actor_embedding)

    #     actor_net = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
    #         actor_embedding
    #     )
    #     actor_net = nn.LayerNorm()(actor_net)
    #     actor_net = nn.relu(actor_net)
        
    #     actor_mean = nn.Dense(
    #         self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(-2.0)  # CAVE: init as -2.0 important (low range for execution)
    #         # self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5)
    #     )(actor_net)
    #     # make sure the action mean is within the bounds (but allow slightly negative values)
    #     actor_mean = (jnp.tanh(actor_mean) * 0.5 + 0.5) * 1.01 - 0.01

    #     # max_action_logstd = -1.6  # exp -1.6 = 0.2
    #     # init_action_logstd = -4.0  # exp -4.0 ~= 0.018
    #     init_action_logstd = -6.0  # exp -6.0 ~= 0.0025
    #     if self.config['ACTOR_STD'] == "state_dependent":
    #         actor_logtstd = nn.Dense(
    #             # self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(max_action_logstd), name="log_std"
    #             self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(init_action_logstd), name="log_std"
    #         )(actor_net)
    #     elif self.config['ACTOR_STD'] == "param":
    #         init_action_logstd += jnp.log(2)  # more variance for exploration
    #         actor_logtstd = self.param("log_std", nn.initializers.constant(init_action_logstd), (self.action_dim,))
    #     elif self.config['ACTOR_STD'] == "fixed":
    #         init_action_logstd += jnp.log(2)  # more variance for exploration
    #         actor_logtstd = init_action_logstd
    #     else:
    #         raise ValueError(f"Invalid ACTOR_STD: {self.config['ACTOR_STD']}")
    #     # actor_logtstd = self.param("log_std", nn.initializers.constant(-1.6), (self.action_dim,))
    #     #Trying to get an initial std_dev of 0.2 (log(0.2)~=-0.7)
    #     # pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
    #     pi = MultiVariateNormalDiagClipped(
    #         actor_mean * self.config['MAX_TASK_SIZE'],  # mean
    #         jnp.exp(actor_logtstd) * self.config['MAX_TASK_SIZE'],  # std
    #         self.config['MAX_TASK_SIZE'] / 100,  # max std
    #     )

    #     ## CRITIC

    #     critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
    #         embedding
    #     )
    #     critic = nn.LayerNorm()(critic)
    #     critic = nn.relu(critic)

    #     critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
    #         critic
    #     )
    #     critic = nn.LayerNorm()(critic)
    #     critic = nn.relu(critic)

    #     critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
    #         critic
    #     )

    #     if not self.config['JOINT_ACTOR_CRITIC_NET']:
    #         hidden = (hidden, actor_hidden)

    #     return hidden, pi, jnp.squeeze(critic, axis=-1)

class MultiVariateNormalDiagClipped(distrax.MultivariateNormalDiag):
    def __init__(
            self,
            loc: Optional[jax.Array] = None,
            scale_diag: Optional[jax.Array] = None,
            max_scale_diag: Optional[jax.Array] = None,
        ):
        self.max_scale_diag = max_scale_diag
        scale_diag = jnp.minimum(max_scale_diag, scale_diag)
        super().__init__(loc, scale_diag)

    def __getitem__(self, index) -> distrax.MultivariateNormalDiag:
        """See `Distribution.__getitem__`."""
        index = distrax.distribution.to_batch_shape_index(self.batch_shape, index)
        return MultiVariateNormalDiagClipped(
            loc=self.loc[index],
            scale_diag=self.scale_diag[index],
            max_scale_diag=self.max_scale_diag,
        )