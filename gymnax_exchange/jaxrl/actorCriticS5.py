import functools
from typing import Callable, Optional, Sequence, NamedTuple, Any, Dict
import jax
import jax.numpy as jnp
from jax._src import dtypes
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

from purejaxrl.purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from purejaxrl.purejaxrl.experimental.s5.wrappers import FlattenObservationWrapper, LogWrapper


d_model = 64  # === HIDDEN_SIZE
ssm_size = 128  # 256
C_init = "lecun_normal"
discretization="zoh"
dt_min=0.001
dt_max=0.1
n_layers = (2, 2, 2)
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


ssm_init_fn = init_S5SSM(
    H=d_model,
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
    bidirectional=bidirectional
)


class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    activation_fn_str: str = "half_glu1"

    def setup(self):
        self.encoder = EncoderS5(name='embedding', config=self.config, activation_fn_str=self.activation_fn_str)
        self.critic = CriticS5(self.config, activation_fn_str=self.activation_fn_str)
        if self.config['CONT_ACTIONS']:
            raise NotImplementedError('Continuous actions not supported for S5')
        else:
            self.actor = ActorDiscS5(self.action_dim, self.config, activation_fn_str=self.activation_fn_str)
        if not self.config['JOINT_ACTOR_CRITIC_NET']:
            self.actor_embedding = EncoderS5(name='actor_embedding', config=self.config, activation_fn_str=self.activation_fn_str)

    def __call__(self, hidden_all, x):
        obs, dones = x
        hidden_enc, hidden_actor, hidden_critic = hidden_all

        # jax.debug.print('obs {}', obs)
        # jax.debug.print('hidden {}', hidden)

        hidden_enc, embedding = self.encoder(hidden_enc, obs, dones)
        # self.sow("intermediates", "embedding_s5", embedding)

        ## ACTOR
        if self.config['JOINT_ACTOR_CRITIC_NET']:
            actor_embedding = embedding
        else:
            # hidden_actor, actor_embedding = self.actor_embedding(hidden_actor, obs, dones)
            # self.sow("intermediates", "actor_embedding_s5", actor_embedding)
            raise NotImplementedError('Not implemented')

        pi = self.actor(hidden_actor, actor_embedding, dones)

        ## CRITIC
        value = self.critic(hidden_critic, embedding, dones)

        hidden_all = (hidden_enc, hidden_actor, hidden_critic)

        return hidden_all, pi, value

    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_layers):
        # Use a dummy key since the default state init fn is just zeros.
        return tuple([jnp.zeros((1, batch_size, hidden_size), dtype=jnp.complex64) for _ in range(n)] for n in n_layers)


class EncoderS5(nn.Module):
    name: str = "encoder"
    config: Dict
    activation_fn_str: str

    def setup(self):
        self.dense_0 = nn.Dense(
            self.config["HIDDEN_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )
        # self.ln_0 = nn.LayerNorm()
        self.s5 = StackedEncoderModel(
            name=self.name + "_s5",
            ssm=ssm_init_fn,
            d_model=self.config["HIDDEN_SIZE"],
            n_layers=n_layers[0],
            activation=self.activation_fn_str,
        )

    def __call__(self, hidden, obs, dones):
        # jax.debug.print('encoder obs: {}, hidden {}', obs, hidden)

        embedding = self.dense_0(obs)
        # embedding = self.ln_0(embedding)
        hidden, embedding = self.s5(hidden, embedding, dones)
        return hidden, embedding


class ActorDiscS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    activation_fn_str: str

    def setup(self):
        self.s5 = StackedEncoderModel(
            name=self.name + "_s5",
            ssm=ssm_init_fn,
            d_model=self.config["HIDDEN_SIZE"],
            n_layers=n_layers[1],
            activation=self.activation_fn_str,
        )
        self.action_outs = [
            nn.Dense(
                int(self.config['MAX_TASK_SIZE'] // self.config['REDUCE_ACTION_SPACE_BY']) + 1,  # +1 for the 0 action
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0)
            ) for _ in range(self.action_dim)
        ]

    def __call__(self, hidden, actor_embedding, dones):
        hidden_out, actor_net = self.s5(hidden, actor_embedding, dones)
        # print('actor_net', actor_net.shape, actor_net.dtype)

        action_logits = jnp.moveaxis(
            jnp.array([out(actor_net) for out in self.action_outs]),
            0, -2
        )
        # print('action_logits', action_logits.shape, action_logits.dtype)
        pi = distrax.Independent(
            distrax.Categorical(logits=action_logits),
            1
        )
        return pi


class CriticS5(nn.Module):
    config: Dict
    activation_fn_str: str

    def setup(self):
        self.s5 = StackedEncoderModel(
            name=self.name + "_s5",
            ssm=ssm_init_fn,
            d_model=self.config["HIDDEN_SIZE"],
            n_layers=n_layers[2],
            activation=self.activation_fn_str,
        )
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def __call__(self, hidden, embedding, dones):
        hidden_out, critic = self.s5(hidden, embedding, dones)
        critic = self.value_decoder(critic)
        return jnp.squeeze(critic, axis=-1)
