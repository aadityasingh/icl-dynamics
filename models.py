"""Implementation of a Transformer model.

Adapted/inspired from:
  - https://github.com/paganpasta/eqxvision
  - https://github.com/neelnanda-io/TransformerLens

As is standard for JAX-based codebases, our forward passes are implemented in terms
of a single data point. vmap is used in the train step to make these batched calls.

Please see artificial_optogenetics_guide.md for explanation of most of this code.

Some functionality in this file (e.g., dropout) wasn't actually used in any of
our experiments, so we don't provide any guarantees on correctness.
"""
import numpy as np
from math import prod, sqrt
from functools import partial

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import equinox.nn as enn
import pdb

from jax import Array

from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


class Zero(eqx.Module):
  def __call__(
    self, 
    x: Array, 
    *, 
    key: Optional[jax.random.PRNGKey] = None,
    inference: Optional[bool] = None, 
    deterministic: Optional[bool] = None
  ):
    return 0


class Zeros(eqx.Module):
  def __call__(
    self, 
    x: Array, 
    *, 
    key: Optional[jax.random.PRNGKey] = None,
    inference: Optional[bool] = None, 
    deterministic: Optional[bool] = None
  ):
    return jnp.zeros(x.shape)


class CustomLinear(enn.Linear):
  """Copied from eqx source code and modified to use custom init"""

  weight: Array
  bias: Optional[Array]
  in_features: int = eqx.static_field()
  out_features: int = eqx.static_field()
  use_bias: bool = eqx.static_field()

  def __init__(
    self,
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    w_init = None,
    b_init = None,
    *,
    key: "jax.random.PRNGKey"
  ):
    """**Arguments:**
    - `in_features`: The input size.
    - `out_features`: The output size.
    - `use_bias`: Whether to add on a bias as well.
    - `w_init`: If not None, can be a str of function.
                if str, it should be in jax.nn.initializers and not require 
                  additional args
                if fn, it should be a partial that accepts in_axis and 
                  out_axis args to create an initializer, which then 
                  accepts a random key, shape and dtype
    - `b_init`: If not None, must be a function that accepts a shape and dtype. 
                Probably a partial.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

    Note initialization defaults to eqx initialization for backwards compat.
    A new module was used since weight and bias of enn.Linear cannot be naively
    reassigned later (dataclasses.FrozenInstanceError: cannot assign to field 'weight'),
    which was the first attempt at changing initialization via a wrapper.

    An alternate way (as I discovered later) was using eqx.tree_at, but as we were
    already using this custom module I left it as is. Could be cleaned up in future.

    Note that weight is also stored as (out_features, in_features) for eqx compat
    (despite haiku and jax.nn.init expecting (in_features, out_features)).
    """
    # Init eqx Module, but not Linear since we don't want to make weight/bias
    eqx.Module.__init__(self)
    wkey, bkey = jrandom.split(key, 2)
    lim = 1 / sqrt(in_features)

    if w_init is None:
      self.weight = jrandom.uniform(
        wkey, (out_features, in_features), minval=-lim, maxval=lim
      )
    else:
      if isinstance(w_init, str):
        init_fn = getattr(jax.nn.initializers, w_init)
      else:
        # allow for a partially constructed initializer to be passed in
        init_fn = w_init
      initializer = init_fn(in_axis=-1, out_axis=-2)
      self.weight = initializer(key=wkey, shape=(out_features, in_features))
    
    if use_bias:
      if b_init is None:
        self.bias = jrandom.uniform(bkey, (out_features,), minval=-lim, maxval=lim)
      else:
        self.bias = b_init(key=bkey, shape=(out_features,))
    else:
      self.bias = None

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias


class PositionalEmbed(enn.Linear):
  """Simple learned absolute positional embedding."""

  def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
    """Pad input to be compatible with the weight matrix."""
    pad_size = self.in_features - x.size
    return super().__call__(jnp.pad(x, (0, pad_size), "constant"))


class MLP(eqx.Module):
  """MLP block, doesn't use assign_fn formalism"""

  fc1: eqx.Module
  act: Callable
  drop1: enn.Dropout
  fc2: eqx.Module
  drop2: enn.Dropout

  def __init__(
    self,
    in_features: int,
    hidden_features: Optional[int] = None,
    out_features: Optional[int] = None,
    act: Callable = jnn.gelu,
    drop: Union[float, Tuple[float]] = 0.0,
    *,
    key: Array = None,
  ):
    """MLP block

    Args:
    - `in_features`: The expected dimension of the input.
    - `hidden_features`: Dimensionality of the hidden layer.
    - `out_features`: The dimension of the output feature.
    - `act`: Activation function to be applied to the intermediate layers.
    - `drop`: The probability associated with `Dropout`.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
    keys = jrandom.split(key, 2)

    self.fc1 = CustomLinear(key=keys[0], 
                              w_init='lecun_normal',
                              b_init=jax.nn.initializers.zeros,
                              in_features=in_features, 
                              out_features=hidden_features,
                              use_bias=True)
    self.act = act
    self.drop1 = enn.Dropout(drop_probs[0])
    self.fc2 = CustomLinear(key=keys[1], 
                              w_init='lecun_normal',
                              b_init=jax.nn.initializers.zeros,
                              in_features=hidden_features, 
                              out_features=out_features,
                              use_bias=True)
    self.drop2 = enn.Dropout(drop_probs[1])

  def __call__(self, x: Array, *, key: Array) -> Array:
    keys = jrandom.split(key, 2)
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x, key=keys[0])
    x = self.fc2(x)
    x = self.drop2(x, key=keys[1])
    return x


# Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_flax_roformer.py
def create_sinusoidal_positions(n_pos, dim, time=10000.0):
  denom = jnp.power(time, 2 * (jnp.arange(dim) // 2)/dim)
  position_enc = jnp.arange(n_pos)[:, None] / denom[None, :]
  return jnp.concatenate([jnp.sin(position_enc[:, 0::2]), jnp.cos(position_enc[:, 1::2])], axis=-1)


class SinusoidalEmbed:
  def __init__(self, embed_dim, time=10000.0):
    self.embed_dim = embed_dim
    self.time = time

  def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
    # Expect x to be a one-hot vector encoding position
    del key
    position = jnp.argmax(x)
    denom = jnp.power(
      self.time, 2 * (jnp.arange(self.embed_dim) // 2) / self.embed_dim)
    embedding = position / denom

    # Create sinusoidal embedding
    embedding = embedding.at[::2].set(jnp.sin(embedding[::2]))
    embedding = embedding.at[1::2].set(jnp.cos(embedding[1::2]))

    return embedding


def apply_rope(x, time=10000.0):
  # x should have dims: ... x len x model_dim
  assert x.shape[-1] % 2 == 0, 'Rope can only be used with even model dimension'
  pos = create_sinusoidal_positions(x.shape[-2], x.shape[-1], time=time)
  sin, cos = jnp.split(pos, 2, axis=-1)
  sin_pos = jnp.stack([sin, sin], axis=-1).reshape(pos.shape)
  cos_pos = jnp.stack([cos, cos], axis=-1).reshape(pos.shape)

  rotated = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)

  return x*cos_pos + rotated*sin_pos


def make_cache_assign(retval, cache=dict(), cache_mask=dict()):
  def cache_assign(name, value):
    '''
    DO NOT USE ON value THAT IS NOT "ATOMIC"
    '''
    if name in cache:
      assert name in cache_mask, 'Cache mask should match cache'
      retval[name] = jnp.where(cache_mask[name], cache[name], value)
    else:
      retval[name] = value
  return cache_assign


class AttentionBlock(eqx.Module):
  """Standard multi-headed self-attention block, with assign_fn formalism"""

  num_heads: int
  causal: bool
  use_rope: bool
  sin_time: float
  scale: float
  qkv: enn.Linear
  attn_drop: enn.Dropout
  proj: enn.Linear
  proj_drop: enn.Dropout

  def __init__(
    self,
    dim: int,
    num_heads: int,
    causal: bool = False,
    use_rope: bool = False,
    sin_time: float = 10000.0,
    qkv_bias: bool = False,
    qk_scale: Optional[float] = None,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    *,
    key: Array,
  ):
    """
    Args:
    - `dim`: The feature dimensions of the input.
    - `num_heads`: The number of attention heads.
    - `qkv_bias`: Whether to use bias a bias term in the query-key-value computation.
    - `qk_scale`: Scalar multiplier for the query-value (unnormalized attention) computation.
    - `attn_drop`: Dropout rate for attention.
    - `proj_drop`: Dropout rate for projection.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """

    super().__init__()
    keys = jrandom.split(key, 2)
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = qk_scale or head_dim**-0.5
    self.causal = causal
    self.use_rope = use_rope
    self.sin_time = sin_time

    self.qkv = CustomLinear(key=keys[0], 
                            w_init='lecun_normal',
                            b_init=jax.nn.initializers.zeros,
                            in_features=dim, 
                            out_features=dim * 3, 
                            use_bias=qkv_bias)
    self.attn_drop = enn.Dropout(attn_drop)
    self.proj = CustomLinear(key=keys[1], 
                              w_init='lecun_normal',
                              b_init=jax.nn.initializers.zeros,
                              in_features=dim, 
                              out_features=dim,
                              use_bias=True)
    self.proj_drop = enn.Dropout(proj_drop)

  def call_with_all_aux(self, x: Array, *, key: Array, cache=dict(), cache_mask=dict()) -> Sequence[Array]:
    # Initialize return dict and cache-aware assignment function
    r = dict(x=x)
    assign_fn = make_cache_assign(r, cache=cache, cache_mask=cache_mask)

    # N is sequence length, C is model dim
    N, C = x.shape
    keys = jrandom.split(key, 2)
    qkv = jax.vmap(self.qkv)(r['x'])
    qkv = jnp.reshape(qkv, (N, 3, self.num_heads, C // self.num_heads))
    # Reshape to qkv x heads x len x model dim
    assign_fn('qkv', jnp.transpose(qkv, axes=(1, 2, 0, 3)))
    # Split to 1 x heads x len x model dim
    pre_rope_q, pre_rope_k, v = jnp.split(r['qkv'], indices_or_sections=3)
    assign_fn('pre_rope_q', pre_rope_q)
    assign_fn('pre_rope_k', pre_rope_k)
    assign_fn('v',v)

    if self.use_rope:
      assign_fn('q', apply_rope(r['pre_rope_q'], time=self.sin_time))
      assign_fn('k', apply_rope(r['pre_rope_k'], time=self.sin_time))
    else:
      assign_fn('q', r['pre_rope_q'])
      assign_fn('k', r['pre_rope_k'])

    attn_pre_softmax = (r['q'] @ jnp.transpose(r['k'], (0, 1, 3, 2))) * self.scale
    if self.causal:
      mask = jnp.arange(N)[:, None] >= jnp.arange(N)[None, :]
      attn_pre_softmax = attn_pre_softmax*mask[None, None, :, :] + (1-mask)[None, None, :, :] * -1e20
    assign_fn('attn_pre_softmax', attn_pre_softmax)
    assign_fn('attn_scores', jnn.softmax(r['attn_pre_softmax'], axis=-1))
    assign_fn('attn_post_drop', self.attn_drop(r['attn_scores'], key=keys[0]))

    assign_fn('values_pre_proj', jnp.reshape(jnp.transpose((r['attn_post_drop'] @ r['v']), axes=(0, 2, 1, 3)), (N, C)))
    assign_fn('values_post_proj', jax.vmap(self.proj)(r['values_pre_proj']))
    assign_fn('out', self.proj_drop(r['values_post_proj'], key=keys[1]))

    return r

  def __call__(self, x: Array, *, key: Array, cache=dict(), cache_mask=dict()) -> Sequence[Array]:
    return self.call_with_all_aux(x, key=key, cache=cache, cache_mask=cache_mask)['out']


class TransformerBlock(eqx.Module):
  """Standard transformer block (MHA + MLP, with residuals), with assign_fn formalism"""
  norm1: eqx.Module
  attn: AttentionBlock
  drop_path1: enn.Dropout

  norm2: eqx.Module
  mlp: MLP
  drop_path2: enn.Dropout

  def __init__(
    self,
    dim: int,
    num_heads: int,
    causal: bool = False,
    use_rope: bool = False,
    sin_time: float = 10000.0,
    mlp_ratio: Optional[float] = 4.0,
    qkv_bias: bool = False,
    qk_scale: Optional[float] = None,
    mlp_drop: float = 0.0,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    path_drop: float = 0.0,
    act: Callable = jnn.gelu,
    norm_layer: eqx.Module = enn.LayerNorm,
    *,
    key: Array,
  ) -> None:
    """
    Args:
    - `dim`: The feature dimensions of the input.
    - `num_heads`: The number of equal parts to split the input along the `dim`.
    - `causal`: Whether or not to make the transformer causal
    - `use_rope`: Whether or not to use RoPE
    - `mlp_ratio`: For computing hidden dimension of the `MLP` (=`dim * mlp_ratio`).
    - `qkv_bias`: To add `bias` within the `qkv` computation.
    - `qk_scale`: For scaling the `query` `value` computation.
    - `mlp_drop`: Dropout rate for the `MLP`.
    - `attn_drop`: Dropout rate for the `AttentionBlock`.
    - `proj_drop`: Dropout rate for the `projection.
    - `path_drop`: Dropout rate for the non-residual pathway.
    - `act`: Activation applied on the intermediate outputs.
    - `norm_layer`: Normalisation applied to the intermediate outputs.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, 2)

    self.norm1 = norm_layer(dim) if norm_layer else enn.Identity()
    self.attn = AttentionBlock(
      dim,
      num_heads=num_heads,
      causal=causal,
      use_rope=use_rope,
      sin_time=sin_time,
      qkv_bias=qkv_bias,
      qk_scale=qk_scale,
      attn_drop=attn_drop,
      proj_drop=proj_drop,
      key=keys[0],
    )
    self.drop_path1 = (
      enn.Dropout(path_drop) if path_drop > 0.0 else enn.Identity()
    )

    if mlp_ratio is not None:
      self.norm2 = norm_layer(dim) if norm_layer else enn.Identity()
      self.mlp = MLP(
        in_features=dim,
        hidden_features=int(dim * mlp_ratio),
        act=act,
        drop=mlp_drop,
        key=keys[1],
      )
      self.drop_path2 = (
        enn.Dropout(path_drop) if path_drop > 0.0 else enn.Identity()
      )
    else:
      # No MLP head.
      self.norm2 = enn.Identity()
      self.mlp = enn.Identity()
      self.drop_path2 = Zeros()

  def call_with_all_aux(self, x: Array, *, key: Array, cache=dict(), cache_mask=dict()) -> Array:
    # Initialize return dict and cache-aware assignment function
    r = dict(x=x)
    assign_fn = make_cache_assign(r, cache=cache, cache_mask=cache_mask)

    keys = jrandom.split(key, 4)

    # Attention block.
    assign_fn('norm_inp', jax.vmap(self.norm1)(r['x']))
    r['attn_output'] = self.attn.call_with_all_aux(r['norm_inp'], key=keys[0], cache=cache.get('attn_output', dict()), cache_mask=cache_mask.get('attn_output', dict()))
    assign_fn('residual_post_attn', r['x'] + self.drop_path1(r['attn_output']['out'], key=keys[1]))

    # MLP head.
    assign_fn('norm_residual_post_attn', jax.vmap(self.norm2)(r['residual_post_attn']))
    assign_fn('mlp', jax.vmap(self.mlp)(r['norm_residual_post_attn'], key=jrandom.split(keys[2], x.shape[0])))
    assign_fn('out', r['residual_post_attn'] + self.drop_path2(r['mlp'], key=keys[3]))

    return r

  def __call__(self, x: Array, *, key: Array, cache=dict(), cache_mask=dict()) -> Array:
    return self.call_with_all_aux(x, key=key, cache=cache, cache_mask=cache_mask)['out']


class Transformer(eqx.Module):
  """Full transformer, with assign_fn formalism. This is likely the component most generally useful"""

  embed_dim: int

  blocks: Sequence[TransformerBlock]
  norm: eqx.Module

  unembed: enn.Linear
  inference: bool

  def __init__(
    self,
    num_classes: Optional[int],
    embed_dim: int,
    depth: int = 2,
    num_heads: int = 8,
    causal: bool = False,
    use_rope: bool = False,
    sin_time: float = 10000.0,
    mlp_ratio: Optional[float] = 4.0,
    qkv_bias: bool = True,
    qk_scale: Optional[float] = None,
    tok_embed_drop_rate: float = 0.0,
    pos_embed_drop_rate: float = 0.0,
    mlp_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    path_drop_rate: float = 0.0,
    norm_layer: eqx.Module = enn.LayerNorm,
    *,
    key: Array,
  ) -> None:
    """
    Args:
    - `num_classes`: Number of classes in the classification task, or `None` to omit output projection.
    - `embed_dim`: The input embedding dimension.
    - `depth`: Number of `TransformerBlock`s in the network.
    - `num_heads`: Number of attention heads within each `AttentionBlock`.
    - `causal`: Whether or not to make the transformer causal
    - `use_rope`: Whether or not to use RoPE
    - `mlp_ratio`: For computing hidden dimension of the `MLP`s, or `None` to omit `MLP` heads.
    - `qkv_bias`: Whether to use bias a bias term in the query-key-value computation.
    - `qk_scale`: Scalar multiplier for the query-value computation; defaults to `1 / sqrt(head_dim)`.
    - `embed_drop_rate`: Dropout rate used for the embedding matrix.
    - `mlp_drop_rate`: Dropout rate used within the `MLP`.
    - `attn_drop_rate`: Dropout rate used within the `AttentionBlock`s.
    - `path_drop_rate`: Dropout rate used within `TransformerBlock`s.
    - `norm_layer`: Normalisation applied to the intermediate outputs.
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
    """
    super().__init__()
    keys = jrandom.split(key, depth + 3)

    # Switch to `inference` mode for evaluations via `model = eqx.tree_inference(model)`.
    self.inference = False

    # Size of the embedding and thus the residual stream.
    self.embed_dim = embed_dim

    pdr = np.linspace(0, path_drop_rate, depth)
    self.blocks = [
      TransformerBlock(
        dim=embed_dim,
        num_heads=num_heads,
        causal=causal,
        use_rope=use_rope,
        sin_time=sin_time,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        mlp_drop=mlp_drop_rate,
        attn_drop=attn_drop_rate,
        path_drop=pdr[i],
        norm_layer=norm_layer,
        key=keys[i + 2],
      )
      for i in range(depth)
    ]

    self.norm = norm_layer(embed_dim) if norm_layer else enn.Identity()

    if num_classes is None:
      self.unembed = enn.Identity()
    else:
      self.unembed = CustomLinear(key=keys[-1], 
                                  w_init='lecun_normal',
                                  b_init=jax.nn.initializers.zeros,
                                  in_features=dim, 
                                  out_features=num_classes, 
                                  use_bias=True)

  def call_with_all_aux(self, x: Array, *, key: Array, cache=dict(), cache_mask=dict()) -> Array:
    # Initialize return dict and cache-aware assignment function
    r = dict(x=x)
    assign_fn = make_cache_assign(r, cache=cache, cache_mask=cache_mask)

    keys = jrandom.split(key, len(self.blocks))

    # `x` should be a sequence of embeddings.
    assert len(x.shape) == 2 and x.shape[1] == self.embed_dim

    block_cache = cache.get('block_outputs', [dict()]*len(self.blocks))
    block_cache_mask = cache_mask.get('block_outputs', [dict()]*len(self.blocks))

    # Residual stream
    r['block_outputs'] = [self.blocks[0].call_with_all_aux(r['x'], key=keys[0], cache=block_cache[0], cache_mask=block_cache_mask[0])]
    for l in range(1, len(keys)):
      r['block_outputs'].append(self.blocks[l].call_with_all_aux(r['block_outputs'][-1]['out'], 
                                                                  key=keys[l], 
                                                                  cache=block_cache[l], 
                                                                  cache_mask=block_cache_mask[l]))
    assign_fn('pre_unembed', jax.vmap(self.norm)(r['block_outputs'][-1]['out']))

    # Unembedding.
    assign_fn('out', jax.vmap(self.unembed)(r['pre_unembed']))

    return r

  def __call__(
    self, x: Array, *, key: Array, cache=dict(), cache_mask=dict()
  ) -> Array:
    return self.call_with_all_aux(x=x, key=key, cache=cache, cache_mask=cache_mask)['out']


class SequenceClassifier(eqx.Module):
  """A wrapper around Transformer that is specific to our data generating process.

  Specifically, this module ingests image and label pairs and reformats them
  as a interleaved sequence to feed to the transformer. It also handles embedding
  each modality.
  """

  embed_dim: int
  num_classes: int

  example_embed: enn.Linear
  example_embed_drop: enn.Dropout

  label_embed: enn.Linear

  pos_embed: PositionalEmbed
  pos_embed_drop: enn.Dropout

  transformer: Transformer

  unembed: enn.Linear
  inference: bool

  def __init__(
    self,
    example_shape: Tuple[int],
    num_classes: int,
    embed_dim: int,
    key: Array,
    example_embed_drop_rate: float = 0.0,
    pos_embed_drop_rate: float = 0.0,
    example_type: str = 'file',
    emb_init_scale: float = 0.02,
    pos_embedding_type: str = 'rope',
    sin_time: float = 10000.0,
    **transformer_kwargs,
  ):
    super().__init__()
    keys = jrandom.split(key, 4)

    # Switch to `inference` mode for evaluations via `model = eqx.tree_inference(model)`.
    self.inference = False

    # Example and label embeddings.
    self.embed_dim = embed_dim
    self.num_classes = num_classes

    if example_type == 'onehot':
      self.example_embed = CustomLinear(key=keys[0], 
                                        w_init=embed_init,
                                        in_features=prod(example_shape), 
                                        out_features=embed_dim, 
                                        use_bias=False)
    elif example_type == 'file':
      # In the case where the inputs are embeddings, and not onehot
      # we have a standard linear projection. We include a bias here,
      # since input embeddings may not be "centered"
      self.example_embed = CustomLinear(key=keys[0], 
                                        w_init='lecun_normal',
                                        b_init=jax.nn.initializers.zeros,
                                        in_features=prod(example_shape), 
                                        out_features=embed_dim, 
                                        use_bias=True)
    else:
      raise NotImplementedError
    self.example_embed_drop = enn.Dropout(p=example_embed_drop_rate)


    # To match Chan et al.'s code, we initialize label embeddings
    # to a truncated normal with stddev emb_init_scale = 0.02
    def _trunc_normal_init(key, shape):
      return emb_init_scale * jax.random.truncated_normal(key=key, 
                                                          lower=-2.,
                                                          upper=2.,
                                                          shape=shape)
    # Since custom_init_linear expects a w_init fn that takes
    # as arguments in_axis and out_axis, we create a fake wrapper
    embed_init = (lambda **kwargs: _trunc_normal_init)

    self.label_embed = CustomLinear(key=keys[1], 
                                    w_init=embed_init,
                                    in_features=num_classes, 
                                    out_features=embed_dim, 
                                    use_bias=False)

    if pos_embedding_type == 'rope':
      self.pos_embed = Zeros()
      self.pos_embed_drop = Zero()
    elif pos_embedding_type == 'sinusoidal':
      self.pos_embed = SinusoidalEmbed(embed_dim, time=sin_time)
      self.pos_embed_drop = enn.Identity()
    elif pos_embedding_type == 'ape':
      max_seq_len = 32
      self.pos_embed = PositionalEmbed(max_seq_len, embed_dim, key=keys[1], use_bias=False)
      self.pos_embed_drop = enn.Dropout(p=pos_embed_drop_rate)
    else:
      raise NotImplementedError

    self.transformer = Transformer(
      embed_dim=embed_dim,
      num_classes=None,  # Handle output projection elsewhere.
      key=keys[2],
      use_rope= (pos_embedding_type=='rope'),
      sin_time=sin_time,
      **transformer_kwargs
    )

    if num_classes is None:
      self.unembed = enn.Identity()
    else:
      self.unembed = CustomLinear(key=keys[-1], 
                                  w_init='lecun_normal',
                                  b_init=jax.nn.initializers.zeros,
                                  in_features=embed_dim, 
                                  out_features=num_classes, 
                                  use_bias=True)

  def call_with_all_aux(
    self, examples: Array, labels: Array, *, key: Array, cache=dict(), cache_mask=dict()
  ) -> Array:
    """Process the sequence of `examples` and `labels`."""

    # Initialize return dict and cache-aware assignment function
    r = dict(examples=examples, labels=labels)
    assign_fn = make_cache_assign(r, cache=cache, cache_mask=cache_mask)

    keys = jrandom.split(key, 2)

    num_pairs = examples.shape[0]
    assert num_pairs == labels.shape[0]

    # Example embedding.
    # Example dropout is deterministic based on exemplar, to align with
    # https://github.com/deepmind/emergent_in_context_learning/blob/eba75a4208b8927cc1e981384a2cc7e014677095/modules/embedding.py#L137-L142.
    example_embedding = self.example_embed_drop(
      jax.vmap(self.example_embed)(r['examples']),
      # TODO(eringrant): Is this kosher (runtime value for PRNGKey seed)?
      key=jrandom.PRNGKey(jnp.sum(r['examples']).astype(int)),  # type: ignore[arg-type]
      inference=False,
    )
    # (aadityasingh): Note we didn't end up using embedding dropout
    # so the above TODO wasn't resolved.
    assign_fn('example_embedding', example_embedding)

    # Label embedding.
    assign_fn('onehot_labels', jnn.one_hot(r['labels'], self.num_classes))
    assign_fn('label_embedding', jax.vmap(self.label_embed)(r['onehot_labels']))

    # Interleave example and label embeddings, except for the final (query) label.
    assert r['example_embedding'].dtype == r['label_embedding'].dtype

    tok_embedding = jnp.empty(
      (num_pairs * 2 - 1, self.embed_dim), dtype=r['example_embedding'].dtype
    )
    tok_embedding = tok_embedding.at[0::2, :].set(r['example_embedding'])
    tok_embedding = tok_embedding.at[1::2, :].set(r['label_embedding'][:-1, :])
    assign_fn('tok_embedding', tok_embedding)

    # Positional embedding.
    seq_len, _ = r['tok_embedding'].shape
    pos = jax.nn.one_hot(jnp.arange(seq_len, dtype=jnp.float32), seq_len)
    pos_embedding = self.pos_embed_drop(
      jax.vmap(self.pos_embed)(pos), key=keys[0]
    )
    assign_fn('pos_embedding', pos_embedding)

    assign_fn('embedding', r['tok_embedding'] + r['pos_embedding'])
    r['transformer_output'] = self.transformer.call_with_all_aux(r['embedding'], key=keys[1], cache=cache.get('transformer_output', dict()), cache_mask=cache_mask.get('transformer_output', dict()))
    assign_fn('unembedding', jax.vmap(self.unembed)(r['transformer_output']['out']))

    # Discard labels predicted for labels, i.e., undo interleaving.
    assign_fn('out', r['unembedding'][0::2, :])

    return r

  def __call__(
    self, examples: Array, labels: Array, *, key: Array, cache=dict(), cache_mask=dict()
  ) -> Array:
    """Process the sequence of `examples` and `labels`."""
    return self.call_with_all_aux(examples=examples, labels=labels, key=key, cache=cache, cache_mask=cache_mask)['out']

