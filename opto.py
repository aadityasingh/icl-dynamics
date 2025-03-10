'''
This file contains argparse options and forward function
constructors for all artificial optogenetics experiments involving
clamping throughout training.

See artificial_optogenetics_guide.md for more information.

Note: this file contains a lot of extra functionality (e.g., match_heads)
beyond what is present in the papers -- these functions were used at one
point or another in our experimentation. We leave them in with this final
release as they might be useful to future researchers.
'''

import numpy as np
from functools import partial
import argparse
import pdb
import json

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import equinox.nn as enn

import models
import main_utils


def add_args_to_parser(parser):
  parser.add_argument('--opto_specific_fn', type=str, default=None, 
                      choices=['standard',
                                'perfect_prev_token_input'],
                      help='Allows for specific implementations. Most optogenetic manipulations are preformed using "standard", which supports many different options (see other opts). However, in some niche cases (e.g., layer 1 clamp -- "perfect_prev_token_input"), we found it useful to have separate custom implementations to allow for more involved clamps. This setup also makes it easy to add more such custom implementations.')
  parser.add_argument('--opto_induction_heads', nargs='+', type=str, default=None, help="List of pairs of layer:head to give perfect induction patterns. Layers are 0-indexed. For example, to make heads 3 and 7 in layer 1 induction heads, this would be '1:3 1:7'")
  parser.add_argument('--opto_induction_head_strength', type=float, default=1.0, help="Attention given to correct token. 1-this is given to incorrect token. Assumes only two exemplar label pairs. All induction heads specified in previous step are given this same strength (could be modified to support different strengths for each head to measure competition). Note, for legacy reasons this is a bit confusingly named (it's different than the 'induction strength' we plot, which is `2*this - 1`)")
  parser.add_argument('--opto_match_heads', nargs='+', type=str, default=None, help="List of pairs of layer:head to make operate like IHs, except the strength is based on input similarity with some temperature (essentially, this is setting W_K^T W_Q to be temp*I, and ignoring positional embeddings and disallowing attending to self). Layers are 0-indexed. For example, to make heads 3 and 7 in layer 1 match heads, this would be '1:3 1:7'")
  parser.add_argument('--opto_match_head_temperature', type=float, default=0.0, help="Temperature in match heads. Defaults to 0, which will be a hardmax.")
  parser.add_argument('--opto_prev_token_heads', nargs='+', type=str, default=None, help="list of pairs of layer:head to give perfect previous token patterns. Layers are 0-indexed. For example, to make heads 0 and 1 in layer 0 previous token heads, this would be '0:0 0:1'")
  parser.add_argument('--opto_prev_token_head_strength', type=float, default=1.0, help="Attention given to the previous token. 1-this is given to self. Made to be compatible with opto_prev_token_only_label.")
  parser.add_argument('--opto_prev_token_only_label', action='store_true', help='Whether or not to make previous token head clamps only apply to label tokens')
  parser.add_argument('--opto_disallow_prev_token_heads', nargs='+', type=str, default=None, help="list of pairs of layer:head to prevent attending to previous token. Layers are 0-indexed. For example, to make heads 0 and 1 in layer 0 previous token heads, this would be '0:0 0:1'")
  parser.add_argument('--opto_disallow_prev_token_only_label', action='store_true', help='Whether or not to make disallow previous token attention clamps to only apply to label tokens')
  parser.add_argument('--opto_ablate_heads', nargs='+', type=str, default=None, help="list of pairs of layer:head to ablate the values of. Layers are 0-indexed. For example, to make heads 3 and 7 in layer 1 not contribute to output, this would be '1:3 1:7'")
  parser.add_argument('--opto_ablate_embedding', action='store_true', help="If true, ablates the embedding. If not preserving things, this is pretty degenrate/should 0 out the network.")
  parser.add_argument('--opto_perfect_copy_from_head_pattern', type=str, default=None, help="Single pair of layer:head whose pre softmax attention scores will be used to make the output logits. This automatically sets logits for labels not in context to -inf.")
  parser.add_argument('--opto_set_head_temperature', nargs='+', type=str, default=None, help="list of triplets of layer:head:temp to rescale temperature on. Layers are 0-indexed, temp can/should be a float. Currently rescales at ALL token indices, could make an analogous version just for query. Current implementation only supports doing this for heads in a *single* layer.")
  parser.add_argument('--opto_final_layer_silence_non_label_attn', nargs='+', type=int, default=None, help="list of heads which are only allowed to attend to label tokens in the final layer of the model.")
  parser.add_argument('--opto_preserve_patterns', action='store_true', help="Whether or not to preserve attention patterns when performing value ablations. Note, if induction heads are concurrently specified, the preserved patterns will be overwritten with IHs. Note, this should be equivalent to preserving queries and keys.")
  parser.add_argument('--opto_preserve_values', action='store_true', help="Whether or not to preserve the values of future layers.")
  parser.add_argument('--opto_preserve_queries', action='store_true', help="Whether or not to preserve the queries of future layers.")
  parser.add_argument('--opto_preserve_keys', action='store_true', help="Whether or not to preserve the keys of future layers.")

  # Grafting involves combining an existing model with the current model.
  # Grafting in can be used to fix network weights up to a certain layer.
  parser.add_argument('--opto_graft_in_model_ckpt', type=str, default=None, help="Path to checkpoint from which to graft activations")
  parser.add_argument('--opto_graft_in_model_cfg', type=str, default=None, help="Path to config for model to graft in activations from")
  parser.add_argument('--opto_graft_in_model_till_layer', type=int, default=None, help="Which layer to use grafted model up till")

  # Grafting out can be used to fix network weights from a certain layer.
  parser.add_argument('--opto_graft_out_model_ckpt', type=str, default=None, help="Path to checkpoint from which to take later layers")
  parser.add_argument('--opto_graft_out_model_cfg', type=str, default=None, help="Path to config for model to graft in activations from")
  parser.add_argument('--opto_graft_out_model_from_layer', type=int, default=None, help="Which layer to use grafted model from")


def add_defaults_if_not_present(opts):
  '''
  Method that adds default arguments for opto to a parser output. Useful
  for restoring from older model configs (that were trained without opto 
  or with older opto arguments)
  '''
  parser = argparse.ArgumentParser()
  add_args_to_parser(parser)
  defaults = vars(parser.parse_args([]))
  to_modify = vars(opts)
  for k in defaults:
    if k not in to_modify:
      to_modify[k] = defaults[k]


def get_heads_from_str_list(str_list):
  '''
  Takes in a list of strings, like ['1:3', '1:7'], and returns a dict, like {1: [3,7]}
  '''
  retval = dict()
  for pair in str_list:
    parsed = [int(v) for v in pair.split(':')]
    retval.setdefault(parsed[0], set()).add(parsed[1])
  for k in retval:
    retval[k] = list(retval[k])
  return retval


def get_head_temp_dict_from_str(str_list):
  '''
  Takes in a list of strings, like ['1:3:0.5', '1:7:0.6'], and returns a nested dict of head temperatures, like {1: {3: 0.5, 7: 0.6}}
  '''
  retval = dict()
  for trip in str_list:
    pieces = trip.split(':')
    assert len(pieces) == 3
    retval.setdefault(int(pieces[0]), dict())[int(pieces[1])] = float(pieces[2])
  return retval


def get_default_cache_and_mask(depth):
  '''
  Creates a default cache.

  Note we could also use jnp.zeros_like on the first pass of
  the model. Since we may not always do two passes (e.g. when
  just fixing a single head to be an induction head), we found
  this method useful.
  '''
  cache = dict()
  cache_mask = dict()
  cache.setdefault('transformer_output', dict()).setdefault(
                    'block_outputs', 
                    [dict(attn_output=dict()) for i in range(depth)]
                    )
  cache_mask.setdefault('transformer_output', dict()).setdefault(
                        'block_outputs', 
                        [dict(attn_output=dict()) for i in range(depth)]
                        )
  return cache, cache_mask


def check_non_overlapping_sets(head_dict1, head_dict2):
  '''
  Util method for making sure fixed heads don't overlap
  '''
  retval = True
  for l in set(head_dict1).intersection(set(head_dict2)):
    retval = retval and (len(set(head_dict1[l]).intersection(set(head_dict2[l]))) == 0)
  return retval


def check_anything_to_graft_in(opts):
  return (opts.opto_graft_in_model_till_layer is not None)


def check_anything_to_graft_out(opts):
  return (opts.opto_graft_out_model_from_layer is not None)


def default_model_fwd_fn(model, x, y, key):
  '''
  Default forward function, used to match signature that is returned by make_fn_from_opts
  '''
  return model.call_with_all_aux(examples=x, labels=y, key=key, cache=dict(), cache_mask=dict())


def make_fn_from_opts(opts, default_fn=default_model_fwd_fn):
  '''
  The opts will be used as specified above

  Returns:
    A callable function that takes in model, single x, single y, single key
    and return call_with_all_aux output with the specified caching scheme.

    This forward function can then be vmapped to process batches of data.

    If no opto parameters are specified, this returns the dummy default_model_fwd_fn
  '''
  add_defaults_if_not_present(opts)

  ### First, we check if any optogenetic manipulation is being applied

  opto_used = opts.opto_ablate_embedding

  if opts.opto_specific_fn is not None:
    opto_used = True

  induction_heads = dict()
  if opts.opto_induction_heads is not None:
    opto_used = True
    print("WARNING: opto_induction_heads is only implemented for 2 exemplar-label pairs in context")
    induction_heads = get_heads_from_str_list(opts.opto_induction_heads)

  match_heads = dict()
  if opts.opto_match_heads is not None:
    opto_used = True
    print("WARNING: opto_match_heads is only implemented for 2 exemplar-label pairs in context")
    match_heads = get_heads_from_str_list(opts.opto_match_heads)
    assert opts.opto_match_head_temperature >= 0

  prev_token_heads = dict()
  if opts.opto_prev_token_heads is not None:
    opto_used = True
    prev_token_heads = get_heads_from_str_list(opts.opto_prev_token_heads)

  disallow_prev_token_heads = dict()
  if opts.opto_disallow_prev_token_heads is not None:
    opto_used = True
    disallow_prev_token_heads = get_heads_from_str_list(opts.opto_disallow_prev_token_heads)

  assert check_non_overlapping_sets(induction_heads, prev_token_heads), 'Same head cannot be both previous token and induction'

  assert check_non_overlapping_sets(disallow_prev_token_heads, prev_token_heads), 'Same head cannot have disallowed previous token and previous token'

  ablate_heads = dict()
  if opts.opto_ablate_heads is not None:
    opto_used = True
    ablate_heads = get_heads_from_str_list(opts.opto_ablate_heads)

  set_head_temps = dict()
  if opts.opto_set_head_temperature is not None:
    opto_used = True
    set_head_temps = get_head_temp_dict_from_str(opts.opto_set_head_temperature)
    assert len(set_head_temps) == 1, 'Set head temperature only supports heads in one layer currently'

  silence_non_label_heads = None
  if opts.opto_final_layer_silence_non_label_attn is not None:
    opto_used = True
    silence_non_label_heads = jnp.array(sorted(opts.opto_final_layer_silence_non_label_attn))

  output_from_head = None
  if opts.opto_perfect_copy_from_head_pattern is not None:
    opto_used = True
    print("WARNING: opto_perfect_copy_from_head_pattern is only implemented for 2 exemplar-label pairs in context")
    output_from_head = [int(s) for s in opts.opto_perfect_copy_from_head_pattern.split(':')]

  graft_in_model = None
  if opts.opto_graft_in_model_ckpt is not None:
    assert opts.opto_graft_in_model_cfg is not None, "Must specify graft in model config"
    assert check_anything_to_graft_in(opts), "No point in loading a graft in model if nothing to graft in"
    opto_used = True
    graft_in_opts = main_utils.get_opts_from_json_file(opts.opto_graft_in_model_cfg)
    graft_in_model = main_utils.get_model_from_opts(graft_in_opts)
    # Recursive call, but at some point the grafted model opts should be None
    graft_in_fwd_fn = make_fn_from_opts(graft_in_opts)

    # We just need this for the checkpoint restore
    graft_in_opt_state = main_utils.get_optimizer_from_opts(graft_in_opts).init(eqx.filter(graft_in_model, eqx.is_array))

    graft_in_ckpt_fmt = {'iter': -1,
                'seeds': {'eval_model_seed': jax.random.PRNGKey(0),
                          'train_data_seed': jax.random.PRNGKey(0),
                          'train_model_seed': jax.random.PRNGKey(0)},
                'opt_state': graft_in_opt_state,
                'model': graft_in_model}
    graft_in_model = eqx.tree_deserialise_leaves(opts.opto_graft_in_model_ckpt, graft_in_ckpt_fmt)['model']

  graft_out_model = None
  if opts.opto_graft_out_model_ckpt is not None:
    assert opts.opto_graft_out_model_cfg is not None, "Must specify graft out model config"
    assert check_anything_to_graft_out(opts), "No point in loading a graft out model if nothing to graft out"
    opto_used = True
    graft_out_opts = main_utils.get_opts_from_json_file(opts.opto_graft_out_model_cfg)
    graft_out_model = main_utils.get_model_from_opts(graft_out_opts)
    print("WARNING: Graft out does not support carrying over opto. If the model being used to graft out relied on opto, those manipulations will *not* be applied.")

    graft_out_opt_state = main_utils.get_optimizer_from_opts(graft_out_opts).init(eqx.filter(graft_out_model, eqx.is_array))

    graft_out_ckpt_fmt = {'iter': -1,
                'seeds': {'eval_model_seed': jax.random.PRNGKey(0),
                          'train_data_seed': jax.random.PRNGKey(0),
                          'train_model_seed': jax.random.PRNGKey(0)}, 
                'opt_state': graft_out_opt_state,
                'model': graft_out_model}
    graft_out_model = eqx.tree_deserialise_leaves(opts.opto_graft_out_model_ckpt, graft_out_ckpt_fmt)['model']

  # Return default if not using optogenetics
  if not opto_used:
    return default_fn
  else:
    if opts.opto_specific_fn is None:
      opts.opto_specific_fn = 'standard'

  print("Opto being used...")

  if opts.opto_specific_fn == 'standard':
    def standard_call(model, x, y, key):
      '''
      The "standard" call function used for most experiments.

      This function accepts most of the args from opts and composes them, allowing for various combinations.
      It optionally uses two forward passes when necessary (when preserving patterns/values, or when
      outputting from a specific head's attention pattern).

      For more complex cases requiring more iterative forward passes, we code up custom call functions
      and make them accessible to the "top-level" opts.opto_specific_fn argument.

      This function assumes various things about our specific setup (e.g., sequences that are
      composed of exemplar-label pairs followed by a query)
      '''
      depth = len(model.transformer.blocks)
      context_len = 2*y.shape[0] - 1

      # Currently, all transformers have the same num_heads per layer
      num_heads = model.transformer.blocks[0].attn.num_heads
      per_head_dim = model.transformer.embed_dim // num_heads

      cache, cache_mask = get_default_cache_and_mask(depth)

      # For various optogenetic ablations, we require two forward passes through the same
      # network (e.g., pattern preserving). These are handled here and the cache is updated.

      # We set these activations before other manipulations, since we want those to be able to
      # override things like pattern preservations. This was most useful for our cases
      if (opts.opto_preserve_patterns or opts.opto_preserve_values or
          opts.opto_preserve_keys or opts.opto_preserve_queries or
          (output_from_head is not None) or
          (opts.opto_set_head_temperature is not None)):
        base_cache = model.call_with_all_aux(examples=x, labels=y, key=key, cache=dict(), cache_mask=dict())
        if opts.opto_preserve_patterns:
          for d in range(depth):
            cache['transformer_output']['block_outputs'][d]['attn_output']['attn_scores'] = base_cache['transformer_output']['block_outputs'][d]['attn_output']['attn_scores']
            cache_mask['transformer_output']['block_outputs'][d]['attn_output']['attn_scores'] = jnp.ones_like(cache['transformer_output']['block_outputs'][d]['attn_output']['attn_scores'], dtype=bool)
        if opts.opto_preserve_values:
          for d in range(depth):
            cache['transformer_output']['block_outputs'][d]['attn_output']['v'] = base_cache['transformer_output']['block_outputs'][d]['attn_output']['v']
            cache_mask['transformer_output']['block_outputs'][d]['attn_output']['v'] = jnp.ones_like(cache['transformer_output']['block_outputs'][d]['attn_output']['v'], dtype=bool)
        # We can fix pre_rope_q or q, shouldn't matter since apply_rope is a deterministic function
        if opts.opto_preserve_queries:
          for d in range(depth):
            cache['transformer_output']['block_outputs'][d]['attn_output']['q'] = base_cache['transformer_output']['block_outputs'][d]['attn_output']['q']
            cache_mask['transformer_output']['block_outputs'][d]['attn_output']['q'] = jnp.ones_like(cache['transformer_output']['block_outputs'][d]['attn_output']['q'], dtype=bool)
        if opts.opto_preserve_keys:
          for d in range(depth):
            cache['transformer_output']['block_outputs'][d]['attn_output']['k'] = base_cache['transformer_output']['block_outputs'][d]['attn_output']['k']
            cache_mask['transformer_output']['block_outputs'][d]['attn_output']['k'] = jnp.ones_like(cache['transformer_output']['block_outputs'][d]['attn_output']['k'], dtype=bool)

        # For these two, we carry forward the whole attn_pre_softmax
        # even though we're just clamping a few heads. This may lead
        # to some mild jax speedups, and it's also just easier to implement.
        if opts.opto_set_head_temperature is not None:
          for layer in set_head_temps:
            # Do this per layer since it could be diff
            head_temps = jnp.ones(num_heads)
            for head in set_head_temps[layer]:
              head_temps = head_temps.at[head].set(set_head_temps[layer][head])

            cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'] = base_cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax']/head_temps[None, :, None, None]

            cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'] = jnp.ones_like(cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'], dtype=bool)

        if output_from_head is not None:
          logits = base_cache['transformer_output']['block_outputs'][output_from_head[0]]['attn_output']['attn_pre_softmax'][0, output_from_head[1], -1, jnp.array([1,3])]
          cache['out'] = base_cache['out'].at[-1,:].set(-1e9).at[-1,y[:-1]].set(logits)
          cache_mask['out'] = jnp.zeros_like(cache['out']).at[-1,:].set(True)

      # When grafting in, we require a forward pass through the graft_model
      # Note, the way this is done is actually quite versatile. The model
      # being grafted could have its own optogenetic manipulations specified in graft_opts.
      if graft_in_model is not None:
        graft_in_cache = graft_in_fwd_fn(model=graft_in_model, x=x, y=y, key=key)
        layer = opts.opto_graft_in_model_till_layer
        cache['transformer_output']['block_outputs'][layer]['out'] = graft_in_cache['transformer_output']['block_outputs'][layer]['out']
        cache_mask['transformer_output']['block_outputs'][layer]['out'] = jnp.ones_like(cache['transformer_output']['block_outputs'][layer]['out'], dtype=bool)


      # If we're ablating the embedding layer, we do so here -- this is a blunt instrument to silence all non preserved network function
      if opts.opto_ablate_embedding:
        cache.setdefault('tok_embedding', jnp.zeros((context_len, model.transformer.embed_dim)))
        cache_mask.setdefault('tok_embedding', jnp.ones((context_len, model.transformer.embed_dim)).astype(bool))

      ### Ablating and setting patterns has roughly the same format in our formalism.
      # First, we initialize a cached variable (if it's not already present)
      # Then, we perform the relevant manipulation to the specific layers/heads
      # To make sure only the relevant heads are manipulated, we set the cache_mask
      # accordingly.

      # This allows to keep using many dimensional tensors in our model intermediates (useful
      # for compilation/efficiency reasons), while also allowing single-activation modulation.

      for layer in ablate_heads:
        # Set default values if not specified already
        cache['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('v', jnp.zeros((1, num_heads, context_len, per_head_dim)))
        # Ablate relevant heads
        cache['transformer_output']['block_outputs'][layer]['attn_output']['v'] = cache['transformer_output']['block_outputs'][layer]['attn_output']['v'].at[0, ablate_heads[layer], :, :].set(0)

        # Set default values mask if not specified already
        cache_mask['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('v', jnp.zeros((1, num_heads, context_len, per_head_dim), dtype=bool))
        # Set ablated values to True to make sure the ablation goes into effect
        cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['v'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['v'].at[0, ablate_heads[layer], :, :].set(True)

      # The order in which we set previous token and induction heads get set shouldn't matter
      # Since we've already checked that they're non-overlapping. We do previous token first.

      prev_token_pattern = (jnp.eye(context_len) * (1-opts.opto_prev_token_head_strength)).at[jnp.arange(1, context_len), jnp.arange(context_len-1)].set(opts.opto_prev_token_head_strength).at[0,0].set(1)

      for layer in prev_token_heads:
        cache['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_scores', jnp.zeros((1, num_heads, context_len, context_len)))
        cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,prev_token_heads[layer],:,:].set(prev_token_pattern)

        cache_mask['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_scores', jnp.zeros((1, num_heads, context_len, context_len), dtype=bool))

        if opts.opto_prev_token_only_label:
          cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,prev_token_heads[layer],1::2,:].set(True)
        else:
          cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,prev_token_heads[layer],:,:].set(True)

      ih_pattern = jnp.stack([jnp.zeros(y.shape),
                              jnp.where(y == y[-1], opts.opto_induction_head_strength,
                                                    1-opts.opto_induction_head_strength)
                              ], axis=1).reshape(-1)[:-1]

      for layer in induction_heads:
        cache['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_scores', jnp.zeros((1, num_heads, context_len, context_len)))
        cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,induction_heads[layer],-1,:].set(ih_pattern)

        cache_mask['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_scores', jnp.zeros((1, num_heads, context_len, context_len), dtype=bool))
        cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,induction_heads[layer],-1,:].set(True)

      # Note: match heads will "override" set_head_temperature or
      # silence_non_label_heads. The latter doesn't matter since
      # match heads implicitly also silence non_label. The former
      # is ok since match_heads accept their own head_temperature param.
      if len(match_heads) > 0:
        norm_x = x / jnp.sqrt(jnp.sum(x**2, axis=1))[:, None]
        raw_scores = jnp.einsum('cd,d->c', norm_x[:-1, :], norm_x[-1])
        if opts.opto_match_head_temperature == 0:
          # Pick out max
          scores = (raw_scores == jnp.max(raw_scores))
        else:
          scores = jax.nn.softmax(raw_scores/opts.opto_match_head_temperature)
        # Interleave 0 attention to exemplars, including self
        # We need to make scores the same shape as y, so we add a 0 that later gets truncated
        match_pattern = jnp.zeros(y.shape[0]*2-1).at[1::2].set(scores)

        for layer in match_heads:
          cache['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_scores', jnp.zeros((1, num_heads, context_len, context_len)))
          cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,match_heads[layer],-1,:].set(match_pattern)

          cache_mask['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_scores', jnp.zeros((1, num_heads, context_len, context_len), dtype=bool))
          cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_scores'].at[0,match_heads[layer],-1,:].set(True)

      for layer in disallow_prev_token_heads:
        heads_to_block = jnp.array(disallow_prev_token_heads[layer])
        cache['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_pre_softmax', jnp.zeros((1, num_heads, context_len, context_len)))
        cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'] = cache['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'].at[0, heads_to_block[:, None], jnp.arange(1, context_len)[None, :], jnp.arange(context_len-1)[None, :]].set(-1e9)

        cache_mask['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('attn_pre_softmax', jnp.zeros((1, num_heads, context_len, context_len), dtype=bool))

        if opts.opto_disallow_prev_token_only_label:
          cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'].at[0, heads_to_block[:, None], jnp.arange(1, context_len, 2)[None, :], (jnp.arange(1, context_len, 2)-1)[None, :]].set(True)
        else:
          cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['attn_pre_softmax'].at[0, heads_to_block[:, None], jnp.arange(1, context_len)[None, :], jnp.arange(context_len-1)[None, :]].set(True)

      if silence_non_label_heads is not None:
        # Set default values if not specified already
        cache['transformer_output']['block_outputs'][-1]['attn_output'].setdefault('attn_pre_softmax', jnp.zeros((1, num_heads, context_len, context_len)))
        cache['transformer_output']['block_outputs'][-1]['attn_output']['attn_pre_softmax'] = cache['transformer_output']['block_outputs'][-1]['attn_output']['attn_pre_softmax'].at[0, silence_non_label_heads, -1, 0::2].set(-1e9)

        cache_mask['transformer_output']['block_outputs'][-1]['attn_output'].setdefault('attn_pre_softmax', jnp.zeros((1, num_heads, context_len, context_len), dtype=bool))
        cache_mask['transformer_output']['block_outputs'][-1]['attn_output']['attn_pre_softmax'] = cache_mask['transformer_output']['block_outputs'][-1]['attn_output']['attn_pre_softmax'].at[0, silence_non_label_heads, -1, 0::2].set(True)

      # All supported manipulations are now specified in the cache and cache_mask,
      # so we can now do our forward pass
      out_cache = model.call_with_all_aux(examples=x, labels=y, key=key, cache=cache, cache_mask=cache_mask)

      # If we're grafting out, we need to pass the relevant layer from this out_cache
      # to the graft out model
      if graft_out_model is not None:
        graft_out_cache, graft_out_cache_mask = get_default_cache_and_mask(len(graft_out_model.transformer.blocks))
        layer = opts.opto_graft_out_model_from_layer
        graft_out_cache['transformer_output']['block_outputs'][layer]['out'] = out_cache['transformer_output']['block_outputs'][layer]['out']
        graft_out_cache_mask['transformer_output']['block_outputs'][layer]['out'] = jnp.ones_like(out_cache['transformer_output']['block_outputs'][layer]['out'], dtype=bool)

        return graft_out_model.call_with_all_aux(examples=x, labels=y, key=key, cache=graft_out_cache, cache_mask=graft_out_cache_mask)
      # Otherwise, we can just return directly
      else:
        return out_cache

    return standard_call
  elif opts.opto_specific_fn == 'perfect_prev_token_input':
    print("WARNING: perfect_prev_token_input function only works for two layer transformers and assumes interleaved input")
    def perfect_prev_token_input_call(model, x, y, key):
      '''
      In this function, we essentially get rid of the Layer 0 circuits, and instead
      separately calculate patterns and values for Layer 1 based on the input embeddings.

      For calculating patterns, we assign the input token embeddings in the right blocks,
      and zero out the rest. For example 0 emb(A) 0 emb(B) emb(A) would be the "ouptut" of
      layer 0 for the purpose of calculating patterns.

      We also clamp the values at the end of Layer 0, to avoid any interference effects.
      We clamp them to be the same as the input embeddings.

      This method also supports the output_from_head argument, which was useful for isolating
      the IH QK match subscircuit.
      '''
      depth = len(model.transformer.blocks)
      context_len = 2*y.shape[0] - 1
      # Currently, all transformers have the same num_heads per layer
      num_heads = model.transformer.blocks[0].attn.num_heads
      per_head_dim = model.transformer.embed_dim // num_heads

      # Used essentially to just get model embeddings
      base_cache = model.call_with_all_aux(examples=x, labels=y, key=key, cache=dict(), cache_mask=dict())

      ### Get patterns
      get_pattern_cache, get_pattern_cache_mask = get_default_cache_and_mask(depth)

      pattern_layer_0_output = base_cache['embedding'].at[1:-1, :].set(base_cache['embedding'][:-2, :])

      get_pattern_cache['transformer_output']['block_outputs'][0]['out'] = pattern_layer_0_output
      get_pattern_cache_mask['transformer_output']['block_outputs'][0]['out'] = jnp.ones_like(base_cache['transformer_output']['block_outputs'][0]['out'], dtype=bool)
      # Now that we've clamped the model at the end of layer 0 (including residual), we can get patterns
      pattern_cache = model.call_with_all_aux(examples=x, labels=y, key=key, cache=get_pattern_cache, cache_mask=get_pattern_cache_mask)
      ### Done getting patterns

      ### Assemble full cache
      cache, cache_mask = get_default_cache_and_mask(depth)
      # First, we clamp the output of layer 0 to be the input embeddings. Since we're also clamping
      # patterns, the output of layer 0 is only used to calculate values. This way, we don't suffer
      # from any interference from Layer 0.

      if output_from_head is not None:
          logits = pattern_cache['transformer_output']['block_outputs'][output_from_head[0]]['attn_output']['attn_pre_softmax'][0, output_from_head[1], -1, jnp.array([1,3])]
          # This could be base_cache or pattern_cache I think, since we're overriding all the relevant values
          cache['out'] = base_cache['out'].at[-1,:].set(-1e9).at[-1,y[:-1]].set(logits)
          cache_mask['out'] = jnp.zeros_like(cache['out']).at[-1,:].set(True)

      else:
        cache['transformer_output']['block_outputs'][0]['out'] = base_cache['embedding']
        cache_mask['transformer_output']['block_outputs'][0]['out'] = jnp.ones_like(base_cache['transformer_output']['block_outputs'][0]['out'], dtype=bool)
        # Next, we clamp the patterns to be those using the perfect previous tokens
        cache['transformer_output']['block_outputs'][1]['attn_output']['attn_scores'] = pattern_cache['transformer_output']['block_outputs'][1]['attn_output']['attn_scores']
        cache_mask['transformer_output']['block_outputs'][1]['attn_output']['attn_scores'] = jnp.ones_like(pattern_cache['transformer_output']['block_outputs'][1]['attn_output']['attn_scores'], dtype=bool)
        
        # Note, this doesn't interfere with other ablations since values happen after the block output of
        # layer 0 getting fixed. attn_scores are also separate from values
        for layer in ablate_heads:
          assert layer == 1, "Can only ablate heads in perfect copy input if they're in layer 1"
          cache['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('v', jnp.zeros((1, num_heads, context_len, per_head_dim)))
          cache['transformer_output']['block_outputs'][layer]['attn_output']['v'] = cache['transformer_output']['block_outputs'][layer]['attn_output']['v'].at[0, ablate_heads[layer], :, :].set(0)

          cache_mask['transformer_output']['block_outputs'][layer]['attn_output'].setdefault('v', jnp.zeros((1, num_heads, context_len, per_head_dim), dtype=bool))
          cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['v'] = cache_mask['transformer_output']['block_outputs'][layer]['attn_output']['v'].at[0, ablate_heads[layer], :, :].set(True)


      return model.call_with_all_aux(examples=x, labels=y, key=key, cache=cache, cache_mask=cache_mask)

    return perfect_prev_token_input_call
  else:
    print('Received unknown opto_specific_fn.')
    raise NotImplementedError
