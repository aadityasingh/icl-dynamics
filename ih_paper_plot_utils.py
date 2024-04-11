'''
This file contains the relevant helper functions for generating paper plots.

It was created to make the ih_paper_plots.ipynb notebook more streamlined.
'''

import os
import numpy as np
import h5py as h5
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
import argparse
import pdb
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import samplers
import models
import main
import main_utils
import opto
import pickle as pkl


def get_closest_inds(search_for, search_in):
  '''
  Returns the unique indices in search_in that are closest to the elements of search_for
  '''
  iters_inds = np.argmin(np.abs(search_for[:, None] - search_in[None, :]), axis=1)
  return np.sort(np.unique(iters_inds))


def scientific_notation_ticks(axs, xaxis=True, yaxis=False):
  formatter = ScalarFormatter(useMathText=True)
  formatter.set_scientific(True)
  formatter.set_powerlimits((-1,1))
  if xaxis:
    axs.xaxis.set_major_formatter(formatter)
  if yaxis:
    axs.yaxis.set_major_formatter(formatter)


def make_forward_fn(options, default_fn=opto.default_model_fwd_fn):
  call_fn = opto.make_fn_from_opts(options, default_fn=default_fn)

  def forward_fn(model, x, y, key):
    '''
    Returns dict of:
      activations: bs x depth x num_heads x context_lengthQ x context_lengthK
      logits: bs x num_classes
      loss: bs x 1
    '''
    keys = jax.random.split(key, x.shape[0])
    all_activations = jax.vmap(partial(call_fn, model=model))(x=x, y=y, key=keys)
    head_act = jnp.stack([l['attn_output']['attn_scores'][:, 0, :, :, :] for l in all_activations['transformer_output']['block_outputs']])
    query_ce = main.ce(all_activations['out'][:, -1, :], y[:, -1])
    output_prob = jnp.einsum('bc,bc->b', 
                              jax.nn.softmax(all_activations['out'][:, -1, :]), 
                              jax.nn.one_hot(y[:, -1], all_activations['out'].shape[-1]))
    output_acc = (jnp.argmax(all_activations['out'][:, -1, :], axis=-1) == y[:,-1])

    in_context_mask = jnp.sum(jax.nn.one_hot(y[:, :-1], all_activations['out'].shape[-1]), axis=1) > 0
    in_context_pred_y = in_context_mask*all_activations['out'][:, -1, :] - (1-in_context_mask)*1e20
    fsl_loss = main.ce(in_context_pred_y, y[:, -1])
    fsl_output_prob = jnp.einsum('bc,bc->b', 
                                jax.nn.softmax(in_context_pred_y), 
                                jax.nn.one_hot(y[:, -1], in_context_pred_y.shape[-1]))

    return {'activations': jnp.transpose(head_act, (1,0,2,3,4)), 
            'logits': all_activations['out'][:, -1, :],
            'prob': output_prob,
            'acc': output_acc,
            'in_context_loss': fsl_loss,
            'in_context_prob': fsl_output_prob,
            'loss': query_ce}

  return eqx.filter_jit(forward_fn)