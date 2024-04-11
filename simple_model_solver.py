'''
This is the file used to play around with toy models.

The `train` function accepts a dictionary containing:
- setup: train setup, at minimum consists of:
  - init: A dictionary of initial param values
  - true: A dictionary of true parameter values
  - loss_fn: A loss function that takes as input the current param values
        and outputs a scalar loss. Assumed to be JIT-able.
  - track_fns: Progress measures to compute on the parameter values
- clamp_fn: fn that maps setup + params -> updated params
- Standard gradient descent params:
  - max_iters
  - lr: learning rate
  - thresh: absolute threshold for loss for early termination

We found this breakdown to provide a very modular and functional way to iterate
on toy models (allowing for both clamping and progress measures). For usage in
other models, the `train` function should be usable without modification.

Different setups can be constructed in a functional way, see `make_rank_1_three_vecs_setup`
'''

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from functools import partial
import argparse

def train(setup, clamp_fn = (lambda s, p: p), max_iters = 1000, lr = 1e-3, thresh=1e-5):
  '''
  Runs gradient descent on a given setup, applying clamp_fn at every iteration
  '''
  loss_and_grad_fn = jax.jit(jax.value_and_grad(setup['loss_fn']))

  tracked_vars = {'Train loss': []}
  for fn in setup['track_fns']:
    tracked_vars[fn] = []

  params = setup['init']
  params = clamp_fn(setup, params)
  for i in tqdm(range(max_iters)):
    loss, grad = loss_and_grad_fn(params)
    tracked_vars['Train loss'].append(loss)
    for fn in setup['track_fns']:
      tracked_vars[fn].append(setup['track_fns'][fn](params))

    if loss < thresh:
      print("Terminating early since loss below threshold at iter {}", i)
      break

    params = jax.tree_map(lambda x, y: x - lr*y, params, grad)
    params = clamp_fn(setup, params)
  return tracked_vars


def make_rank_1_three_vecs_setup(dims=[10, 10, 10], seed=0):
  assert len(dims) == 3
  seeds = jax.random.split(jax.random.PRNGKey(seed), 6)
  true_a = jax.random.normal(seeds[0], shape=(dims[0],))
  true_b = jax.random.normal(seeds[1], shape=(dims[1],))
  true_c = jax.random.normal(seeds[2], shape=(dims[2],))
  true = jnp.einsum('i,j,k->ijk', true_a, true_b, true_c)

  def loss(params):
    err = (true - jnp.einsum('i,j,k->ijk', params['a'], params['b'], params['c']))
    return 0.5*jnp.mean(err**2)

  init_a = jax.random.normal(seeds[3], shape=(dims[0],))
  init_b = jax.random.normal(seeds[4], shape=(dims[1],))
  init_c = jax.random.normal(seeds[5], shape=(dims[2],))

  norm_true_a = true_a/jnp.linalg.norm(true_a)
  def track_a_acc(params):
    norm_a = params['a']/jnp.linalg.norm(params['a'])
    return 1-(norm_a@norm_true_a)**2

  # Progress measures version:
  def track_a_prog(params):
    '''
    Measures error as if b, c clamped to true values

    Not presented in paper due to identifiability/rotation issue.
    '''
    err = true - jnp.einsum('i,j,k->ijk', params['a'], true_b, true_c)
    return 0.5*jnp.mean(err**2)

  def track_a_prog2(params):
    '''
    Same as track_a_prog, except takes into account rotations (in this case, negation)
    '''
    return min(track_a_prog(params), track_a_prog(jax.tree_map(lambda x: -x, params)))

  # New lines used in track_fns since those are directly used as plot titles
  # Note, we didn't include track_*_acc here, since we didn't actually plot that
  # for the paper figures.
  return {'init': {'a': init_a, 'b': init_b, 'c': init_c}, 
      'true': {'a': true_a, 'b': true_b, 'c': true_c},
      'loss_fn': loss, 
      'track_fns': {'Progress measure 1:\n'
                      r'$1 - d_{cos}(\mathbf{a}^{true}, \mathbf{a})^2$': track_a_acc,
                      'Progress measure 2:\n'
                      r'Loss with $\mathbf{b}=\pm \mathbf{b}^{true}, \mathbf{c}=\pm \mathbf{c}^{true}$': track_a_prog2, 
                      }
      }


def clamp_rank_1(setup, params, clamp=dict()):
  retval = dict()
  for c in params:
    if c in clamp:
      retval[c] = jnp.where(jnp.arange(params[c].shape[0]) < clamp[c], setup['true'][c], params[c])
    else:
      retval[c] = params[c]
  return retval


def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save', default=None, type=str, help='Path to save figure at. If None, show figure.')
  parser.add_argument('--separate_rows', action='store_true', help="If specified, plots each conditions' training run in a separate row. Defaults to false")
  parser.add_argument('--only_loss', action='store_true', help="If specified, only plots the loss, no progress measures. If specified, no_collapse is ignored.")

  parser.add_argument('--condition_clamp_c', action='store_true', help='If specified, adds a condition for clamping just c (instead of b and c).')
  parser.add_argument('--dim', default=20, type=int, help="Dimension for toy model vectors. Defaults to 20, which was used for the paper.")
  parser.add_argument('--seed', default=1, type=int, help="Seed for toy model (true and init params). Defaults to 1, which was used for the paper.")
  return parser


if __name__ == "__main__":
  parser = create_parser()
  opts = parser.parse_args()
  rank_1_three_interacting = make_rank_1_three_vecs_setup(dims=[opts.dim]*3, seed=opts.seed)

  conditions = {'None': {'setup': rank_1_three_interacting}}

  if opts.condition_clamp_c:
    conditions[r'$\mathbf{c}$'] = {'setup': rank_1_three_interacting, 'clamp_fn': partial(clamp_rank_1, clamp={'c': opts.dim})}

  conditions[r'$\mathbf{b}, \mathbf{c}$'] = {'setup': rank_1_three_interacting, 'clamp_fn': partial(clamp_rank_1, clamp={'b': opts.dim, 'c': opts.dim})}

  all_tracked = dict()
  for k in conditions:
    print("Training condition", k)
    all_tracked[k] = train(**conditions[k], lr=1, max_iters=1000, thresh=0)


  if opts.only_loss:
    # Used to make paper figure
    colors = np.array([[0,0,0], [55,126,184], [152,78,163]])/256
    matplotlib.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for i, k in enumerate(all_tracked):
      axs.plot(all_tracked[k]['Train loss'], label=k, color=colors[i], lw=3)
    axs.legend(title='"Clamped" variable')
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Loss')
    axs.set_title('Loss dynamics of toy model when "clamping" variables')
  elif not opts.separate_rows:
    print("WARNING: Assumes only two conditions, otherwise more colors need to be specified")
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(12, 3)
    colors = np.array([[0,0,0], [152,78,163]])/256
    for i, k in enumerate(all_tracked):
      for j, f in enumerate(all_tracked[k]):
        axs[j].plot(all_tracked[k][f], label=k, color=colors[i])
        axs[j].set_title(f)
    
    for j in range(3):
      axs[j].set_xlabel("Iterations")
    axs[0].legend(title='Clamped variable')
    plt.tight_layout()
  else:
    print("WARNING: This condition isn't well supported, use with care")
    columns = len(all_tracked)
    rows = 7 # hardcoded
    fig, axs = plt.subplots(rows, columns, sharex='col', sharey='row')
    fig.set_size_inches(4*columns, 2*rows)
    for i, k in enumerate(all_tracked):
      for j, f in enumerate(all_tracked[k]):
        axs[j, i].plot(all_tracked[k][f])
        axs[j, i].set_title(f)
      axs[0, i].set_title('Condition: {}'.format(k))

if opts.save is not None:
  fig.savefig(opts.save)
else:
  plt.show()
