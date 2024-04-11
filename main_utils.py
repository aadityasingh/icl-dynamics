'''
This file all the argparse arguments for standard training, as well as
some util methods on top of them (to set defaults etc.)
'''

import argparse
import h5py as h5
import numpy as np
import json

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import models
import pdb


def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_from_file', default=None, type=str, help="Config filename. Note that EVERYTHING will be overridden/any other args you pass in will be ignored. Use this option only when you essentially want to duplicate an experiment.")

  parser.add_argument('--base_folder', default='./runs', type=str, help='Path to base folder to store runs')
  parser.add_argument('--use_wandb', action='store_true', help="Defaults to false. If specified, log to wandb.")
  parser.add_argument('--raw_name', action='store_true', help='Defaults to false. When false, the datetime is prepended to the runname for ease of distinguishing. If specified, we will not prepend the datetime.')
  parser.add_argument('--run', default='run', type=str, help="Run name")
  parser.add_argument('--disable_jit', action='store_true', help="If specified, disable jitting. This is redundant to setting JAX_DISABLE_JIT=1 as an env variable.")


  # Model params
  parser.add_argument('--depth', default=2, type=int, help="Transformer depth.")
  parser.add_argument('--not_causal', action='store_true', help="Whether to make transformer causal. Defaults to causal.")
  parser.add_argument('--init_rescale', default=1.0, type=float, help="How much to scale all weight parameter initializations by before training.")
  parser.add_argument('--pos_embedding_type', default='rope', type=str, choices=['ape', 'rope', 'sinusoidal'], help="What type of positional embedding to use. Defaults to 'rope' (rotary positional embeddings).")
  parser.add_argument('--sin_time', type=float, default=10000.0, help="What the timescale used to calculate sinusoidal embeddings (for RoPE) should be.")
  parser.add_argument('--d_model', default=64, type=int, help="Transformer dimension.")
  parser.add_argument('--num_heads', default=8, type=int, help="Number of heads per layer.")
  parser.add_argument('--mlp_ratio', default=None, type=float, help="Expansion factor to use in MLP layers. When set to None, the model becomes attention-only (no MLP layers are used).")
  parser.add_argument('--model_output_classes', default=None, type=int, help="How many output classes for the model. The default is a bit complicated but 'smart' hopefully. If fs_relabel is nonzero, the default is equal to the train fs_relabel. Otherwise, the default is the number of classes present in training (as having more classes would not get signal). There may still be a reason to make model_output_classes > # of train_classes (e.g. if you want to continue training the model on a new set of classes from a checkpoint/continual learning setup). Note if output classes is smaller than a class present in the data, that class will have loss 0 because of how jax.nn.one_hot behaves.")
  parser.add_argument('--no_norm', action='store_true', help="If specified, no norm layers.")

  # Training params
  parser.add_argument('--init_seed', default=5, type=int, help="Random seed for training")
  parser.add_argument('--train_seed', default=0, type=int, help="Random seed for training")
  parser.add_argument('--train_iters', default=int(1e5), type=int, help="# training iters")
  parser.add_argument('--train_bs', default=32, type=int, help="Train batch size. Note that train sequences will change if batch size changes.")
  parser.add_argument('--train_microbs', default=None, type=int, help="Train microbatch size. If specified, the train batch will be split into microbatches of the specified size, and gradients will be averaged before updating. NOTE: if microbatching is being used, the jitting needs to be modified (as we do not want to jit the full train step, as that would jit a big for loop).")
  parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate")
  parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'warmup_decay', 'cosine'], help="Optimizer to use.")
  parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay.")
  parser.add_argument('--load_from_ckpt', default=None, type=str, help="Checkpoint file to load from")
  parser.add_argument('--load_from_ckpt_cfg', default=None, type=str, help="Config of checkpointed model to load. Overrides current model config params as otherwise equinox won't let us load in.")
  parser.add_argument('--ckpt_every', default=None, type=int, help="If specified, how often to checkpoint")
  parser.add_argument('--ckpt_sched', nargs='+', default=None, type=int, help="A list of iterations to checkpoint at. More easily specified with --ckpt_sched_file.")
  parser.add_argument('--ckpt_sched_file', default=None, type=str, help="If specified, a pickle file that when read in, provides a python array with checkpoint iterations. This array will be stored in ckpt_sched. Provides more flexibility for e.g. sampling more checkpoints around phase changes.")
  parser.add_argument('--train_context_len', default=8, type=int, help='Context length during training')
  parser.add_argument('--warmup_steps', default=4_000, type=int, help="Number of warmup steps for learning rate schedule")
  parser.add_argument('--decay_steps', default=1_000_000, type=int, help="Number of decay steps for learning rate schedule")

  # training data mixing params
  parser.add_argument('--mixing_coeffs', nargs='+', default=[], type=float, help="Mixing coefficients for different samplers")
  # per train sampler params
  parser.add_argument('--pt_burstiness', nargs='+', default=[], type=int, help="Burstiness of each training data generator")
  parser.add_argument('--pt_distract', nargs='*', default=None, type=int, help="Whether or not each train data generator has distractors (0 or 1). Defaults to true (1) for all.")
  parser.add_argument('--pt_no_support', nargs='*', default=None, type=int, help="Whether or not each train data generator has no_support (0 or 1). Defaults to false (0) for all.")
  parser.add_argument('--pt_unique_rest', nargs='*', default=None, type=int, help="Whether or not each train data generator has unique_rest (0 or 1). Defaults to false (0) for all.")


  # eval params
  parser.add_argument('--eval_seed', default=1, type=int, help="Random seed for eval")
  parser.add_argument('--eval_iters', default=int(1e3), type=int, help="# eval sequences")
  parser.add_argument('--eval_bs', default=int(1e3), type=int, help="eval batch size for feeding to model. Note sequences are generated altogether so sequences do not depend on the batch size here (they do in training since batches are sampled sequentially).")
  parser.add_argument('--eval_every', default=None, type=int, help="How often to evaluate training")
  parser.add_argument('--eval_sched', nargs='+', default=None, type=int, help="A list of iterations to evaluate at. More easily specified with --eval_sched_file.")
  parser.add_argument('--eval_sched_file', default=None, type=str, help="If specified, a pickle file that when read in, provides a python array with eval iterations. This array will be stored in eval_sched.")
  parser.add_argument('--save_eval_data', default=None, type=str, help="If specified, file to store eval data to. Gets placed in runs/\{opts.run\}.")
  parser.add_argument('--load_eval_data', default=None, type=str, help="If specified, an h5 file that eval data can be loaded from. Should have a format compatible with save_eval_data")

  # We want the option to have many different evaluators
  # We do this a bit messily by taking in a bunch of lists
  # per evaluator params
  # This could be improved upon in the future
  parser.add_argument('--pe_names', nargs='*', default=[], type=str, help="How many evaluators to run")
  parser.add_argument('--pe_classes', nargs='*', default=[], type=str, help="Which class set to draw from. Choose from ['train', 'val', 'test', 'all'] for each.")
  parser.add_argument('--pe_exemplars', nargs='*', default=[], type=str, help="Which exemplar set to draw from. Choose from ['train', 'val', 'test', 'all'] for each. Note if the class for this eval sampler is not train, then this will get overriden to all.")
  parser.add_argument('--pe_burstiness', nargs='*', default=[], type=int, help="Burstiness of each evaluator [# examples of query class in context].")
  parser.add_argument('--pe_noise_scale', nargs='*', default=None, type=float, help="How much Gaussian noise to add to exemplars from each evaluator. Defaults to noise_scale_train for each.")
  parser.add_argument('--pe_distract', nargs='*', default=None, type=int, help="Whether or not each evaluator has distractors (0 or 1). Defaults to true (1) for all.")
  parser.add_argument('--pe_no_support', nargs='*', default=None, type=int, help="Whether or not each evaluator has no_support (0 or 1). Defaults to false (0) for all.")
  parser.add_argument('--pe_unique_rest', nargs='*', default=None, type=int, help="Whether or not each evaluator has unique_rest (0 or 1). Defaults to false (0) for all.")
  parser.add_argument('--pe_fs_relabel_scheme', nargs='*', default=None, type=str, help="Whether or not to relabel bursty and distractor samples randomly to class pairs in the given split. Defaults to 'None' for all, which corresponds to no relabeling. If specified, each element should be in ['None', 'train', 'val', 'test', 'all', '01']. '01' is to allow for eval relabeling to classes 0 and 1 even when training was not performed with relabeling.")
  parser.add_argument('--pe_context_len', nargs='*', default=None, type=int, help="Context length of each evaluator. Defaults to train context len for each evaluator")
  parser.add_argument('--pe_assign_query_label_random', nargs='*', default=None, type=int, help="Whether or not to assign query label randomly to an in-context exemplar. Defaults to false (0) for all.")

  # Data params
  parser.add_argument('--data_type', type=str, default='file', choices=['file', 'onehot'], help='How to generate data. Current choices are from file or onehot')
  parser.add_argument('--data_file', default='omniglot_features.h5', help='File to get data from')
  parser.add_argument('--data_path_in_file', default='resnet18/224/feat')
  parser.add_argument('--class_split', nargs=3, type=int, default=[1600,23,0], help='train/val/test class split, in # classes')
  parser.add_argument('--exemplar_split', nargs=3, type=int, default=[5,0,0], help='train/val/test class split, in # exemplars')
  parser.add_argument('--fs_relabel', default=0, type=int, help="If >0, relabels bursty and distractor samples randomly to classes in the range 0..fs_relabel. Otherwise, no relabeling is performed.")
  parser.add_argument('--fs_relabel_split', nargs=3, type=float, default=None, help='Splits all possible relabelings into train, validation, and test splits. Only applies if fs_relabel > 0. If not specified, defaults to all relabelings being seen at training. Ensures that all labels are still seen during training (otherwise would have OOD problems). Relabelings are split based on pair (order is ignored) so there are fs_relabel * (fs_relabel-1)/2 total relabelings.')
  parser.add_argument('--fs_relabel_split_seed', type=int, default=20, help="Random seed to use when drawing fs relabel split")
  parser.add_argument('--noise_scale_train', default=0.0, type=float, help="If >0, adds Gaussian noise to training data with given scale")
  parser.add_argument('--zipf_alpha', default=0.0, type=float, help="Sample training classes from a Zipfian distribution with parameter alpha. Defaults to 0 (uniform class distribution).")
  parser.add_argument('--match_query_and_distractors', action='store_true', help="If true, match query exemplar and bursty examplars in context, as well as distractor exemplars")

  return parser


def check_opts(opts):
  assert opts.train_microbs <= opts.train_bs, 'Microbatch size must be smaller than batch size'
  assert opts.train_bs % opts.train_microbs == 0, 'Train batch size should be a multiple of microbatch size'
  if opts.pe_fs_relabel_scheme is not None:
    for s in opts.pe_fs_relabel_scheme:
      assert s in ['None', 'train', 'val', 'test', 'all', '01', 'flip'], "Evaluator relabel splits must come from ['None', 'train', 'val', 'test', 'all', '01']"
  assert opts.init_seed != opts.train_seed, 'Init and train seed must be diff'
  assert opts.train_seed != opts.eval_seed, 'Train and eval seed must be diff'
  assert opts.init_seed != opts.eval_seed, 'Init and eval seed must be diff'
  assert (opts.eval_every is not None) or (opts.eval_sched is not None) or (opts.eval_sched_file is not None), "Some method of evaluating must be specified atm"
  assert (opts.eval_every is None) + (opts.eval_sched is None) + (opts.eval_sched_file is None) == 2, 'Exactly one way of eval iters should be specified'
  assert (opts.ckpt_every is None) + (opts.ckpt_sched is None) + (opts.ckpt_sched_file is None) >= 2, 'At most one way of ckpt iters should be specified'


def get_opts_from_json_file(fname):
  parser = create_parser()
  opts, unknown = parser.parse_known_args()
  with open(fname, 'r') as f:
    vars(opts).update(json.load(f))
  return opts


def get_data_from_opts(opts):
  if opts.data_type == 'file':
    with h5.File(opts.data_file, 'r') as f:
      return jnp.array(f[opts.data_path_in_file])
  elif opts.data_type == 'onehot':
    num_classes = sum(opts.class_split)
    # One exemplar per class:
    return jnp.expand_dims(jnp.eye(num_classes), 1)
  else:
    raise NotImplementedError


def get_splits_from_opts(opts, data_shape):
  print("data shape 0:", data_shape[0], "class split:", opts.class_split)
  assert data_shape[0] == sum(opts.class_split)
  assert data_shape[1] == sum(opts.exemplar_split)

  retval = dict()

  retval['class'] = {'train': jnp.arange(opts.class_split[0]),
                      'val': jnp.arange(opts.class_split[0], opts.class_split[0]+opts.class_split[1]),
                      'test': jnp.arange(opts.class_split[0]+opts.class_split[1], data_shape[0]),
                      'all': jnp.arange(data_shape[0])}
  retval['exemplar'] = {'train': jnp.arange(opts.exemplar_split[0]),
                        'val': jnp.arange(opts.exemplar_split[0], opts.exemplar_split[0]+opts.exemplar_split[1]),
                        'test': jnp.arange(opts.exemplar_split[0]+opts.exemplar_split[1], data_shape[1]),
                        'all': jnp.arange(data_shape[1])}

  retval['relabeling'] = {'train': None, 'val': None, 'test': None, 'all': None, 
                          'None': None, '01': jnp.array([[0,1]]), 'flip': 'flip'}
  if opts.fs_relabel > 0:
    if opts.pe_fs_relabel_scheme is None:
      print("Changing default evaluator relabeling to train relabeling")
      opts.pe_fs_relabel_scheme = ['train']*len(opts.pe_names)
    temp_grid = np.broadcast_to(np.arange(opts.fs_relabel), (opts.fs_relabel, opts.fs_relabel))
    all_label_pairs = jnp.vstack([temp_grid.T.reshape(-1), temp_grid.reshape(-1)]).T
    all_relabelings = all_label_pairs[all_label_pairs[:, 0] < all_label_pairs[:, 1]]
    retval['relabeling']['all'] = all_relabelings
    if opts.fs_relabel_split is not None:
      split = np.array(opts.fs_relabel_split)
      split = np.cumsum(np.flip((all_relabelings.shape[0] * split/split.sum()).astype(int)))
      perm_relabelings = jax.random.permutation(jax.random.PRNGKey(opts.fs_relabel_split_seed), all_relabelings, axis=0, independent=False)
      retval['relabeling']['test'] = perm_relabelings[:split[0]]
      retval['relabeling']['val'] = perm_relabelings[split[0]:split[1]]
      retval['relabeling']['train'] = perm_relabelings[split[1]:]
      retry = 0
      seed = jax.random.PRNGKey(opts.fs_relabel_split_seed)
      while (len(jnp.unique(retval['relabeling']['train'])) < opts.fs_relabel) and (retry < 10):
        print("Uh oh! Not all labels encountering during training. Retrying relabel split... {}".format(retry))
        seed, to_use = jax.random.split(seed)
        perm_relabelings = jax.random.permutation(jax.random.PRNGKey(opts.fs_relabel_split_seed), all_relabelings, axis=0, independent=False)
        retval['relabeling']['test'] = perm_relabelings[:split[0]]
        retval['relabeling']['val'] = perm_relabelings[split[0]:split[1]]
        retval['relabeling']['train'] = perm_relabelings[split[1]:]
        retry += 1
      if retry == 10:
        print("Couldnt't find a relabeling split with all labels appearing during training")
        pdb.set_trace()
      print("Splitting fs relabelings with train: {}, val: {}, test: {}".format(len(retval['relabeling']['train']), len(retval['relabeling']['val']), len(retval['relabeling']['test'])))
    else:
      print("Defaulting fs relabel split to all relabelings appearing in training")
      retval['relabeling']['train'] = all_relabelings
      retval['relabeling']['val'] = jnp.array([])
      retval['relabeling']['test'] = jnp.array([])
  return retval


def scale_model_init(model, scale=1.0):
  '''
  This method will scale all weight matrices down by a given factor
  '''
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [x.weight
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  weights = get_weights(model)
  new_weights = [weight * scale for weight in weights]
  new_model = eqx.tree_at(get_weights, model, [weight * scale for weight in weights])
  return new_model


def get_model_from_opts(opts, input_shape=None):
  if input_shape is None:
    if opts.data_type == 'file':
      with h5.File(opts.data_file, 'r') as f:
        input_shape = (f[opts.data_path_in_file].shape[-1],)
    elif opts.data_type == 'onehot':
      input_shape = (sum(opts.class_split),)
    else:
      raise NotImplementedError
  opts.init_seed = jax.random.PRNGKey(opts.init_seed)
  model = models.SequenceClassifier(
    example_shape=input_shape,
    example_type=opts.data_type,
    num_classes=opts.model_output_classes,
    embed_dim=opts.d_model,
    key=opts.init_seed,
    depth=opts.depth,
    num_heads=opts.num_heads,
    mlp_ratio=opts.mlp_ratio,
    causal=(not opts.not_causal),
    pos_embedding_type=opts.pos_embedding_type,
    sin_time=opts.sin_time,
    norm_layer=(eqx.nn.Identity if opts.no_norm else eqx.nn.LayerNorm)
  )
  model = scale_model_init(model, opts.init_rescale)
  return model


def get_optimizer_from_opts(opts):
  if opts.optimizer == 'adam':
    optimizer = optax.adam(learning_rate=opts.lr)
  elif opts.optimizer == 'sgd':
    optimizer = optax.sgd(learning_rate=opts.lr)
  elif opts.optimizer == 'warmup_decay':
    linear_warmup = optax.polynomial_schedule(
      init_value=1e-5,
      end_value=opts.lr,
      power=1.0,
      transition_steps=opts.warmup_steps)
    sqrt_decay = optax.polynomial_schedule(
      init_value=opts.lr,
      end_value=1e-5,
      power=0.5,
      transition_steps=opts.decay_steps)
    lr_schedule = optax.join_schedules(
      schedules=[linear_warmup, sqrt_decay],
      boundaries=[opts.warmup_steps])
    optimizer = optax.chain(
      optax.clip_by_global_norm(5.0),
      optax.scale_by_adam(),
      optax.scale_by_schedule(lr_schedule),)
  elif opts.optimizer == 'cosine':
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-6,
                                                     peak_value=opts.lr,
                                                     warmup_steps=opts.warmup_steps,
                                                     decay_steps=opts.decay_steps)
    optimizer = optax.chain(
      optax.scale_by_adam(),
      optax.scale_by_schedule(lr_schedule),)
  else:
    raise NotImplementedError
  return optimizer
