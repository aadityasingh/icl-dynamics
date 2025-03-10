'''
A somewhat custom library for plotting various results. This file is feature
"overcomplete" -- it contains may extra features beyond those specified in the
papers. Specifically, a lot of our early investigations focused on how individual
classes/training data sequences get learned. While this is still an interesting
direction, we got mixed results (basically, all the things that may affect this
learning kind of did in some settings, so no clear simplification/insight).

The idea with this file is to standardize the format of plotting. We do this by
having plotting modes, that operate on the same primitives -- a set of checkpoints
specified by an iteration range + data to run them on. The file is also "opto-compatible"
enabling quick visualization of different progress measures (see `make_forward_fn`). 

The Induction heads paper reproduction doesn't rely on this file, but I suspect the code 
there could be simplified by using this file. This file was only included as a dependency 
for the coopetition paper.

To specify data, we offer a few modes:
- class pairs and relabel pairs
- load from eval file
- the file could also be made to wrap `samplers.py` in the future

In terms of plotting modes, each mode consists of an update and plot function
The update function gets called every model evaluation and is meant to maintain an 
updating state needed for that plot mode. Note that `make_forward_fn` returns a lot of
things, and most mode's update functions discard a lot of it -- this is to prevent OOMs,
and works thanks to Python's garbage collector.

The plot function then takes the output of the update function and plots it.
We also expose top level arguments to these plot functions, allowing for plot
customizability (e.g. custom coloring).

Finally, one reason to have this as a file instead of a notebook was to allow launching
these jobs easily on a cluster which allows for quicker iteration. Many of these jobs are
not "cheap", as they require forward passes on thousands of sequences for hundreds of
checkpoints.

Disclaimer: the over-completeness might make it a bit harder to use for newcomers, apologies!
'''

import os
import numpy as np
import h5py as h5
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
import plot_utils
import opto
import pickle as pkl


normalize_rows = lambda x: x/np.sqrt(np.sum(x**2, axis=1))[:,None]


def coloring_helper(color_by, data):
  '''
  Helper method for determining line colors and configuring colorbar
  for different color modes.

  Args:
  color_by: str from args
  data: can be directly passed from a plot fn

  Returns: a dictionary with
    sm: ScalarMappable used for colorbar
    widths: either [] or [0.1], with the latter to make space for colorbar
    num_columns: either 0 or 1, with the latter indicating a cbar_ax should be made
    labels: Labels for colorbar, only used for discrete case
    inds: indices into colors for discrete coloring case.
  '''
  retval = dict(sm=None, widths=[], num_columns=0, labels=None, inds=None, colors_by_label=None)
  if color_by == 'default':
    batch_size = data['correct_ind'].shape[0]
    retval['colors_by_label'] = np.array([[0,0,1,1]])
    retval['colors'] = np.broadcast_to(np.array([0,0,1,1]).astype(float), (batch_size, 4))
    retval['labels'] = ['None']
    retval['inds'] = np.zeros(batch_size, dtype=int)
  elif color_by == 'similarity':
    sims = np.sum(normalize_rows(data['examples'][:, 0, :]) * normalize_rows(data['examples'][:, 1, :]), axis=1)
    norm = mcolors.Normalize(vmin=np.min(sims), vmax=np.max(sims))
    normalized_sims = norm(sims)
    retval['colors'] = plt.cm.viridis(normalized_sims)

    retval['sm'] = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    retval['sm'].set_array([])
    retval['widths'] = [0.1]
    retval['num_columns'] = 1
  elif color_by in ['correct_ind', 'label_pair', 'relabel', 'class_pair', 'output', 'output_and_ind', 'metadata']:
    if color_by == 'correct_ind':
      to_use = data['correct_ind']
    elif color_by == 'label_pair':
      to_use = np.where((data['labels'][:, 0] < data['labels'][:, 1])[:,None], data['labels'][:, :2], data['labels'][:, 1::-1])
    elif color_by == 'relabel':
      to_use = data['labels']
    elif color_by == 'class_pair':
      to_use = data['class_pairs'][:,:2]
    elif color_by == 'output':
      to_use = data['labels'][:, -1]
    elif color_by == 'output_and_ind':
      to_use = np.stack([data['labels'][:, -1], data['correct_ind']], axis=1)
    elif color_by == 'metadata':
      to_use = data['metadata_ind']

    labelings, retval['inds'] = np.unique(to_use, axis=0, return_inverse=True)
    assert labelings.shape[0] <= 10, 'discrete coloring only supported for 10 unique ids'

    cmap = plt.get_cmap('tab10')

    retval['colors_by_label'] = cmap(np.arange(len(labelings)))
    retval['colors'] = cmap(retval['inds'])

    norm = mcolors.BoundaryNorm(np.arange(-0.5, labelings.shape[0], 1), labelings.shape[0])
    if color_by == 'metadata':
      retval['labels'] = [data['metadata_labels'][i] for i in labelings]
    else:
      retval['labels'] = [str(l) for l in labelings]

    retval['sm'] = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    retval['sm'].set_array([])
    retval['widths'] = [0.1]
    retval['num_columns'] = 1
  else:
    print("Unrecognized color_by")
    raise NotImplementedError

  return retval


def add_cbar_from_color_info(fig, axs, color_info, color_by):
  '''
  Helper method for adding a color bar
  '''
  if color_info['sm'] is not None:
    # Remove all axes in the final column and replace with a single axis to draw cbar in
    for ax in axs[:, -1]:
      ax.remove()
    colorbar_ax = fig.add_subplot(axs[0,0].get_gridspec()[:, -1])
    
    if color_info['labels'] is not None:
      cbar = plt.colorbar(color_info['sm'], ticks=np.arange(len(color_info['labels'])), cax=colorbar_ax, label=color_by)
      cbar.ax.set_yticklabels(color_info['labels'])
    else:
      plt.colorbar(color_info['sm'], label=color_by, cax=colorbar_ax)


def update_biases_over_time(info, iternum, results, model, data):
  if info == dict():
    info['iters'] = []
    info['output_bias'] = []
  info['output_bias'].append(model.unembed.bias)
  info['iters'].append(iternum)


def plot_biases_over_time(info, color_by, data, metric_curve, plot_opts, 
                          save_path, integral_metric_starts_to_losses):
  output_bias = jnp.stack(info['output_bias'], axis=1)
  iters = jnp.array(info['iters'])

  fig, axs = plt.subplots()
  fig.set_size_inches(10,5)
  for i, row in enumerate(output_bias):
    axs.plot(iters, row, label=str(i))

  axs.legend()

  fig.suptitle('Unembed biases over time', wrap=True)
  
  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info, 'color_info': color_info}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  plt.close(fig)


def update_metric_vs_sim(info, iternum, results, model, data):
  if info == dict():
    info['iters'] = []
    info['loss'] = []
  info['iters'].append(iternum)
  info['loss'].append(results['loss'])


def plot_metric_vs_sim(info, color_by, data, metric_curve, plot_opts, 
                        save_path, integral_metric_starts_to_losses):
  # bs x time
  loss = jnp.stack(info['loss'], axis=1)
  iters = jnp.array(info['iters'])

  num_columns = len(integral_metric_starts_to_losses)
  widths = [1]*num_columns

  color_info = coloring_helper(color_by, data)
  c = color_info['colors']
  c[:, -1] = 0.5
  widths = widths + color_info['widths']
  num_columns += color_info['num_columns']

  fig, axs = plt.subplots(1, num_columns, squeeze=False, gridspec_kw={'width_ratios': widths})
  fig.set_size_inches(10*sum(widths),10)
  if color_info['labels'] is not None:
    cbar = plt.colorbar(color_info['sm'], ticks=np.arange(len(color_info['labels'])), cax=axs[0, -1], label=color_by)
    cbar.ax.set_yticklabels(color_info['labels'])
  elif color_info['sm'] is not None:
    plt.colorbar(color_info['sm'], label=color_by, cax=axs[0, -1])

  for i, start in enumerate(integral_metric_starts_to_losses):
    select = iters > start
    int_metric = jnp.trapz(loss[:, select], x=iters[select], axis=1)
    sims = np.sum(normalize_rows(data['examples'][:, 0, :]) * normalize_rows(data['examples'][:, 1, :]), axis=1)
    axs[0,i].scatter(sims, int_metric, c=c)
    axs[0,i].set_title("Integral metric calculated starting at {}, where avg_loss is {:.3f}".format(start, integral_metric_starts_to_losses[start]))
    if plot_opts.save_problem_points:
      worst_points = [','.join(['class0','class1','label0','label1','correct_ind','max_loss'])]
      sorted_by_metric = jnp.argsort(int_metric)
      for j in range(1,1+plot_opts.save_problem_points):
        worst_points.append('{},{},{},{},{},{}'.format(data['class_pairs'][sorted_by_metric[-j], 0],
                                                    data['class_pairs'][sorted_by_metric[-j], 1],
                                                    data['labels'][sorted_by_metric[-j], 0],
                                                    data['labels'][sorted_by_metric[-j], 1],
                                                    data['correct_ind'][sorted_by_metric[-j]],
                                                    jnp.max(loss[sorted_by_metric[-j], select])))
      with open(save_path+'_bad_points_{}_{:.3f}.csv'.format(start,integral_metric_starts_to_losses[start]), 'w') as f:
        f.write('\n'.join(worst_points))

  opt_dict = dict(vars(plot_opts))
  opt_dict['plots'] = 'metric_vs_sim'
  opt_dict['color_by'] = color_by
  fig.suptitle(vars(plot_opts), wrap=True)
  
  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'_{}'.format(color_by)+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info, 'color_info': color_info}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  plt.close(fig)


def update_metric_by_color_by(info, iternum, results, model, data):
  if info == dict():
    info['iters'] = []
    info['loss'] = []
    info['prob'] = []
    info['acc'] = []
    info['in_context_loss'] = []
    info['in_context_prob'] = []
    info['in_context_acc'] = []
  info['iters'].append(iternum)
  info['loss'].append(results['loss'])
  info['prob'].append(results['prob'])
  info['acc'].append(results['acc'])
  info['in_context_loss'].append(results['in_context_loss'])
  info['in_context_prob'].append(results['in_context_prob'])
  info['in_context_acc'].append(results['in_context_acc'])


def plot_metric_by_color_by(info, color_by, data, metric_curve, plot_opts, 
                          save_path, integral_metric_starts_to_losses=None):
  metric = jnp.stack(info[plot_opts.baseline_metric], axis=1)
  iters = jnp.array(info['iters'])
  print(jnp.mean(metric[:, -1], axis=0))

  assert color_by in ['default', 'correct_ind', 'label_pair', 'relabel', 'class_pair', 'output', 'output_and_ind', 'metadata']

  color_info = coloring_helper(color_by, data)

  cmap = plt.get_cmap('tab20')

  fig, axs = plt.subplots(1+len(color_info['labels']), 1, sharex=True, sharey=True)
  fig.set_size_inches(10, 5*(1+len(color_info['labels'])))
  axs[0].set_xlim(info['iters'][0]-10, info['iters'][-1]+10)
  axs[0].set_ylim(plot_opts.metric_range[0], plot_opts.metric_range[1])

  means = []
  for i in range(len(color_info['labels'])):
    to_use_metric = metric[color_info['inds'] == i]
    means.append(jnp.mean(to_use_metric, axis=0))
    color = cmap(2*i)
    transparent = np.array(cmap(2*i+1))
    transparent[-1] = min(1,15/to_use_metric.shape[0])
    if not plot_opts.only_plot_avg:
      axs[i+1].add_collection(plot_utils.make_line_collection(iters, to_use_metric, colors=[transparent]))
    axs[i+1].plot(iters, means[i], c=color)
    axs[i+1].plot(metric_curve['x'], metric_curve['y'], c='k', ls='--')
    axs[i+1].set_title(color_info['labels'][i])

  for i, m in enumerate(means):
    axs[0].plot(iters, means[i], c=cmap(2*i), label=color_info['labels'][i])
    print(iters[-1], color_info['labels'][i], means[i][-1])
  axs[0].plot(metric_curve['x'], metric_curve['y'], c='k', ls='--', label='avg {}'.format(plot_opts.baseline_metric))
  if 'None' not in color_info['labels']:
    axs[0].legend(title=color_by, bbox_to_anchor=(1,1), loc='upper left')

  opt_dict = dict(vars(plot_opts))
  opt_dict['plots'] = 'metric_by_color_by'
  opt_dict['color_by'] = color_by
  fig.suptitle(vars(plot_opts), wrap=True)
  plt.tight_layout()
  
  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'_{}'.format(color_by)+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info, 'color_info': color_info}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  plt.close(fig)


def update_prev_token_over_time(info, iternum, results, model, data):
  if info == dict():
    info['iters'] = []
    info['loss'] = []
    info['prob'] = []
    info['activations'] = []
    info['in_context_prob'] = []

  inds = np.arange(1,results['activations'].shape[3])
  prev_inds = np.arange(results['activations'].shape[3]-1)
  info['activations'].append(results['activations'][:,0,:,inds,prev_inds].transpose(1,2,0))
  info['loss'].append(results['loss'])
  info['prob'].append(results['prob'])
  info['in_context_prob'].append(results['in_context_prob'])
  info['iters'].append(iternum)


def plot_prev_token_over_time(info, color_by, data, metric_curve, plot_opts, 
                              save_path, integral_metric_starts_to_losses=None):
  # Time x bs x num_heads x context length
  activations = jnp.stack(info['activations'])
  # Flip to context length x num_heads x bs x time
  activations = jnp.transpose(activations, (3,2,1,0))

  if not plot_opts.use_raw_prev_tok_score:
    baseline = (1 - activations) / (1+np.arange(activations.shape[0]))[:,None,None,None]
    activations = activations - baseline

  # bs x time
  metric = jnp.stack(info[plot_opts.baseline_metric], axis=1)

  num_rows = 2+activations.shape[0]
  num_columns = activations.shape[1]
  widths = [1]*num_columns

  color_info = coloring_helper(color_by, data)
  colors = np.array(color_info['colors'])
  colors[:, -1] = min(1,10/data['examples'].shape[0])
  widths = widths + color_info['widths']
  num_columns += color_info['num_columns']

  def plot_baseline(ax):
    if plot_opts.baseline_true_metric:
      ax.plot(metric_curve['x'], metric_curve['y'], color='k', ls='--', alpha=0.5)
    if plot_opts.baseline_this_metric:
      ax.plot(info['iters'], np.mean(metric, axis=0), color='r', ls='--', alpha=0.5)
    if plot_opts.baseline_this_metric_individual:
      ax.add_collection(plot_utils.make_line_collection(info['iters'], metric, colors=colors, ls='--'))

  group_mask = np.arange(len(color_info['labels']))[:, None] == color_info['inds'][None,:]
  group_counts = np.sum(group_mask, axis=1)
  def plot_averages(ax, x, ys, default_color=[0,0,1]):
    ax.plot(x, np.mean(ys, axis=0), color=default_color)
    if plot_opts.plot_group_avgs:
      ax.add_collection(plot_utils.make_line_collection(x, group_mask @ ys / group_counts[:, None], colors=color_info['colors_by_label']))

  fig, axs = plt.subplots(num_rows, num_columns, squeeze=False, sharex=True, sharey=True, 
                          gridspec_kw={'width_ratios': widths})
  fig.set_size_inches(activations.shape[1]*4, num_rows*3)

  add_cbar_from_color_info(fig, axs, color_info, color_by)

  axs[0,0].set_ylabel("Average")
  axs[1,0].set_ylabel("Average for 1,3")
  axs[0,0].set_xlim(info['iters'][0]-10, info['iters'][-1]+10)
  axs[0,0].set_ylim(-0.05, 1.05)
  for i in range(activations.shape[1]):
    axs[0,i].set_title("Head {}".format(i))
    avg_scores = np.mean(activations[:,i,:,:],axis=0)
    plot_baseline(axs[0,i])
    if not plot_opts.only_plot_avg:
      axs[0,i].add_collection(plot_utils.make_line_collection(info['iters'], avg_scores, colors=colors))
    plot_averages(axs[0,i], info['iters'], avg_scores)

    avg_imp_scores = np.mean(activations[0::2,i,:,:],axis=0)
    plot_baseline(axs[1,i])
    if not plot_opts.only_plot_avg:
      axs[1,i].add_collection(plot_utils.make_line_collection(info['iters'], avg_imp_scores, colors=colors))
    plot_averages(axs[1,i], info['iters'], avg_imp_scores)

  for r in range(activations.shape[0]):
    axs[r+2,0].set_ylabel('Attn score {}->{}'.format(r+1, r))
    for i in range(activations.shape[1]):
      plot_baseline(axs[r+2,i])
      if not plot_opts.only_plot_avg:
        axs[r+2,i].add_collection(plot_utils.make_line_collection(info['iters'], activations[r,i,:,:], colors=colors))
      plot_averages(axs[r+2,i], info['iters'], activations[r,i,:,:])

  opt_dict = dict(vars(plot_opts))
  opt_dict['plots'] = 'prev_token_over_time'
  opt_dict['color_by'] = color_by
  fig.suptitle(vars(plot_opts), wrap=True)

  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'_{}'.format(color_by)+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info, 'color_info': color_info}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  # return fig, axs
  plt.close(fig)


def update_attention_over_time(info, iternum, results, model, data):
  if info == dict():
    info['iters'] = []
    info['loss'] = []
    info['prob'] = []
    info['activations'] = []
    info['in_context_prob'] = []
  info['activations'].append(results['activations'][:, 1, :, -1, :])
  info['loss'].append(results['loss'])
  info['prob'].append(results['prob'])
  info['in_context_prob'].append(results['in_context_prob'])
  info['iters'].append(iternum)


def plot_attention_over_time(info, color_by, data, metric_curve, plot_opts, 
                              save_path, integral_metric_starts_to_losses=None):
  # Time x bs x num_heads x context length
  activations = jnp.stack(info['activations'])
  # Flip to context length x num_heads x bs x time
  activations = jnp.transpose(activations, (3,2,1,0))

  # bs x time
  metric = jnp.stack(info[plot_opts.baseline_metric], axis=1)

  additional_metrics = 1 + plot_opts.row_for_token_diff
  num_rows = additional_metrics + plot_opts.row_per_token*activations.shape[0]
  num_columns = activations.shape[1]
  widths = [1]*num_columns

  color_info = coloring_helper(color_by, data)
  # Leave colors in color_info unchanges in case averages are being plotted
  colors = np.array(color_info['colors'])
  colors[:, -1] = min(1,10/data['examples'].shape[0])
  widths = widths + color_info['widths']
  num_columns += color_info['num_columns']

  fig, axs = plt.subplots(num_rows, num_columns, sharex=True,# sharey=True, 
                          squeeze=False, gridspec_kw={'width_ratios': widths})
  fig.set_size_inches(activations.shape[1]*4, num_rows*3)

  add_cbar_from_color_info(fig, axs, color_info, color_by)

  axs[0,0].set_xlim(info['iters'][0]-10, info['iters'][-1]+10)
  axs[0,0].set_ylabel("Attention given to correct token")
  for i in range(activations.shape[1]):
    axs[0,i].set_title("Head {}".format(i))
    axs[0,i].set_ylim(-0.05, 1.05)
    if plot_opts.row_for_token_diff:
      axs[1,i].set_ylim(plot_opts.token_diff_range[0], plot_opts.token_diff_range[1])

  if plot_opts.row_for_token_diff:
    axs[1,0].set_ylabel("Attention delta of correct over\n incorrect token")

  def plot_baseline(ax):
    if plot_opts.baseline_true_metric:
      ax.plot(metric_curve['x'], metric_curve['y'], color='k', ls='--', alpha=0.5)
    if plot_opts.baseline_this_metric:
      ax.plot(info['iters'], np.mean(metric, axis=0), color='r', ls='--', alpha=0.5)
    if plot_opts.baseline_this_metric_individual:
      ax.add_collection(plot_utils.make_line_collection(info['iters'], metric, colors=colors, ls='--'))
  
  group_mask = np.arange(len(color_info['labels']))[:, None] == color_info['inds'][None,:]
  group_counts = np.sum(group_mask, axis=1)
  def plot_averages(ax, x, ys, default_color=[0,0,1]):
    ax.plot(x, np.mean(ys, axis=0), color=default_color)
    if plot_opts.plot_group_avgs:
      ax.add_collection(plot_utils.make_line_collection(x, group_mask @ ys / group_counts[:, None], colors=color_info['colors_by_label']))

  for j in range(activations.shape[1]):
    correct_activations = activations[2*data['correct_ind']+1, j, np.arange(activations.shape[2]), :]
    incorrect_activations = activations[2*(1-data['correct_ind'])+1, j, np.arange(activations.shape[2]), :]

    plot_baseline(axs[0,j])
    if not plot_opts.only_plot_avg:
      axs[0,j].add_collection(plot_utils.make_line_collection(info['iters'], correct_activations, colors=colors))
    plot_averages(axs[0,j], info['iters'], correct_activations)

    if plot_opts.row_for_token_diff:
      plot_baseline(axs[1,j])
      if not plot_opts.only_plot_avg:
        axs[1,j].add_collection(plot_utils.make_line_collection(info['iters'], correct_activations-incorrect_activations, colors=colors))
      plot_averages(axs[1,j], info['iters'], correct_activations-incorrect_activations)

  for i in range(additional_metrics, num_rows):
    axs[i,0].set_ylabel("Attention given to token {}".format(i-additional_metrics))
    for j in range(activations.shape[1]):
      plot_baseline(axs[i,j])
      if not plot_opts.only_plot_avg:
        axs[i,j].add_collection(plot_utils.make_line_collection(info['iters'], activations[i-additional_metrics,j], colors=colors))
      plot_averages(axs[i,j], info['iters'], activations[i-additional_metrics,j,:,:])
      axs[i,j].set_ylim(-0.05, 1.05)

  opt_dict = dict(vars(plot_opts))
  opt_dict['plots'] = 'attention_over_time'
  opt_dict['color_by'] = color_by
  fig.suptitle(vars(plot_opts), wrap=True)

  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'_{}'.format(color_by)+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info, 'color_info': color_info}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  plt.close(fig)


update_avg_attention_over_time = update_attention_over_time


def plot_avg_attention_over_time(info, color_by, data, metric_curve, plot_opts, 
                                  save_path, integral_metric_starts_to_losses=None):
  matplotlib.rcParams.update({'font.size': 18})
  # Time x bs x num_heads x context length
  activations = jnp.stack(info['activations'])
  # Flip to context length x num_heads x bs x time
  activations = jnp.transpose(activations, (3,2,1,0))

  # bs x time
  metric = jnp.stack(info[plot_opts.baseline_metric], axis=1)

  fig, axs = plt.subplots(1,1)
  fig.set_size_inches(10,6)
  axs.set_ylabel("Induction strength")
  axs.set_xlabel("# sequences seen")

  assert color_by == 'default', "plot_avg_attention_over_time only supports default coloring"
  colors = plt.cm.get_cmap('Dark2')(np.arange(8))

  for h in range(activations.shape[1]):
    correct_activations = activations[2*data['correct_ind']+1, h, np.arange(activations.shape[2]), :]
    incorrect_activations = activations[2*(1-data['correct_ind'])+1, h, np.arange(activations.shape[2]), :]
    axs.plot(info['iters'], np.mean(correct_activations-incorrect_activations, axis=0), color=colors[h], label=h)
    print(h, np.mean(correct_activations-incorrect_activations, axis=0)[-1])

  axs.legend(title='Layer 2\nHead #', loc='upper left', bbox_to_anchor=(1, 1))

  plot_utils.scientific_notation_ticks(axs, xaxis=True, yaxis=False)

  if plot_opts.baseline_true_metric:
    axs.plot(metric_curve['x'], metric_curve['y'], color='k', ls='--', alpha=0.5)
  if plot_opts.baseline_this_metric:
    axs.plot(info['iters'], np.mean(metric, axis=0), color='r', ls='--', alpha=0.5)

  axs.set_ylim(-0.05, 1)
  if plot_opts.plot_range is not None:
    axs.set_xlim(plot_opts.plot_range[0], plot_opts.plot_range[1])
  plt.tight_layout()

  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'_{}'.format(color_by)+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info, 'color_info': None}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  plt.close(fig)


def update_all_token_layer_attention_over_time(info, iternum, results, model, data):
  if info == dict():
    info['iters'] = []
    info['loss'] = []
    info['prob'] = []
    info['activations_ind0'] = []
    info['activations_ind1'] = []
    info['in_context_prob'] = []
  # batch x layer x head x query x key -> layer x head x query x key
  info['activations_ind0'].append(jnp.mean(results['activations'][data['correct_ind'] == 0], axis=0))
  info['activations_ind1'].append(jnp.mean(results['activations'][data['correct_ind'] == 1], axis=0))
  # Unlike other update methods, we only store 1D values here
  # Since this mode does not support plotting individual data points.
  info['loss'].append(jnp.mean(results['loss']))
  info['prob'].append(jnp.mean(results['prob']))
  info['in_context_prob'].append(jnp.mean(results['in_context_prob']))
  info['iters'].append(iternum)

def plot_all_token_layer_attention_over_time(
    info, color_by, data, metric_curve, plot_opts, save_path, 
    integral_metric_starts_to_losses=None):
  print("WARNING: plot_all_token_layer_attention_over_time does not support color_by")
  print("WARNING: plot_all_token_layer_attention_over_time assumes context length 5/only two possible correct_inds")
  # Time x layer x num_heads x query x key
  activations = [jnp.stack(info['activations_ind0']), jnp.stack(info['activations_ind1'])]

  layers = activations[0].shape[1]
  num_heads = activations[0].shape[2]
  tokens = activations[0].shape[3]

  colors = plt.cm.get_cmap('Dark2')(np.arange(num_heads))

  fig, axs = plt.subplots(2*tokens, tokens*layers, sharex=True, sharey=True)
  fig.set_size_inches(tokens*layers*4, tokens*4)
  if plot_opts.plot_range is not None:
    axs[0,0].set_xlim(plot_opts.plot_range[0], plot_opts.plot_range[1])
  axs[0,0].set_ylim(-0.05,1.05)

  def plot_baseline(ax):
    if plot_opts.baseline_true_metric:
      ax.plot(metric_curve['x'], metric_curve['y'], color='k', ls='--', alpha=0.5)
    if plot_opts.baseline_this_metric:
      ax.plot(info['iters'], jnp.array(info[plot_opts.baseline_metric]), color='r', ls='--', alpha=0.5)

  for correct_ind in range(2):
    for l in range(layers):
      for qt in range(tokens):
        for kt in range(qt+1):
          plot_baseline(axs[correct_ind*tokens+qt, l*tokens+kt])
          axs[correct_ind*tokens+qt, l*tokens+kt].add_collection(
            plot_utils.make_line_collection(info['iters'], 
                                            activations[correct_ind][:, l, :, qt, kt].T, 
                                            colors=colors))
  
  opt_dict = dict(vars(plot_opts))
  opt_dict['plots'] = 'all_token_layer_attention_over_time'
  fig.suptitle(vars(plot_opts), wrap=True)

  if 'pkl' in plot_opts.save_plot_as:
    with open(save_path+'_{}'.format(color_by)+'.pkl', 'wb') as f:
      pkl.dump({'fig': fig, 'axs': axs, 'info': info}, f)
  if 'pdf' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.pdf')
  if 'png' in plot_opts.save_plot_as:
    fig.savefig(save_path+'_{}'.format(color_by)+'.png')
  if 'pdb' in plot_opts.save_plot_as:
    print("Done plotting. Dropping into pdb as requested...")
    pdb.set_trace()
  plt.close(fig)


def get_update_fn(plot_type):
  return globals()['_'.join(['update', plot_type])]


def get_plot_fn(plot_type):
  return globals()['_'.join(['plot', plot_type])]


def parse_opts_csv_str(s):
  return jnp.array([[int(n) for n in p.split(',')] for p in s])


def get_closest_inds(search_for, search_in):
  '''
  Returns the unique indices in search_in that are closest to the elements of search_for
  '''
  iters_inds = np.argmin(np.abs(search_for[:, None] - search_in[None, :]), axis=1)
  return np.sort(np.unique(iters_inds))


def make_forward_fn(options, default_fn=opto.default_model_fwd_fn):
  call_fn = opto.make_fn_from_opts(options, default_fn=default_fn)

  def forward_fn(model, x, y, key):
    '''
    Returns dict of:
      activations: bs x depth x num_heads x context_lengthQ x context_lengthK
      logits: bs x num_classes
      loss: bs x 1
      ...varous other metrics
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
    fsl_output_acc = (jnp.argmax(in_context_pred_y, axis=-1) == y[:,-1])

    return {'activations': jnp.transpose(head_act, (1,0,2,3,4)), 
            'logits': all_activations['out'][:, -1, :],
            'prob': output_prob,
            'acc': output_acc,
            'in_context_loss': fsl_loss,
            'in_context_prob': fsl_output_prob,
            'in_context_acc': fsl_output_acc,
            'loss': query_ce}

  return eqx.filter_jit(forward_fn)


def make_dense_icl_data(meta_opts, raw_data=None):
  '''
  Creates a set of ICL sequences to feed to model. Makes a dense set across all provided class_pairs and label_pairs.

  Possible subsample (for speed) using the max_sequences parameter
  '''
  print("WARNING: Dense ICL data only allows context length 5 (two exemplar-label pairs)")
  ### Get all class pairs
  if meta_opts.class_range is not None:
    n_classes = meta_opts.class_range[1] - meta_opts.class_range[0]
    temp = np.broadcast_to(np.arange(meta_opts.class_range[0], meta_opts.class_range[1]), (n_classes, n_classes))
    all_pairs = jnp.vstack([temp.T.reshape(-1), temp.reshape(-1)]).T
    if meta_opts.plot_same:
      class_pairs = all_pairs[all_pairs[:, 0] == all_pairs[:, 1]]
    else:
      class_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
  else:
    assert meta_opts.class_pairs is not None
    class_pairs = parse_opts_csv_str(meta_opts.class_pairs)

  ### Get all label pairs
  raw_labels = parse_opts_csv_str(meta_opts.label_pairs)
  label_pairs = jnp.concatenate([raw_labels, raw_labels[:, ::-1]], axis=0)

  ### Make product of class pairs and label pairs
  total = class_pairs.shape[0]*label_pairs.shape[0]
  all_class_pairs = jnp.broadcast_to(class_pairs, (label_pairs.shape[0], *class_pairs.shape)).reshape(total, class_pairs.shape[1])
  all_label_pairs = jnp.broadcast_to(label_pairs[:, None, :], (label_pairs.shape[0], class_pairs.shape[0], label_pairs.shape[1])).reshape(total, label_pairs.shape[1])

  # Doubles the set of sequences to account for either input being the query
  duplicate = lambda x: jnp.concatenate([jnp.concatenate([x, x[:, 0:1]], axis=1), jnp.concatenate([x, x[:, 1:2]], axis=1)], axis=0)

  all_data = dict()
  all_data['class_pairs'] = duplicate(all_class_pairs)
  all_data['labels'] = duplicate(all_label_pairs)
  all_data['correct_ind'] = jnp.concatenate([jnp.zeros(all_class_pairs.shape[0]), jnp.ones(all_class_pairs.shape[0])]).astype(int)

  if meta_opts.max_sequences is not None:
    subselect_inds = jax.random.choice(jax.random.PRNGKey(meta_opts.sample_seed), all_data['class_pairs'].shape[0], (meta_opts.max_sequences,), replace=False)
    for k in all_data:
      all_data[k] = all_data[k][subselect_inds]

  if raw_data is None:
    # Load in data from meta_opts.data_file to raw_data
    with h5.File(meta_opts.data_file, 'r') as f:
      raw_data = jnp.array(f[meta_opts.data_path_in_file])

  # Read in just the relevant data points
  all_data['examples'] = raw_data[all_data['class_pairs'], meta_opts.icl_exemplar_ind]
  all_data['metadata_ind'] = jnp.zeros(all_data['class_pairs'].shape[0], dtype=int)
  all_data['metadata_labels'] = ['icl']

  return all_data


def make_eval_based_data(meta_opts):
  '''
  Loads in data from preconstructed eval files.
  '''
  # Note the subset implementation is a bit obtuse, but this is to ensure consistent metadata_ind,
  # assuming the same set of data files and prefixes were used.
  # Essentially, if the eval file has 3 subsets, but we're only plotting the first and third,
  # we want their metadata inds to be 0, 2 instead of 0, 1.
  # The original idea here was to allow plots to have consistent coloring, even when plotting
  # different subsets of evaluators. However, this isn't currently supported by color_by. We
  # plan to add this in the future.
  all_data = dict()
  keys = []
  global_ind = 0
  for prefix, fname in zip(meta_opts.prefixes, meta_opts.data_file):
    eval_h5 = h5.File(fname, 'r')
    these_keys = [str(s) for s in eval_h5.keys()]
    print("Found these subsets in file prefix {}: {}".format(prefix, these_keys))
    for i, k in enumerate(these_keys):
      if meta_opts.eval_subsets is not None:
        check = '_'.join([prefix, k]) if len(prefix) > 0 else k
        if check not in meta_opts.eval_subsets:
          continue
      size = -1
      for s in eval_h5[k].keys():
        loaded_in_arr = jnp.array(eval_h5['/'.join([k, s])])
        all_data.setdefault(s, []).append(loaded_in_arr)
        if size > -1:
          assert size == loaded_in_arr.shape[0], 'Eval data file is invalid'
        else:
          size = loaded_in_arr.shape[0]
      all_data.setdefault('metadata_ind', []).append(jnp.ones(size, dtype=int)*(i+global_ind))
    global_ind += len(these_keys)
    keys.extend(['_'.join([prefix, s]) for s in these_keys])
    eval_h5.close()

  for k in all_data:
    all_data[k] = jnp.concatenate(all_data[k], axis=0)

  assert 'examples' in all_data, "Must have examples in eval file"
  assert 'labels' in all_data, "Must have labels in eval file"

  print("Loaded in all eval data", all_data['examples'].shape)

  if meta_opts.max_sequences is not None:
    subselect_inds = jax.random.choice(jax.random.PRNGKey(meta_opts.sample_seed), all_data['examples'].shape[0], (meta_opts.max_sequences,), replace=False)
    for k in all_data:
      all_data[k] = all_data[k][subselect_inds]
  
  if 'class_pairs' not in all_data:
    print("WARNING: eval data did not have class pair information, so any coloring modes requiring this will be unsupported")
  
  if 'correct_ind' not in all_data:
    print("WARNING: eval data did not have correct ind, attempting to add it based on label info. Will assign -1 if not found")
    match_output = all_data['labels'][:, :-1] == all_data['labels'][:, -1][:, None]
    all_data['correct_ind'] = jnp.where(jnp.sum(match_output, axis=1) > 0, jnp.argmax(match_output, axis=1), -1)

  all_data['metadata_labels'] = keys

  return all_data


def make_data(meta_opts, raw_data=None):
  if meta_opts.data_mode == 'icl':
    return make_dense_icl_data(meta_opts, raw_data=raw_data)
  elif meta_opts.data_mode == 'eval':
    return make_eval_based_data(meta_opts)
  elif meta_opts.data_mode == 'samplers':
    raise NotImplementedError
  else:
    print("Unsupported data_mode")
    raise NotImplementedError

def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--plots', nargs='+', default=['attention_over_time', 'metric_vs_sim'], choices=['attention_over_time', 'metric_vs_sim', 'metric_by_color_by', 'prev_token_over_time', 'biases_over_time', 'avg_attention_over_time', 'all_token_layer_attention_over_time'], type=str, help='Which plot types to make')
  parser.add_argument('--color_by', nargs='+', default=None, type=str, choices=['default', 'similarity', 'correct_ind', 'label_pair', 'relabel', 'class_pair', 'output', 'output_and_ind', 'metadata'], help="How to color plots. One argument per argument in plots, if specified. Note, all modes are only supported when data_mode = icl. Support in other data modes is not guaranteed.")

  parser.add_argument('--base_folder', default='./runs', type=str, help="Base folder to fetch runs from")
  parser.add_argument('--run_folder', nargs='+', default=['20240103144741_omniglot50_rl5'], type=str, help="Run folders to visualize")

  parser.add_argument('--batch_size', type=int, default=1024, help='What batch size to run the model with.')

  parser.add_argument('--plot_range', nargs='+', type=int, default=None, help="Range to plot.")
  parser.add_argument('--num_ckpts_to_plot', type=int, default=int(1e9), help="max number of checkpoints to plot")

  # Data construction arguments:
  parser.add_argument('--data_mode', type=str, default='icl', choices=['icl', 'eval', 'samplers'], help='Mode for constructing data. "icl" was used for the induction heads paper and is the only mode that supports full features (e.g., all coloring modes). That said, it\'s not that useful outside that paper (e.g., for transience results we used a large number of classes. For generating icl data from this larger set, it\'s better to use the samplers mode to avoid constructing the full dense array of class pairs). The other modes were used for better understanding transience. As a result, some of the coloring modes may not be meaningful (the main mode that was used for those results was the metadata mode).')

  # Generally useful data parameters
  parser.add_argument('--data_file', type=str, nargs='*', default=None, help='If specified, the file(s) to get data from. Defaults to the data file used for the first run in opts.run_folder. Note, if data_mode is "icl" or "samplers" this is the file containing exemplars. If data mode is "eval", this should be a file with already constructed sequences.')
  parser.add_argument('--data_path_in_file', type=str, default='resnet18/224/feat', help='A vestige of older versions of code where we considered different underlying feature sets. This subpath is basically always correct. Only used when data_mode is "icl" or "samplers"')
  parser.add_argument('--max_sequences', type=int, default=None, help="If specified, downsamples data generated by mode uniformly to max sequences. Note, this may cause imbalance across different pooling modes, so use with care. Mostly useful for quick iteration.")
  parser.add_argument('--sample_seed', type=int, default=10, help='Seed used to generate (in "samplers" mode) and/or subselect data (in "icl" and "eval" mode). For reproducibility purposes.')

  ## File data mode parameters
  parser.add_argument('--prefixes', nargs='+', type=str, default=[''], help='If more than one eval data file is provided, prefixes must be provided to distinguish possibly overlapping keys from the files.')
  parser.add_argument('--eval_subsets', type=str, nargs='*', default=None, help='Only used in data_mode = "eval". If specified, only the specified keys from the data_file are used.')

  ## ICL data mode parameters
  parser.add_argument('--class_range', nargs='*', type=int, default=None, help='Range of classes to plot')
  parser.add_argument('--plot_same', action='store_true', help='Use sequences of the form A 0 A 1 A. Useful for verifying induction heads getting split patterns.')
  parser.add_argument('--class_pairs', nargs='*', type=str, default=None, help='CSV pairs of classes to investigate. E.g. "--class pairs 6,11 0,1.')
  parser.add_argument('--label_pairs', nargs='+', type=str, default=['0,1'], help='CSV pairs of labels to use as relabelings')
  parser.add_argument('--icl_exemplar_ind', type=int, default=0, help="Only used in ICL mode. What exemplar index to use.")

  # General baseline metric params
  parser.add_argument('--baseline_metric', type=str, default='loss', choices=['loss', 'prob', 'acc', 'in_context_loss', 'in_context_prob'], help='Which metric to plot as the baseline.')
  parser.add_argument('--metric_curve', nargs=2, type=str, default=['eval_iter', 'fsl_train/loss'], help='Details to use to plot metric curve as the "average". First var is key for x axis, second is for y axis.')
  parser.add_argument('--metric_range', nargs=2, type=float, default=[0,2], help='Range for metric y axis in metric_by_color_by mode.')

  parser.add_argument('--only_plot_avg', action='store_true', help="If specified, individual traces will not be plotted (will significantly speed up plot generation).")

  # Used for calculating integral metric
  parser.add_argument('--loss_curve', nargs=2, type=str, default=['eval_iter', 'train_eval/loss'], help='Used for calculating integral metric ONLY.')
  parser.add_argument('--integral_metric_start', nargs='+', type=int, default=None, help='Start iteration to calculate integral metric. If multiple, calculate an integral metric for each value')
  parser.add_argument('--integral_metric_thresh', nargs='+', type=float, default=[0.6], help='Threshold under which the loss_curve has to be to start integral_metric. Only used if integral_metric_start is not specified. If multiple, calculate an integral metric for each value')

  # Attention over time plot args (also used for prev token over time)
  parser.add_argument('--row_per_token', action='store_true')
  parser.add_argument('--row_for_token_diff', action='store_true')
  parser.add_argument('--token_diff_range', nargs=2, type=float, default=[-0.05, 1.05], help="y-axis range when plotting the row for token difference")
  parser.add_argument('--baseline_true_metric', action='store_true', help='If true, plots the baseline_metric, as specified by metric_curve, in black')
  parser.add_argument('--baseline_this_metric', action='store_true', help='If true, plots a red line with the specified baseline_metric computed on the data used to make this plot (which may be different than the values in metric_curve because different subsets).')
  parser.add_argument('--baseline_this_metric_individual', action='store_true', help='If true, plots a line with the specified baseline_metric for each data point considered. Typically only used when plotting a small number of points to avoid confusion.')
  parser.add_argument('--plot_group_avgs', action='store_true', help='If true, plots an average (darkened) line for each subgroup average')

  # metric vs sim plot args
  parser.add_argument('--save_problem_points', type=int, default=None, help='Top K problem points to save for each timestamp')

  # For prev token over time
  parser.add_argument('--use_raw_prev_tok_score', action='store_true')

  # For saving pickle to format with in notebook
  parser.add_argument('--save_plot_as', type=str, nargs='+', default=['png'], choices=['png', 'pdf', 'pkl', 'pdb'], help="File format to store plot in. Defaults to png. pkl not fully supported. If pdb, drops into a pdb.set_trace() after saving the rest.")

  return parser


def populate_plot_info(plot_info, meta_opts, run_folder, all_data):
  # Load relevant run params
  opts = main_utils.get_opts_from_json_file('/'.join([run_folder, 'config.json']))

  ##### Load models
  model = main_utils.get_model_from_opts(opts, (all_data['examples'].shape[-1], ))
  fwd_fn_from_train = opto.make_fn_from_opts(opts)
  # Note the meta opts do not *add on* to the default_fn=fwd_fn_from_train. The default_fn is only used if meta_opts
  # does not do any optogenetics. This is intentional as we want to allow turning off the optogenetics from
  # train time in some cases probably.
  model_fn = main.make_batched_fn(make_forward_fn(meta_opts, default_fn=fwd_fn_from_train), meta_opts.batch_size)

  print("model output shape", model.unembed.weight.shape)
  opt_state = main_utils.get_optimizer_from_opts(opts).init(eqx.filter(model, eqx.is_array))

  ckpt_fmt = {'iter': -1, 
              'seeds': {'eval_model_seed': jax.random.PRNGKey(0),
                        'train_data_seed': jax.random.PRNGKey(0),
                        'train_model_seed': jax.random.PRNGKey(0)}, 
              'opt_state': opt_state,
              'model': model}

  filenames = sorted(os.listdir('/'.join([run_folder, 'checkpoints'])))
  iters = np.array([int(cn.split('.')[0]) for cn in filenames])

  ##### Figure out which indices to plot
  if meta_opts.plot_range is None:
    plot_range = [iters[0], iters[-1]]
  else:
    plot_range = meta_opts.plot_range
  iters_mask = np.logical_and(iters >= plot_range[0], iters <= plot_range[1])
  if np.sum(iters_mask) > meta_opts.num_ckpts_to_plot:
    inds_to_plot = get_closest_inds(np.linspace(plot_range[0], plot_range[1], meta_opts.num_ckpts_to_plot), iters)
  else:
    inds_to_plot = np.where(iters_mask)[0]

  for ind in tqdm(inds_to_plot):
    ckpt_fname = '/'.join([run_folder, 'checkpoints', filenames[ind]])
    ckpt = eqx.tree_deserialise_leaves(ckpt_fname, ckpt_fmt)
    results = model_fn(ckpt['model'], all_data['examples'], all_data['labels'], key=jax.random.PRNGKey(0))
    for plot_type in plot_info:
      get_update_fn(plot_type)(plot_info[plot_type], iters[ind], 
                                results, ckpt['model'], all_data)


if __name__ == '__main__':
  parser = create_parser()
  opto.add_args_to_parser(parser)
  meta_opts = parser.parse_args()

  if 'pkl' in meta_opts.save_plot_as:
    print("WARNING: pkl saving mode not fully supported")
  if 'pdb' in meta_opts.save_plot_as:
    print("WARNING: pdb saving mode should only be used when only making one plot")
    assert len(meta_opts.plots) == 1
    assert len(meta_opts.run_folder) == 1

  run_time = datetime.now().strftime("%Y%m%d%H%M%S")
  if meta_opts.color_by is not None:
    assert len(meta_opts.plots) == len(meta_opts.color_by)
  
  # Only used if data file is not None, as in this case user might want onehot data or some other mode
  raw_data = None
  
  if meta_opts.data_mode in ['icl', 'samplers']:
    if meta_opts.data_file is None:
      run_folder = '/'.join([meta_opts.base_folder, meta_opts.run_folder[0]])
      opts = main_utils.get_opts_from_json_file('/'.join([run_folder, 'config.json']))
      raw_data = main_utils.get_data_from_opts(opts)
      print("No data file provided, getting data from", opts.data_file)
    else:
      assert len(meta_opts.data_file) == 1, 'Only 1 data file can be provided in modes "icl" and "samplers"'
  else:
    assert meta_opts.data_file is not None, 'Data file must be provided in "eval" mode'
    assert len(meta_opts.data_file) == len(meta_opts.prefixes), 'If providing more than one eval file, prefixes must be given for each'
  
  all_data = make_data(meta_opts, raw_data)
  print("Examples shape:", all_data['examples'].shape)

  for rf in meta_opts.run_folder:
    run_folder = '/'.join([meta_opts.base_folder, rf])
    plots_dir = '/'.join([run_folder, 'plots'])
    os.makedirs(plots_dir, exist_ok=True)

    with h5.File('/'.join([run_folder, 'log.h5']),'r') as log:
      loss_curve = {'x': log[meta_opts.loss_curve[0]][:], 'y': np.mean(log[meta_opts.loss_curve[1]], axis=1)}
      metric_curve = {'x': log[meta_opts.metric_curve[0]][:], 'y': np.mean(log[meta_opts.metric_curve[1]], axis=1)}
    if meta_opts.integral_metric_start is None:
      assert meta_opts.integral_metric_thresh is not None
      integral_metric_start = [loss_curve['x'][np.argmax(loss_curve['y'] < t)] for t in meta_opts.integral_metric_thresh]
      print("Integral metric start set to {} based on thresh".format(integral_metric_start))
    else:
      integral_metric_start = meta_opts.integral_metric_start

    integral_metric_start_loss = {start: loss_curve['y'][np.argmax(loss_curve['x'] >= start)] for start in integral_metric_start}

    plot_info = {plot_type: dict() for plot_type in meta_opts.plots}

    populate_plot_info(plot_info, meta_opts, run_folder, all_data)

    for i, plot_type in enumerate(meta_opts.plots):
      color_by = main.smart_index(meta_opts.color_by, i, 'default')
      get_plot_fn(plot_type)(plot_info[plot_type], 
                              color_by, 
                              all_data, 
                              metric_curve, 
                              meta_opts, 
                              integral_metric_starts_to_losses=integral_metric_start_loss,
                              save_path='/'.join([plots_dir, '_'.join([run_time, plot_type])]))
