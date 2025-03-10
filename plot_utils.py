'''
Various useful plotting functionality. Combined with `visualize_runs.py`,
this should be a superset of `ih_paper_plot_utils.py`. The latter is left
in the codebase for backwards compatibility/reproducing the induction heads
paper.

If run directly (e.g., `python plot_utils.py`), this file generates some 
basic plots of loss curves etc. We stopped using this functionality as wandb
offered a better interface to quickly compare runs. It may still be useful
to study learning of individual points/classes throughout training.
'''
import h5py as h5
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from time import time
import argparse


def clear_axticks(*args):
    for ax in args:
        ax.set_xticks([])
        ax.set_yticks([])


def convert_arr_to_hex(arr):
    '''
    Converts numpy arr of shape N x 3 to hex codes, returned as a list.
    Useful for coloring runs in wandb.
    '''
    return [''.join([hex(int(el))[2:].zfill(2) for el in row]) for row in arr]


def make_line_collection(x, ys, **kwargs):
    '''
    Returns a line collection object that can be used
    to plot a bunch of lines given by x,[y for y in ys]

    This is more efficient than a for loop with many
    calls to `plot` over all the ys. If the color of all
    lines is the same, then a single `plot` command
    can be used. However, even in those cases, using LineCollection
    appears to be *way* faster (like 10x). Furthermore,
    a single plot command does not support different colors
    for each line, but this method does (through **kwargs)
    '''
    points = np.transpose(np.stack([np.broadcast_to(x, ys.shape), ys]), (1,2,0))
    return matplotlib.collections.LineCollection(points, **kwargs)


def errorfill(x, y, yerr, color='C0', alpha=0.3, line_alpha=1.0, ax=None, label=None, lw=1, marker=None, ms=50, ls='-'):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    pl = ax.plot(x, y, color=color, label=label, lw=lw, marker=marker, ls=ls, ms=ms, alpha=line_alpha)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.7)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha)
    return pl


def scientific_notation_ticks(axs, xaxis=True, yaxis=False):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    if xaxis:
        axs.xaxis.set_major_formatter(formatter)
    if yaxis:
        axs.yaxis.set_major_formatter(formatter)


def get_series_baseline(s):
    if s == 'prob':
        return 0.5
    if s == 'loss': 
        return np.log(2)
    if s == 'use_context_prob':
        return 1
    if s.endswith('acc'):
        return 0.5
    return 0


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_folder', default='./runs', type=str, help="Base folder that runs are from")

    parser.add_argument('--series', nargs='+', default=['prob', 'loss'], type=str, help='Series to plot')
    parser.add_argument('--run_folder', nargs='+', default=['20240103144741_omniglot50_rl5'], type=str, help="Run folders to visualize")
    parser.add_argument('--run_inds', nargs='+', default=None, type=int, help="Run inds for coloring")
    parser.add_argument('--evals', nargs='+', default=['fsl_train', 'fsl_val_rl', 'fsl_test_class'], type=str, help="Eval datasets to visualize")
    parser.add_argument('--iter_range', nargs=2, default=None, type=int, help="What iteration range to display")
    parser.add_argument('--save_prefix', default='combined', type=str, help="prefix to use when saving plot")

    parser.add_argument('--only_avg', action='store_true')
    parser.add_argument('--alpha', default=0.002, type=int, help="Alpha value to use for individual run lines, if present")
    
    return parser


eval_map_readable = {'fsl_train': 'Train', 'fsl_test_class': 'Test (exemplar)', 'fsl_val_rl': 'Test (relabel)'}

series_map_readable = {'loss': 'Loss', 'acc': 'Accuracy'}


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    matplotlib.rcParams.update({'font.size': 18})

    all_plots = dict()
    for series in opts.series:
        all_plots[series] = plt.subplots(len(opts.evals),1, sharex=True, sharey=True, squeeze=False)
        all_plots[series][0].set_size_inches(10, 5*len(opts.evals))

    colors = plt.cm.tab10(np.arange(len(opts.run_folder)))

    if opts.run_inds is None:
        opts.run_inds = np.arange(len(opts.run_folder))

    for run_ind, rf in zip(opts.run_inds, opts.run_folder):
        run_folder = '/'.join([opts.base_folder, rf])

        f = h5.File('/'.join([run_folder, "log.h5"]), 'r')

        iters = f['eval_iter'][:]

        if opts.iter_range is not None:
            mask = np.logical_and(iters >= opts.iter_range[0], iters <= opts.iter_range[1])
        else:
            mask = np.ones(iters.shape, dtype=bool)

        for series in opts.series:
            fig, ax = all_plots[series][0], all_plots[series][1]

            start = time()
            for j, n in enumerate(opts.evals):
                to_plot = f['/'.join([n, series])][:]
                print(to_plot.shape, iters.shape)
                color = plt.cm.tab10(run_ind)
                each_line_color = np.array(plt.cm.tab20(2*run_ind+1))
                each_line_color[-1] = opts.alpha
                if not opts.only_avg:
                    ax[j,0].add_collection(make_line_collection(iters[mask], to_plot[mask, :].T, colors=[each_line_color]))
                ax[j,0].plot(iters[mask], np.mean(to_plot, axis=1)[mask], color=color, lw=3, label = rf[-1])#label='_'.join(rf.split('_')[1:]))
                ax[j,0].axhline(get_series_baseline(series), color='k', ls='--', lw=0.5)
                if series == 'loss':
                    ax[j,0].set_ylim([-0.05,2])
                    # pass
                elif series.endswith('acc') or series.endswith('prob'):
                    ax[j,0].set_ylim([-0.05,1.05])
                if len(opts.evals) > 1:
                    ax[j,0].set_title(eval_map_readable.get(n, n))
                ax[j,0].set_ylabel(series_map_readable.get(series, series))
                ax[j,0].set_xlabel('# sequences seen')
                # ax[j,0].set_yscale('log')
                scientific_notation_ticks(ax[j,0],xaxis=True,yaxis=False)

            print(time()-start)

        f.close()

    

    if len(opts.run_folder) == 1:
        prefix = '/'.join([opts.base_folder, opts.run_folder[0], ''])
    else:
        assert opts.save_prefix is not None
        prefix = opts.save_prefix
    for series in opts.series:
        if len(opts.run_folder) > 1:
            for ax in all_plots[series][1][:,0]:
                ax.legend(title='Init seed')
        all_plots[series][0].savefig(prefix+"{}.png".format(series))
