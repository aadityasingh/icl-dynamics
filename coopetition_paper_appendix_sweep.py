'''
This file can be used to get all the runs for the appendix of the Strategy Coopetition paper.

This file assumes that `coopetition_paper_sweep.py` has been run. This file can be run in a single
"block" of runs.
'''
import submitit
from copy import deepcopy
import os
import pdb

from main_utils import create_parser
from main import run_with_opts
import opto

################################################################################
### SETUP
################################################################################

# Toggle this on if you want to use wandb to monitor runs. See main.py for default project name,
# currently set to `icl-transience`
use_wandb = False

base_path = '/path/to/save'
os.makedirs(base_path, exist_ok=True)
os.makedirs(f'{base_path}/runs', exist_ok=True)
os.makedirs(f'{base_path}/log_submitit', exist_ok=True)

executor = submitit.AutoExecutor(folder=f"{base_path}/log_submitit")
executor.update_parameters(
  name='coopetition',
  nodes=1,
  gpus_per_node=1,
  tasks_per_node=1,
  cpus_per_task=20,
  mem_gb=32,
  timeout_min=1440*2, # 2 days to give plenty of buffer for the longest run, which takes 16hrs
  slurm_partition='TODO FILL IN',
)

################################################################################
### APPENDIX RUNS (18 runs)
################################################################################

parser = create_parser()
opto.add_args_to_parser(parser)

# We don't specify 'run', as this should be specified later
opts = parser.parse_args(
  ['--raw_name', '--base_folder', f'{base_path}/runs',
    '--suppress_output',
    '--data_type', 'file', '--data_file', f'{base_path}/embeddings/omniglot_features_reordered.h5',
    '--pos_embedding_type', 'ape', '--depth', '2', '--d_model', '64', '--num_heads', '8',
    '--init_seed', '5', '--train_seed', '2',
    '--mixing_coeffs', '1.0', '--pt_burstiness', '1', '--train_context_len', '2', '--fs_relabel', '0',
    '--class_split', '12800', '0', '184', '--exemplar_split', '20', '0', '0',
    '--train_iters', '64000000', '--eval_every', '100000', '--ckpt_every', '16000000',
    '--lr', '0.00001', '--optimizer', 'adam',
    '--load_eval_data', f'{base_path}/runs/default_eval/eval_data.h5']
)
opts.use_wandb = use_wandb

all_opts = []

### Main run, for way longer, used for Figure 11
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'long_main'
all_opts[-1].train_iters = 320000000
all_opts[-1].ckpt_every = 80000000

### Reproduce across different random seeds
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'is6'
all_opts[-1].init_seed = 6

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'is7'
all_opts[-1].init_seed = 7

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'ts3'
all_opts[-1].train_seed = 3

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'ts4'
all_opts[-1].train_seed = 4

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'is6_ts3'
all_opts[-1].init_seed = 6
all_opts[-1].train_seed = 3


### Different positional embeddings
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'rope'
all_opts[-1].pos_embedding_type = 'rope'
all_opts[-1].sin_time = 30

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'sinpe'
all_opts[-1].pos_embedding_type = 'sinusoidal'
all_opts[-1].sin_time = 30


### Data properties (run for 1e8 iterations)
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'c12800_ex5'
all_opts[-1].exemplar_split = [5,0,15]
all_opts[-1].train_iters = 100000000
all_opts[-1].ckpt_every = 20000000

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'c1600_ex20'
all_opts[-1].data_file = f'{base_path}/embeddings/omniglot_features_norotate.h5'
all_opts[-1].class_split = [1600, 0, 23]
all_opts[-1].train_iters = 100000000
all_opts[-1].ckpt_every = 20000000

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'c1600_ex5'
all_opts[-1].data_file = f'{base_path}/embeddings/omniglot_features_norotate.h5'
all_opts[-1].class_split = [1600, 0, 23]
all_opts[-1].exemplar_split = [5,0,15]
all_opts[-1].train_iters = 100000000
all_opts[-1].ckpt_every = 20000000


### MLPs (run for 3.2e8 iterations)
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'mlp'
all_opts[-1].mlp_ratio = 4
all_opts[-1].train_iters = 320000000
all_opts[-1].ckpt_every = 80000000

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'mlp_c12800_ex5'
all_opts[-1].mlp_ratio = 4
all_opts[-1].exemplar_split = [5,0,15]
all_opts[-1].train_iters = 320000000
all_opts[-1].ckpt_every = 80000000

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'mlp_c1600_ex20'
all_opts[-1].mlp_ratio = 4
all_opts[-1].data_file = f'{base_path}/embeddings/omniglot_features_norotate.h5'
all_opts[-1].class_split = [1600, 0, 23]
all_opts[-1].train_iters = 320000000
all_opts[-1].ckpt_every = 80000000

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'mlp_c1600_ex5'
all_opts[-1].mlp_ratio = 4
all_opts[-1].data_file = f'{base_path}/embeddings/omniglot_features_norotate.h5'
all_opts[-1].class_split = [1600, 0, 23]
all_opts[-1].exemplar_split = [5,0,15]
all_opts[-1].train_iters = 320000000
all_opts[-1].ckpt_every = 80000000


### 1 Layer CIWL-only run
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'depth1'
all_opts[-1].depth = 1


### ICL is not necessary for CIWL to emerge
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'disallow_pt_head_attn_on_labels'
all_opts[-1].opto_disallow_prev_token_heads = ['0:{}'.format(i) for i in range(8)]
all_opts[-1].opto_disallow_prev_token_only_label = True


### Reject lottery ticket
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'nol0h147'
# Through visualize_runs.py, we found that these are the three heads that have any PT behavior
# By ablating them, we provide evidence against the hypothesis that ICL may emerge due to a "lottery ticket"
all_opts[-1].opto_ablate_heads = ['0:1', '0:4', '0:7']


jobs = executor.map_array(run_with_opts, all_opts)
