'''
This file can be used to get all the runs for the main text of the Strategy Coopetition paper.

It assumes that `omni_features_extract.py` has been run already, with the correct base_path.

We format as a submitit file rather than a bash script for ease of parallelization (the general
compute cost for this paper is higher than the Induction Head Dynamics paper, so we recommend
using GPUs). 

To keep things in one file, we first do a few dummy runs to generate eval data for all cases.

Given how some runs depend on checkpoints from others, we then run the "blocking ones" (the main
run and the ciwl-only run).

Once those are completed, we schedule the main body of runs.

We'll use the blocking `outputs = [job.result() for job in jobs]` calls to enforce the bottlenecks above. 
As a result, we recommend running this script in a tmux session so it doesn't get killed.
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

run_make_eval = False
run_first_two = False
run_rest = True

# Toggle this on if you want to use wandb to monitor runs. See main.py for default project name,
# currently set to `icl-transience`
use_wandb = False

base_path = '/path/to/save'
os.makedirs(base_path, exist_ok=True)
os.makedirs(f'{base_path}/runs', exist_ok=True)
os.makedirs(f'{base_path}/log_submitit', exist_ok=True)

parser = create_parser()
opto.add_args_to_parser(parser)

executor = submitit.AutoExecutor(folder=f"{base_path}/log_submitit")
executor.update_parameters(
  name='coopetition',
  nodes=1,
  gpus_per_node=1,
  tasks_per_node=1,
  cpus_per_task=20,
  mem_gb=32,
  timeout_min=1440*4, # 4 days, since the 12L model takes just over 2 days
  slurm_partition='TODO FILL IN',
)

################################################################################
### MAKE EVAL DATA
################################################################################

print("Making eval data (4 jobs)")
all_make_eval_opts = []

# We make all OOD eval data using the first 1600 classes so that we don't need
# to remake it for later experiments. Similarly, we only use the first 5 exemplars
# per class (since some of our data ablations in the appendix go down to 5 exemplars)
eval_opts = parser.parse_args(
  ['--raw_name', '--base_folder', f'{base_path}/runs', '--suppress_output',
    '--data_type', 'file', '--data_file', f'{base_path}/embeddings/omniglot_features_norotate.h5',
    '--pos_embedding_type', 'ape', '--depth', '2', '--d_model', '64', '--num_heads', '8',
    '--init_seed', '5', '--train_seed', '2',
    '--mixing_coeffs', '1.0', '--pt_burstiness', '1', '--train_context_len', '2', '--fs_relabel', '0',
    '--class_split', '1600', '0', '23', '--exemplar_split', '5', '0', '15',
    '--train_iters', '101', '--eval_every', '50', '--eval_iters', '5000',
    '--lr', '0.00001', '--optimizer', 'adam', 
    '--pe_names', 'train_eval', 'iwl_copy_avail', 'flip_icl', 'icl', 'pure_iwl',
    '--pe_classes', 'train', 'train', 'train', 'train', 'train',
    '--pe_exemplars', 'train', 'train', 'train', 'train', 'train',
    '--pe_burstiness', '1', '0', '1', '1', '0',
    '--pe_fs_relabel_scheme', 'None', 'None', 'flip', '01', 'None',
    '--pe_no_support', '0', '1', '0', '0', '1', 
    '--pe_unique_rest', '0', '1', '0', '0', '1', 
    '--pe_assign_query_label_random', '0', '1', '0', '0', '0',
    '--save_eval_data', 'eval_data.h5']
)

all_make_eval_opts.append(deepcopy(eval_opts))
all_make_eval_opts[-1].run = 'default_eval'

# For the 12L model, we have ctx 8 sequences and need to adapt the evaluators accordingly.
# We use all 20 exemplars, and all 12800 classes for full parity to the transience paper.
ctx8_eval_opts = parser.parse_args(
  ['--raw_name', '--base_folder', f'{base_path}/runs',
    '--suppress_output',
    '--data_type', 'file', '--data_file', f'{base_path}/embeddings/omniglot_features_reordered.h5',
    '--pos_embedding_type', 'ape', '--depth', '2', '--d_model', '64', '--num_heads', '8',
    '--init_seed', '5', '--train_seed', '2',
    '--mixing_coeffs', '1.0', '--pt_burstiness', '3', '--train_context_len', '8', '--fs_relabel', '0',
    '--class_split', '12800', '0', '184', '--exemplar_split', '20', '0', '0',
    '--train_iters', '101', '--eval_every', '50', 
    '--lr', '0.0003', '--optimizer', 'adam', 
    '--pe_names', 'b2_train_eval', 'b3_train_eval', 'iwl_copy_avail', 'flip_icl', 'icl', 'iwl',
    '--pe_classes', 'train', 'train', 'train', 'train', 'train', 'train',
    '--pe_exemplars', 'train', 'train', 'train', 'train', 'train', 'train',
    '--pe_burstiness', '2', '3', '0', '4', '4', '0',
    '--pe_fs_relabel_scheme', 'None', 'None', 'None', 'flip', '01', 'None',
    '--pe_no_support', '0', '0', '1', '0', '0', '1', 
    '--pe_unique_rest', '0', '0', '1', '0', '0', '1', 
    '--pe_assign_query_label_random', '0', '0', '1', '0', '0', '0',
    '--save_eval_data', 'eval_data.h5']
)
all_make_eval_opts.append(deepcopy(ctx8_eval_opts))
all_make_eval_opts[-1].run = 'ctx8_eval'

### We need to run the above to make the eval data
if run_make_eval:
    jobs = executor.map_array(run_with_opts, all_make_eval_opts)
    # Wait for them to finish
    outputs = [job.result() for job in jobs]
print("Finished making eval data!")


################################################################################
### FIRST TWO RUNS (BLOCKING)
################################################################################
print("First two runs (main + ciwl_only)")

# We don't specify 'run', as this should be specified later
# Note that here we use 64M iters. Most plots in the paper used 20M iters,
# but we include the longer run here for reference
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


### Main run
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'main'
all_opts[-1].ckpt_every = 50000 # frequent checkpoints for posthoc analyses


### CIWL-only run (needed for checkpoints later)
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'ciwl_only'
# Assign data properties to train on CIWL-only data
all_opts[-1].assign_query_label_random = True
all_opts[-1].pt_burstiness = [0]
all_opts[-1].pt_no_support = [1]
all_opts[-1].pt_unique_rest = [1]
# Take frequent checkpoints for use later
all_opts[-1].ckpt_every = 50000


### We need the above two runs to to do the rest, so we send them first
if run_first_two:
    jobs = executor.map_array(run_with_opts, all_opts)
    # Wait for them to finish
    outputs = [job.result() for job in jobs]
print("Finished first two mandatory runs (main + ciwl_only)!")
# and clear the queue to do the rest
all_opts = []

################################################################################
### REMAINING RUNS
################################################################################
print("Remaining main paper runs (25 runs)")

### ICL-only run
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'icl_only'
# Assign data properties to train on ICL-only data
all_opts[-1].fs_relabel = 12800
# For the paper, we actually ran this for _way longer_ but the loss plateau
# is _very long_. We comment these out to make reproduction faster.
# all_opts[-1].train_iters = 640000000
# all_opts[-1].ckpt_every = 64000000


### ICL-only run, with second half grafted in from main run
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'icl_only_graftl1'
all_opts[-1].opto_graft_out_model_ckpt = f'{base_path}/runs/main/checkpoints/00020000000.eqx'
all_opts[-1].opto_graft_out_model_cfg = f'{base_path}/runs/main/config.json'
all_opts[-1].opto_graft_out_model_from_layer = 0
# Assign data properties to train on ICL-only data
all_opts[-1].fs_relabel = 12800


### Patch in different second halves of the network from a CIWL-
### only run, then train on ICL-only data
# When first doing the experiments we ran a full binary search.
# Here we just use the checkpoints we know to matter/based on
# the boundaried found by the binary search
ckpts_to_graft = {'4M': '00004000000.eqx',
                  '8M': '00008000000.eqx',
                  # The first boundary
                  '9.75M': '00009750016.eqx',
                  '9.8M': '00009800000.eqx',
                  # equally spaced "inside" points
                  '15M': '00015000000.eqx',
                  '20M': '00020000000.eqx',
                  '25M': '00025000000.eqx',
                  # The second boundary
                  '31.5M': '00031500000.eqx',
                  '32M': '00032000000.eqx',
                  # equally spaced "outside" points
                  '48M': '00048000000.eqx',
                  '64M': '00064000000.eqx'}
for ckpt_name in ckpts_to_graft:
    all_opts.append(deepcopy(opts))
    all_opts[-1].run = 'icl_only_graftl1_from_ciwl_only_{}'.format(ckpt_name)
    all_opts[-1].opto_graft_out_model_ckpt = f'{base_path}/runs/ciwl_only/checkpoints/{ckpts_to_graft[ckpt_name]}'
    all_opts[-1].opto_graft_out_model_cfg = f'{base_path}/runs/ciwl_only/config.json'
    all_opts[-1].opto_graft_out_model_from_layer = 0
    # Train on ICL only data
    all_opts[-1].fs_relabel = 12800

### Train on bursty data, starting at different checkpoints from
### CIWL-only run
ckpts_to_init = {'4M': '00004000000.eqx',
                 '5M': '00005000000.eqx',
                 '6M': '00006000000.eqx',
                 '7M': '00007000000.eqx',
                 '8M': '00008000000.eqx',
                 '9M': '00009000000.eqx',
                 '10M': '00010000000.eqx',
                 '12M': '00010000000.eqx',
                 '16M': '00016000000.eqx'}
for ckpt_name in ckpts_to_init:
    all_opts.append(deepcopy(opts))
    all_opts[-1].run = f'init_from_ciwl_only_{ckpt_name}'
    # Run for 32M more iters, no matter where the start was
    # We originally did 64M iters, but truncate to make reproducing quicker
    all_opts[-1].train_iters = int(32000000 + float(ckpt_name[:-1])*1000000)
    all_opts[-1].load_from_ckpt = f'{base_path}/runs/ciwl_only/checkpoints/{ckpts_to_init[ckpt_name]}'
    all_opts[-1].load_from_ckpt_cfg = f'{base_path}/runs/ciwl_only/config.json'
    # Don't restore optimizer state or seeds (latter doesn't matter) to avoid leaking
    all_opts[-1].no_load_opt_state = True
    all_opts[-1].no_load_seeds = True

### MQ run
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'mq'
all_opts[-1].match_query_and_distractors = True

### 12L baseline run

### Make eval data separately for baseline and MQ

opts_12l = parser.parse_args(
  ['--raw_name', '--base_folder', f'{base_path}/runs',
    '--suppress_output',
    '--data_type', 'file', '--data_file', f'{base_path}/embeddings/omniglot_features_reordered.h5',
    '--pos_embedding_type', 'sinusoidal', '--sin_time', '30',
    '--mlp_ratio', '4', '--depth', '12', '--d_model', '64', '--num_heads', '8',
    '--init_seed', '5', '--train_seed', '2',
    '--mixing_coeffs', '1.0', '--pt_burstiness', '3', '--train_context_len', '8', '--fs_relabel', '0',
    '--class_split', '12800', '0', '184', '--exemplar_split', '20', '0', '0',
    '--train_iters', '400000000', '--eval_every', '2000000', '--ckpt_every', '40000000',
    '--lr', '0.0003', '--optimizer', 'adam', 
    '--load_eval_data', f'{base_path}/runs/ctx8_eval/eval_data.h5']
)
opts_12l.use_wandb = use_wandb

### 12L run
all_opts.append(deepcopy(opts_12l))
all_opts[-1].run = '12l'

### 12L MQ run
all_opts.append(deepcopy(opts_12l))
all_opts[-1].run = '12l_mq'
all_opts[-1].match_query_and_distractors = True

if run_rest:
    jobs = executor.map_array(run_with_opts, all_opts)
# Wait for them to finish
print("Finished queueing all runs")
