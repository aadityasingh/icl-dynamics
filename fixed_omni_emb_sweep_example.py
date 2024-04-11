'''
This file was used to run the experiments provided in Appendix B of the transience paper:
Namely, those on fixed omniglot embeddings, extracted with omni_features_extract.py
'''
import submitit
from copy import deepcopy
import pdb

from main_utils import create_parser
from main import run_with_opts
import opto

base_path = '/path/to/save'

parser = create_parser()
opto.add_args_to_parser(parser)

run_name = 'omni'
# Note default data file is for 1600 classes
opts = parser.parse_args(
  ['--run', run_name,
    '--data_type', 'file', '--data_file', f'{base_path}/embeddings/omniglot_features_norotate.h5',
    '--pos_embedding_type', 'sinusoidal', '--sin_time', '30',
    '--mlp_ratio', '4', '--depth', '12', '--d_model', '64', '--num_heads', '8',
    '--init_seed', '2023', '--train_seed', '2',
    '--mixing_coeffs', '1.0', '--pt_burstiness', '3', '--train_context_len', '8', '--fs_relabel', '0',
    '--class_split', '1600', '23', '0', '--exemplar_split', '20', '0', '0',
    '--train_iters', '1600000000', '--eval_every', '500000', '--ckpt_every', '40000000',
    '--lr', '0.0003', '--optimizer', 'adam', 
    '--pe_names', 'b2_train_eval', 'b3_train_eval', 'iwl_copy_avail', 'flip', 'fsl_train', 'mem_train_unique',
    '--pe_classes', 'train', 'train', 'train', 'train', 'train', 'train',
    '--pe_exemplars', 'train', 'train', 'train', 'train', 'train', 'train',
    '--pe_burstiness', '2', '3', '0', '4', '4', '0',
    '--pe_fs_relabel_scheme', 'None', 'None', 'None', 'flip', '01', 'None',
    '--pe_no_support', '0', '0', '1', '0', '0', '1', 
    '--pe_unique_rest', '0', '0', '1', '0', '0', '1', 
    '--pe_assign_query_label_random', '0', '0', '1', '0', '0', '0',
    '--use_wandb', '--save_eval_data', 'eval_data.h5']
)

executor = submitit.AutoExecutor(folder=f"{base_path}/log_submitit")
executor.update_parameters(
  name=run_name,
  nodes=1,
  gpus_per_node=1,
  tasks_per_node=1,
  cpus_per_task=10,
  timeout_min=4320,
  slurm_partition='TODO',
  slurm_additional_parameters={'constraint': 'TODO'} 
)

all_opts = []

# Just some example runs.. create similar versions to reproduce paper
all_opts.append(deepcopy(opts))
all_opts[-1].run = 'omniglot_c12800_is6'
# Data file containing 12800 features in right order
all_opts[-1].data_file = f'{base_path}/embeddings/omniglot_features_reordered.h5'
all_opts[-1].class_split = [12800, 184, 0]
all_opts[-1].init_seed = 6

all_opts.append(deepcopy(opts))
all_opts[-1].run = 'omniglot_c1600_is5'
# default class split is 1600, 23, 0
# default init seed is 5

jobs = executor.map_array(run_with_opts, all_opts)

