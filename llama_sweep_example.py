'''
This file was used to run the experiments in Section 4.3 of the transience paper:
Namely, those on LLaMa 1 token embeddings, pre-clustered into "classes" using FAISS
'''
import submitit
from copy import deepcopy
import h5py
import pdb

from main_utils import create_parser
from main import run_with_opts
import opto

base_path = '/path/to/save'

parser = create_parser()
opto.add_args_to_parser(parser)

run_name = 'llama'
opts = parser.parse_args(
  ['--run', run_name,
    '--data_type', 'file', '--data_file', 'TO_SWEEP',
    '--pos_embedding_type', 'sinusoidal', '--sin_time', '30',
    '--mlp_ratio', '4', '--depth', '12', '--d_model', '64', '--num_heads', '8',
    '--init_seed', '2023', '--train_seed', '2',
    '--mixing_coeffs', '1.0', '--pt_burstiness', '3', '--train_context_len', '8', '--fs_relabel', '0',
    '--class_split', '1600', '23', '0', '--exemplar_split', '10', '0', '0',
    '--train_iters', '64000000', '--eval_every', '100000', '--ckpt_every', '10000000',
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

for model_size in ['7B', '13B', '30B', '70B']:
  for ex, c in [(10, 1600), (5, 3200)]:
    # Clustering seed (aka changes underlying dataset classes)
    for seed in [4,5]:
      # Could add more init seeds
      for init_seed in [5, 6]:
        all_opts.append(deepcopy(opts))
        all_opts[-1].run = 'llama1_{}_furthest_c{}_s{}_is{}'.format(model_size, c, seed, init_seed)
        print(all_opts[-1].run)
        all_opts[-1].data_file = f'{base_path}/embeddings/llama/llama1_{model_size}_embeds_furthest.h5'
        print(all_opts[-1].data_file)
        all_opts[-1].data_path_in_file = '{}/{}/feat'.format(seed, ex)
        print(all_opts[-1].data_path_in_file)
        f = h5py.File(all_opts[-1].data_file, 'r')
        data_shape = f[all_opts[-1].data_path_in_file].shape
        f.close()
        assert data_shape[1] == ex
        all_opts[-1].class_split = [c, data_shape[0]-c, 0]
        all_opts[-1].exemplar_split = [ex,0,0]
        print("Class split: {}, Exemplar split: {}".format(all_opts[-1].class_split, all_opts[-1].exemplar_split))
        all_opts[-1].init_seed = init_seed
        print('-'*30)

jobs = executor.map_array(run_with_opts, all_opts)

