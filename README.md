# In context learning dynamics in transformers

## Table of Contents

1. [Overview](#overview)
2. [Codebase structure](#code)
2. [Data Setup](#data_setup)
3. [Artificial optogenetics framework](#artificial_opto) -- if you're looking to use this code for your work on different data setup/tasks, skip to this section.
3. [Reproducing the transience paper](#transience)
4. [Reproducing the induction heads paper](#ih_paper)
5. (in progress) [mechanistic explanation of transience](#paper3)
7. [Installation](#installation)
8. [Contributors](#contributors)

<a name="overview"></a>
## 1. Overview 

This is the codebase for a sequence of work investigating the dynamics of in-context learning (ICL) in transformers, through empirical and mechanistic lenses. This project started with the finding that ICL is often transient, and then dove into mechanistic interpretations of dynamics. The codebase relies extensively on [JAX](https://jax.readthedocs.io/en/latest/) and [Equinox](https://docs.kidger.site/equinox/all-of-equinox/).

Our first paper, [The Transient Nature of Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2311.08360), demonstrates that emergent ICL may disappear when overtraining. We demonstrated this on setups previously shown to heavily incentivize ICL over alternative, in-weights learning (IWL) strategies. See [data setup](#data_setup) for more detail. 
This establishes emergent in-context learning as a **dynamical** phenomenon, as opposed to an asymptotic one, which motivated the subsequent works on mechanistic understandings of the dynamics of formation. This first work was primarily done in 12L (~893K params) transformers. See the [corresponding section](#transience) for more details.

Our second paper, [What needs to go right for an induction head?](https://arxiv.org/abs/2404.07129), studies emergence dynamics of [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) on synthetic data that *requires ICL* to solve the task. We pursued this direction (rather than directly targeting the dynamics of transience in setups where ICL or IWL can solve the task) since we felt that there was more to be understood on the nature of induction heads and the dynamics of their formation before we were ready to tackle transience. This work also helped us build out our mechanistic toolkit, which is discussed further in the [artificial optogenetics framework section](#artificial_opto). See the [corresponding section](#ih_paper).

Our third paper is in progress (results mostly finalized) and identifies a new mechanism relevant in understanding the transience phenomenon. This codebase will be updated when the work is posted to Arxiv.

Our fourth paper is also in progress (still figuring out some results) and aims connects the pieces to provide a mechanistic understanding of ICL transience. This codebase will be updated when the work is posted to Arxiv.

As it covers the above works, the codebase has many parts. We highlight the key pieces and provide more detail in the subsections for reproducing each paper. Most files are also supplemented with comments to aid understanding.

We're very excited to start getting this work out, and if you're also interested (or have questions about the code), feel free to reach out -- aaditya.singh.21@ucl.ac.uk. 

<a name="code"></a>
## Codebase structure

General codebase sturcture:
- [main_utils.py](main_utils.py): contains basically all argparse functionality. Each parameter has a dedicated help string which can be used to understand it. Many of the options are unused/set to their default values for the papers, but we found useful to play around with in earlier stages of the project to build intuition.
- [main.py](main.py): Used to run training experiments. Contains training code (e.g., loss computation and `train_step` -- this is where the jitting happens). Uses the functions from `main_utils` to create dataset, model, etc. and run training + evals throughout training.
- [samplers.py](samplers.py): JAX-based data sampling for our synthetic setup. 
- [models.py](models.py): Implements causal transformer models using our artificial optogenetics framework, allowing for easy recording and manipulation of intermediate activations
- [opto.py](opto.py): Contains the options and implementations of various optogenetic manipulations we used for some of the work. Argparse arguments for optogenetic variations are added here. The general idea was to have this be similar intuitively to a [visitor pattern](https://web.mit.edu/6.031/www/sp22/classes/27-little-languages-2/), where to add optogenetic manipulations, all one has to do is modify opto.py. See [Artificial optogenetics framework](#artificial_opto) for more detail.

Generally, the codebase makes use of a lot of functional programming, as is common with JAX codebases.

### Random seed guarantees

Part of why we used JAX is to ensure good random seed reproducibility (similar to other [JAX-based transformer frameworks](https://levanter.readthedocs.io/en/latest/Levanter-1.0-Release/#reproducibility-bitwise-determinism-with-levanter-and-jax)). To this end, we have a few different seeds (listed in [main_utils.py](main_utils.py), but repeated here):
- `init_seed`: Used to initialize model (when training from scratch)
- `train_seed`: Used to generate training data (via [samplers.py](samplers.py)). Also used for things like dropout etc. if those are used (none of our experiments used these features)
- `eval_seed`: Used to generate eval data. We also have an option to directly read in pre-constructed eval data (`load_eval_data`). Also used for things like dropout etc. if those are used (none of our experiments used these features)

See lines ~335-336 in [main.py](main.py) to see how the latter seeds get split into data and model. 

When checkpoints are saved, we save the three relevant seeds (see line ~435 in [main.py](main.py)).

Note, the way our training process works is that a seed is used at every step to generate a batch. This means if the batch size changes, the exact sequence of examples seen by the model will change. A batch size of 32 was used for all experiments. Tests with varying `train_seed` did not show much variance.

<a name="data_setup"></a>
## Setup (data: [samplers.py](samplers.py), overall: [main.py](main.py))

Our setups builds off that introduced by Chan et al. (2022), in [Data Distributional Properties Drive Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2205.05055). We are grateful for the authors of that work for open-sourcing their [code](https://github.com/google-deepmind/emergent_in_context_learning). We used their code for early experiments, but ended up creating our own repository tailored to our analyses. Our work also uses JAX, but relies on Equinox instead of Haiku. We found the PyTree formalism of Equinox easier to work with, especially for [artificial optogenetics](#artificial_opto).

Our data generator assumes a set of classes. Each class can be composed of one or more exemplars. Sequences are composed of a _context_ of exemplar-label pairs, followed by a _query_ exemplar, for which the model needs to output a label. For sampling ([samplers.py](samplers.py)), we disentangle sampling class sequences for the context (`get_constant_burst_seq_idxs`) from exemplars within each class (`get_exemplar_inds`). Though all our experiments tended to just use a single form of class sampling, we offer a way to mix samplers (`get_mixed_seq_idxs`), which could be used to reproduce the experiments with varying p(bursty) in [Chan et al. (2022)](https://arxiv.org/abs/2205.05055) or for other experiments. To support "ICL-only" sequences, we offer `fewshot_relabel` which changes the class labels to be random across contexts (but consistent within a sequence). These sequences force the network to use ICL, which we found useful for our work studying [What needs to go right for an induction head?](https://arxiv.org/abs/2404.07129). Finally, our data samplers work by sampling class and exemplar indices, and only indexing to the data matrix (of dimension `# classes x # exemplars x input_dim`) at the last step. Our process is also end-to-end JIT-able. See [samplers.py](samplers.py) for more detail.

[main.py](main.py) uses the argparse options from [main_utils.py](main_utils.py) to construct the training and eval data iterators, the model, etc. It also contains the train and eval steps and conducts training. We support saving and loading from checkpoints at custom schedules (we found this useful to e.g., upsample checkpoints during a phase change). This is also where the JIT-ing happens (via `eqx.filter_jit`).

<a name="artificial_opto"></a>
## Artificial Optogenetics framework ([models.py](models.py))

A key contribution of our work is the artificial optogenetics framework. This is mostly manifest in [models.py](models.py), which implements a `Transformer` that contains all elements of the framework. We wrap it with `SequenceClassifier` for our specific exemplar-label sequences. All manipulations on top of the framework (for the experiments in our papers) are implemented in [opto.py](opto.py). For full documentation on this portion of the code, see [artificial_optogenetics_guide.md](artificial_optogenetics_guide.md). As always, feel free to reach out with questions or collaborations -- aaditya.singh.21@ucl.ac.uk.

<a name="transience"></a>
## Reproducing [The Transient Nature of Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2311.08360)

Most of the runs in this paper were conducted with the [original codebase](https://github.com/google-deepmind/emergent_in_context_learning) from Chan et. al. (2022). Namely, the runs using a jointly trained Resnet encoder (which is most of the results). This codebase was used for the remaining runs -- those with fixed LLaMa embedding vectors as exemplars (Section 4.3), and those with fixed Omniglot embeddings (Appendix C). 

LLaMa embedding vectors (extracted from LLaMa 1 open-source weights) were clustered using [FAISS](https://github.com/facebookresearch/faiss) using the procedure in the paper and then turned into h5 files (with dimensions `# classes x # exemplars x input_dim`, where `input_dim` varies based on LLaMa source model size). An example sweep file operating on these h5's is [llama_sweep_example.py](llama_sweep_example.py).

Omniglot embeddings were extracted using [omni_features_extract.py](omni_features_extract.py) and then experiments for Appendix C were run using a sweep file like [fixed_omni_emb_sweep_example.py](fixed_omni_emb_sweep_example.py). This file may be of use to see how the evaluators were structured.

### Cite

If citing these evaluators/experiments, please use:

```
@misc{singh2023transient,
      title={The Transient Nature of Emergent In-Context Learning in Transformers}, 
      author={Aaditya K. Singh and Stephanie C. Y. Chan and Ted Moskovitz and Erin Grant and Andrew M. Saxe and Felix Hill},
      year={2023},
      eprint={2311.08360},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<a name="ih_paper"></a>
## Reproducing [What needs to go right for an induction head?](https://arxiv.org/abs/2404.07129)

This codebase was largely made to support this paper. The dataset uses pre-processed omniglot features, similar to the transience paper. Namely, [omni_features_extract.py](omni_features_extract.py) was used to extract features. For simplicity, only 5 exemplars per class were processed (as the paper only uses 1 exemplar per class for training, and the remaining 4 for one of the test sets). These features were then re-ordered randomly to form the data file `omniglot_resnet18_randomized_order_s0.h5`, provided in the codebase. We directly provide this file to enable researchers without access to GPUs to quickly get started with the codebase -- all experiments for this paper can be run on a laptop!

To reproduce the figures of the paper, one should first run [ih_paper_runs.sh](ih_paper_runs.sh), which contains all the relevant runs for the paper (for a given initialization seed). Then, one can use the [ih_paper_plots.ipynb](ih_paper_plots.ipynb) to reproduce all figures from the paper.

Those files minimally reproduce the paper. To additionally obtain all appendix results, run [ih_paper_appendix_runs.sh](ih_paper_appendix_runs.sh) and [ih_paper_appendix_plots.ipynb](ih_paper_appendix_plots.ipynb).

All of the above rely on [ih_paper_plot_utils.py](ih_paper_plot_utils.py) for some utils (e.g., a simplified forward function wrapper).

For the toy model of phase changes, we have a separate file [simple_model_solver.py](simple_model_solver.py). This file is called in the scripts above to generate the corresponding figures. It is completely independent of the rest of codebase, and may also be useful to those looking to further study toy models with clamping and/or progress measures.

### Additional results on different initialization seeds

We include notebook copies of [ih_paper_plots.ipynb](ih_paper_plots.ipynb) that plot results from runs we did on different initialization seeds in the folder `ih_paper_additional_seeds`. To actually run these notebooks, one would have to run [ih_paper_runs.sh](ih_paper_runs.sh) with other seeds, then move the notebook to the top-level folder and run it. Our intent with these notebooks is just to share additional results showing qualitative reprodubility of the observed phenomenon.

### Cite

```
@misc{singh2024needs,
      title={What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation}, 
      author={Aaditya K. Singh and Ted Moskovitz and Felix Hill and Stephanie C. Y. Chan and Andrew M. Saxe},
      year={2024},
      eprint={2404.07129},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<a name="paper3"></a>
## Reproducing paper 3

In progress. Check back later.

<a name="installation"></a>
## Installation

See [setup.md](setup.md) for instructions for various CUDA driver versions.

<a name="contributors"></a>
## Contributors

The primary creator and maintainer of this code was Aaditya Singh. Ted Moskovitz also contributed to various parts (model, data samplers, training). The code was based off an earlier transformer implementation by Erin Grant. A special thanks to Stephanie Chan, who all this work was done in close collaboration with. The overall project was supervised by Andrew Saxe and Felix Hill, with Andrew Saxe also contributing some code for the tensor product toy model. 
