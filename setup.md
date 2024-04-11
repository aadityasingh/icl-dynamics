# Installation guide

Earlier versions of the code were run on V100 GPUs using CUDA 11.4. Newer versions were tested on A100 GPUs using CUDA 12.0. We include instructions for both.

For running only on CPUs, we suggest modifying some of the scripts below. Namely, one would want to install the CPU-compatible version of JAX. Code is tested and works with JAX 0.4.26.

## CUDA 12.0 instructions

```
conda create --name iclmi python=3.10
conda activate iclmi

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install numpy scipy matplotlib tqdm h5py argparse nptyping equinox optax wandb
```

At this point, we were able to reproduce all results in [What needs to go right for an induction head?](https://arxiv.org/abs/2404.07129), using the instructions from [the readme](README.md), on an A100 GPU using CUDA 12.0. We were also able to re-extract omniglot features used for all papers using [omni_features_extract.py](omni_features_extract.py).

Note, at the time of writing this guide/publishing this code, the most recent JAX version was *0.4.26*, and the code is thus cmopatible with this version. 

### Some (possibly useful) notes

The code was originally developed in jax 0.4.8, which doesn't support CUDA 12, and thus we needed to update the code. We include this short history since, as JAX and CUDA versions progress, we presume that another update to the code will be necessary, since 0.4.26 will no longer support CUDA, and so we'll have to update to the newest JAX which will come with its own list of deprecations. We found the [JAX changelog](https://jax.readthedocs.io/en/latest/changelog.html) very useful when updating our code, and include the link here so it may help others debug.

Currently, we get a deprecation warning on the use of `tree_map`. We flag this here since it may create issues in newer versions of JAX.

Optionally, to get a `jupyter lab` setup (where the `iclmi` environment can be used as a kernel), one can use these additional steps after the above (taken from [here](https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab)):
```
conda install ipykernel
ipython kernel install --user --name=iclmi
pip install ipywidgets
```

## CUDA 11.4 instructions (older)

Earlier, I was running on V100 GPUs that have CUDA 11.4, as a result, some special things need to be done.

We can only use an older PyTorch version, which means an older Python version (see [link](https://pytorch.org/blog/deprecation-cuda-python-support/)), so we start with:

```
conda create --name iclmi python=3.10
conda activate iclmi
```

Next, we install [pytorch](https://pytorch.org/get-started/previous-versions/#v182-with-lts-support). Note, according to [link](https://github.com/pytorch/pytorch/issues/75992), cuda11.3 pytorch is compatible with 11.4.

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Next, we install jax, according to [link](https://jax.readthedocs.io/en/latest/installation.html). Note, we have to use jax version 0.4.16 to avoid [this error](https://github.com/google-research/multinerf/issues/139).

```
pip install --upgrade "jax[cuda11_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note we still get a ptxas warning, not sure how to deal with it.

And finally, the rest of the packages
```
pip install numpy scipy matplotlib tqdm h5py argparse jaxtyping nptyping equinox optax wandb
```

At this point we were able to run the code on a V100 GPU using CUDA 11.4,