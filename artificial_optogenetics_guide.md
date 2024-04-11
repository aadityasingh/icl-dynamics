# Artificial Optogenetics

Desiderata: A transformer model implementation that makes it easy to access and manipulate activations throughout training.

The first parts of this guide go through the general structures we used to satisfy these desiderata. As far as we can tell, these are not specific to transformers, though our implementation is. The latter parts are more specific to transformers. 

## How it works ([models.py](models.py))

### `call_with_all_aux` for intermediate activations

All forward passes are implemented in a `call_with_all_aux` method, which returns all intermediate activations as a nested PyTree. This allows re-use of JAX utils for working with PyTrees (e.g., `jax.tree_map`), as well as folding well into Equinox's philosophy of "everything is a PyTree".

When a "parent" module contains submodules, the parent's output dictionary contains sub-trees for the children submodules.

We follow a naming convention for these dictionaries:
- `'out'`: always used for the output of the given network element
- `'*_output(s)'`: used when nesting one block within another (for example, the transformer block has a field called `'attn_output'` to store the output of its nested self-attention layer)

The actual `__call__` function then wraps `call_with_all_aux` and just returns the output of the module.

It may seem that returning these dictionaries is inefficient. This is where `jax.jit` comes in: when using `jax.jit` (or specifically, `eqx.filter_jit` in our case), we didn't observe significant slow-downs compared to earlier versions of the code (that didn't use `call_with_all_aux`). We belive that jit recognizes/minimizes the overhead from activations that are returned but then not used. Intuitively, one can think of this approach compared to prior works as: whereas other frameworks require adding hooks to observe intermediates, our framework natively returns all intermediates, and relies on jax.jit not tracking unused quantities.

That said, we don't provide performance guarantees on very large scale setups. The largest models we trained were 12L transformers (839k params), for the transience paper, which we were able to train on a single 32GB V100 GPU. 

### `cache` and `cache_mask` for manipulating activations

A key aspect of our work is not just recording intermediate activations, but
manipulating them. To allow this, call_with_all_aux accepts a cache and cache_mask
which should be of the same format as the output dictionary (missing entries are ignored).
This cache is what allows prespecificationof certain activations in the network, and 
computing off of them. The cache_mask allows single neuron level manipulations.

We then generalize the notion of assignment (normally done using the native `=`) to be *cache-aware*. Specifically, each call_with_all_aux method starts by creating its return dictionary `r` and an assignment function `assign_fn`. For example, instead of setting:

`q = wq(x)`,

where `wq` is for example a linear layer, we would instead do:

`assign_fn('q', wq(r['x']))`

One can thus view `r` as a version of the python native `locals()`. This syntax of looping through `r` as the local context and `assign_fn` as the assignment operator allows for cache-aware assingment with relatively little code overhead (see examples in [models.py](models.py)). 

The internal working of `assign_fn` is extremely simple: it simply checks if the element being assigned (e.g., `'q'`) is present in the cache, and if it is, it overrides the value being assigned according to the cache and cache_mask. We create `assign_fn` using the `make_cache_assign` function, which wraps up `r`, `cache`, and `cache_mask` so they don't have to be passed every time (making it simpler to write code).

*NOTE:* `assign_fn` should only be used for assigning tensors.

Some care needs to be taken when calling sub-modules. For example, the transformer block has a nested self-attention layer. To ensure propagation of the cache, it's important to pass the sub-dictionary (e.g., cache['attn_output']) to the submodule being called. Since the submodule will then factor in the cache, we use direct assignment: 

```
r['attn_output'] = self.attn.call_with_all_aux(r['norm_inp'], 
												key=keys[0], 
												cache=cache.get('attn_output', dict()), cache_mask=cache_mask.get('attn_output', dict()))
```

Another example of care when calling sub-modules can be found by looking at how `Transformer` calls and stores outputs from `TransformerBlock` submodules. 

### Sharp bits/extra details

This framework requires some care on the model developer's part. When adding new modules or editing modules,
one must remember to use `assign_fn` (where possible) and `r`. 

Furthermore, when dealing with submodules, as mentioned earlier, its important to properly propagate the cache down and only use `assign_fn` on "atomic" values. 

For example, if we want to modify layer 2 of a 3 layer model, the relevant part of the cache would look like `[dict(), modify, dict()]`. We would want to make sure each element of the cache gets properly to the submodule for that layer, and then in this rare case use `'='` to assign `r['layer_outputs']` to be a list.

When working with `cache` objects, we recommend using functions such as `jax.tree_map` as is standard with PyTrees.

### Specific output and `cache` structure for our implemenation

For completeness, we provide a complete description of what the cache for an example 2L model with MLPs would look like in our case in [example_cache_explanation.md](example_cache_explanation.md). Note, our specific "model" is called `SequenceClassifier` and is specific to the paired exemplar-label setting we operate in. The underlying `Transformer` is likely more generally useful to users. Our implementation also contains various dropout layers (legacy remnants from older versions), though these were not used for any experiments and thus correctness is not guaranteed.

## How to use it, and how we used it ([opto.py](opto.py))

The idea behind [models.py](models.py) and the above explanation is to provide a general framework for JAX-based interpretability. Of course, for specific projects and needs, different things will be done on top of the framework.

All optogenetic manipulations (e.g., ablating heads) are done in [opto.py](opto.py). Our philosophy with this file was that of a [visitor pattern](https://web.mit.edu/6.031/www/sp22/classes/27-little-languages-2/). For example, this file has an `add_args_to_parser` method which adds optogenetic specific args to the argparse. This allows for easily adding new arguments without modifying any other code file. Similarly, we provide a wrapper formalism for optogenetic manipulations via the `make_fn_from_opts` method. This method takes in all specified options and returns a forward function of the form `call(model, x, y, key)`. This function operates on a single example level, and then can be vmapped for use in training and inference. This abstractions removes the need for calling files (e.g., [main.py](main.py)) to reference the underlying workings of [opto.py](opto.py).

Our optogenetic manipulations were mainly used in the [What needs to go right for an induction head?](https://arxiv.org/abs/2404.07129) paper. As a result, many of them assume that the input sequences are 5 tokens long and have the format exemplar-label-exemplar-label-query. Many optogenetic manipulations rely on two (or more) passes through the network -- in the first, various activations are computed using the unperturbed network. These activations are then used to populate the cache, which is then used during a second pass through the network. This is, for example, how pattern- and value-preserving ablations are coded up.

Beyond these python files, we strongly suggest users to go through the [ih_paper_appendix_runs.sh](ih_paper_appendix_runs.sh) script and [ih_paper_appendix_plots.ipynb](ih_paper_appendix_plots.ipynb) notebook, which really demonstrate how simple it is to do various analyses. The main paper files also have useful examples, but the appendix contains more uses which could be of interest for future work.

Note: Most comments in [opto.py](opto.py) refer to the first two layers of the model as "layer 0" and "layer 1" (essentially, 0-indexing).

## Possible extensions

These are a smattering of TODOs/general ideas we had but didn't end up doing. Including here
in case its useful to people building off this code:
- Formalize some of the defaults by implementing a subclass of [`eqx.Module`](https://docs.kidger.site/equinox/api/module/module/) called `ModuleWithAux` that enforces a `call_with_all_aux` method and implements `__call__(self, **kwargs)` as a wrapper around `call_with_all_aux` that just returns the output. Then, all modules would inherit `ModuleWithAux` instead of `eqx.Module`. One could also enforce using `r` and `assign_fn` somehow.
- Currently, we don't look at intermediate activations in MLPs. Our focus was on induction heads and attention layers so far, so this wasn't necessary. Furthermore, we currently don't use (SwiGLU)[https://arxiv.org/abs/2002.05202v1], but given its increasing prevalence it could be cool to consider/add.
- Converting an existing framework used for LLMs to this formalism, allowing for easier interpretability work moving forward. Due to our limited computational resources, we did not do experiments on frontier LLMs.
- Provide better support for [path patching](https://arxiv.org/abs/2304.05969). We hadn't done this as we didn't use this method. However, it should be relatively easy to do:
	- add parameters to the `opto_opts` for what input sequence to use for patching which activations to patch
	- add support in the `standard_call` function or in a separate `opto_specific_fn` for setting the activations by using two forward passes
	- Interestingly, path patching could also be applied during training. We haven't thought in depth about the uses of this functionality, but welcome future work in this direction.

## Cite

To cite this framework, use the following:
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

