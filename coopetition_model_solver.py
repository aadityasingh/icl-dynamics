'''
This is a revamped version of `simple_model_solver.py` that's a bit cleaner for the types of experiments
we want to do for the transience paper. Specifically, it abstracts away a bit more of the
boilerplate for tracking individual variables, allowing for more clean specification of loss
functions.

Unlike the `simple_model_solver.py`, which is standalone, this file is meant to provide useful functions 
for use in a jupyter notebook.

Another key difference from `simple_model_solver.py` is the use of a variable aggregation function, 
which defaults to jnp.mean (as it did earlier). We used jnp.mean since it requires less adjustment 
of hypers and keeps norms closer. If using jnp.sum, you would need to adapt the hyperparameters
accordingly.
'''

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm.notebook import tqdm
from functools import partial

def outer_product(*vecs):
    arr = [chr(97+i) for i in range(len(vecs))]
    return jnp.einsum('{}->{}'.format(','.join(arr), ''.join(arr)), *vecs)


def outer_product_loss(params, true, keys, agg_fn=jnp.mean):
    err = outer_product(*[true[k] for k in keys]) - outer_product(*[params[k] for k in keys])
    return 0.5*agg_fn(err**2)


def make_3_loss_fn(agg_fn=jnp.mean):
    return {'loss_fn': partial(outer_product_loss, keys='abc', agg_fn=agg_fn)}


def make_4share2_loss_fn(mu1=0, mu2=0, alpha=0, agg_fn=jnp.mean):
    retval = dict(track_fns=dict())

    retval['track_fns']['Mechanism 1 loss'] = partial(outer_product_loss, keys='abc', agg_fn=agg_fn)
    retval['track_fns']['Mechanism 2 loss'] = partial(outer_product_loss, keys='dbc', agg_fn=agg_fn)
    retval['track_fns']['Cost'] = lambda p, t: 0.5*agg_fn(outer_product(p['a'], p['d'])**2)
    
    retval['loss_fn'] = (lambda p, t: ((retval['track_fns']['Mechanism 1 loss'](p, t) + mu1)
                                        * (retval['track_fns']['Mechanism 2 loss'](p, t) + mu2) 
                                        + alpha*retval['track_fns']['Cost'](p, t))
    )

    return retval


def make_rank_1_vecs_setup(loss_fn, track_fns=None, dims=[10, 10, 10, 10], seed=0):
    '''
    loss_fn should take in true and params
    extra_track_fns should be a dictionary to add
    '''
    retval = dict(init=dict(), true=dict())

    key = jax.random.PRNGKey(seed)
    for i in range(len(dims)):
        # We split the key iteratively to allow for getting the same init vectors for the same seed
        key, this_key = jax.random.split(key)
        k1, k2 = jax.random.split(this_key)
        retval['true'][chr(97+i)] = jax.random.normal(k1, shape=(dims[i],))
        retval['init'][chr(97+i)] = jax.random.normal(k2, shape=(dims[i],))

    retval['loss_fn'] = loss_fn

    retval['track_fns'] = dict() if (track_fns is None) else track_fns
    for j in range(len(dims)):
        l = chr(97+j)
        # Note the l=l is necessary here to "freeze" the value of l in the lambda
        retval['track_fns']['Norm of {}'.format(l)] = ( lambda p, t, l=l: jnp.linalg.norm(p[l]) )
        retval['track_fns']['Grad norm of {}'.format(l)] = ( lambda p, t, l=l: jnp.linalg.norm(p['grad'][l]) )
        retval['track_fns']['Cosine dist {}'.format(l)] = (
            lambda p, t, l=l: 1 - ( jnp.dot(t[l], p[l]) / (jnp.linalg.norm(t[l]) * jnp.linalg.norm(p[l])) ) ** 2)
        retval['track_fns']['Euclid dist {}'.format(l)] = (
            lambda p, t, l=l: jnp.linalg.norm(p[l]-t[l]))

    return retval


def train(setup, clamp_fn = (lambda s, p: p), max_iters = 1000, lr = 1, thresh=1e-6):
    '''
    setup should be a dict with fields init, true, loss_fn, track_fns
    '''
    loss_and_grad_fn = jax.jit(jax.value_and_grad(setup['loss_fn']))

    tracked_vars = {'Train loss': []}
    for fn in setup['track_fns']:
        tracked_vars[fn] = []

    params = setup['init']
    params = clamp_fn(setup, params)
    for i in tqdm(range(max_iters)):
        loss, grad = loss_and_grad_fn(params, setup['true'])
        tracked_vars['Train loss'].append(loss)
        for fn in setup['track_fns']:
            tracked_vars[fn].append(setup['track_fns'][fn]({'grad': grad, **params}, setup['true']))

        if loss < thresh:
            print("Terminating early since loss below threshold at iter {}".format(i))
            break

        params = jax.tree_map(lambda x, y: x - lr*y, params, grad)
        params = clamp_fn(setup, params)
    return tracked_vars


def clamp_rank_1(setup, params, clamp=dict()):
    retval = dict()
    for c in params:
        if c in clamp:
            retval[c] = jnp.where(jnp.arange(params[c].shape[0]) < clamp[c], setup['true'][c], params[c])
        else:
            retval[c] = params[c]
    return retval
