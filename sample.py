#!/usr/bin/env python

from models import hubbard
from mc import metropolis
from contour import *
import argparse
import itertools
import pickle
import sys
import time
from typing import Callable, Sequence

import torch

# Don't print annoying CPU warning.

parser = argparse.ArgumentParser(description="Train contour")
parser.add_argument('model', type=str, help="model filename")
#parser.add_argument('contour', type=str, help="contour filename")
parser.add_argument('-r', '--replica', action='store_true',
                    help="use replica exchange")
parser.add_argument('-nrep', '--nreplicas', type=int, default=30,
                    help="number of replicas (with -r)")
parser.add_argument('-maxh', '--max-hbar', type=float,
                    default=10., help="maximum hbar (with -r)")
parser.add_argument('-N', '--samples', default=-1, type=int,
                    help='number of samples before termination')
parser.add_argument('-S', '--skip', default=30, type=int,
                    help='number of steps to skip')
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--seed-time', action='store_true',
                    help="seed PRNG with current time")
#parser.add_argument('--dp', action='store_true',
#                    help="turn on double precision")                    
parser.add_argument('-T', '--thermalize', default=0,
                    type=int, help="number of MC steps (* d.o.f) to thermalize")
args = parser.parse_args()

seed = args.seed
if args.seed_time:
    seed = time.time_ns()
torch.manual_seed(seed)

with open(args.model, 'rb') as f:
    model = eval(f.read())

'''
with open(args.contour, 'rb') as f:
    contour, contour_params = pickle.load(f)

contour_ikey, chain_key = jax.random.split(key, 2)
'''

V = model.dof
skip = args.skip

if args.skip == 30:
    skip = V

'''
if type(contour) == AffineContour or type(contour) == NearestNeighborAffineContour or type(contour) == PeriodicAffineContour:
    @jax.jit
    def Seff(x, p):
        j = jax.jacfwd(lambda y: contour.apply(p, y))(x)
        logdet = jnp.log(j.diagonal().prod())
        xt = contour.apply(p, x)
        Seff = model.action(xt) - logdet
        return Seff

else:
'''


#@jax.jit
def Seff(x):
    # j = jax.jacfwd(lambda y: contour.apply(p, y))(x)
    # s, logdet = jnp.linalg.slogdet(j)
    # xt = contour.apply(p, x)
    Seff = model.action(x)  # - jnp.log(s) - logdet
    return Seff


#@jax.jit
def observe(x):
    phase = torch.exp(-1j*Seff(x, p).imag)
    # phi = contour.apply(p, x)
    return phase, model.observe(x)


'''
if args.replica:
    chain = replica.ReplicaExchange(lambda x: Seff(x, contour_params), jnp.zeros(
        V), chain_key, delta=1./jnp.sqrt(V), max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
else:
'''
chain = metropolis.Chain(lambda x: Seff(
    x), torch.zeros(V),  delta=1./torch.sqrt(V))

chain.calibrate()
chain.step(N=args.thermalize*V)
chain.calibrate()
try:
    def slc(it): return it
    if args.samples >= 0:
        def slc(it): return itertools.islice(it, args.samples)

    for x in slc(chain.iter(skip)):
        phase, obs = observe(x)
        obsstr = " ".join([str(x) for x in obs])
        print(f'{phase} {obsstr} {chain.acceptance_rate()}', flush=True)

except KeyboardInterrupt:
    pass
