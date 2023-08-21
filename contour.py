#!/usr/bin/env python

from models import hubbard
from mc import metropolis
import pickle
import sys
import time
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, features):
        super(MLP, self).__init__()
        layers = []
        for in_features, out_features in zip(features, features[1:]):
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ])
        self.layers = nn.Sequential(*layers)
        self.apply_weight_init

    def apply_weight_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, nonlinearity='relu')  # He initialization
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


class Contour(nn.Module):
    def __init__(self, volume, features):
        super(Contour, self).__init__()
        self.volume = volume
        # self.mlp_u = MLP(features)
        self.mlp_v = MLP(features)
        # self.dense_y_r = nn.Linear(features[-1], volume)
        self.dense_y_i = nn.Linear(features[-1], volume)

    def forward(self, x):
        # u = self.mlp_u(x)
        v = self.mlp_v(x)
        # y_r = self.dense_y_r(u)
        y_i = self.dense_y_i(v)
        return torch.complex(x, y_i)
        # return x + torch.complex(y_r, y_i)


class RealContour(nn.Module):
    def forward(self, x):
        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train contour",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, help="model filename")
    parser.add_argument('contour', type=str, help="contour filename")
    parser.add_argument('-R', '--real', action='store_true',
                        help="output the real plane")
    parser.add_argument('-i', '--init', action='store_true',
                        help="re-initialize even if contour already exists")
    parser.add_argument('-f', '--from', dest='fromfile', type=str,
                        help="initialize from other file")
    parser.add_argument('-l', '--layers', type=int, default=0,
                        help='number of (hidden) layers')
    parser.add_argument('-w', '--width', type=int, default=1,
                        help='width (scaling)')
    # parser.add_argument('-r', '--replica', action='store_true',
    #                    help="use replica exchange")
    # parser.add_argument('-nrep', '--nreplicas', type=int, default=30,
    #                    help="number of replicas (with -r)")
    # parser.add_argument('-maxh', '--max-hbar', type=float,
    #                    default=10., help="maximum hbar (with -r)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--seed-time', action='store_true',
                        help="seed PRNG with current time")
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('-S', '--skip', default=30, type=int,
                        help='number of steps to skip')
    # parser.add_argument('-A', '--affine', action='store_true',
    #                   help='use affine coupling layer')
    # parser.add_argument('-nnA', '--nnaffine', action='store_true',
    #                    help='use nearest neighbor affine coupling layer')
    parser.add_argument('-N', '--nstochastic', default=1, type=int,
                        help="number of samples to estimate gradient")
    parser.add_argument('-T', '--thermalize', default=0, type=int,
                        help="number of MC steps (* d.o.f) to thermalize")
    parser.add_argument('-Nt', '--tsteps', default=10000000, type=int,
                        help="number of training")
    parser.add_argument('-o',  '--optimizer', choices=['adam', 'sgd', 'yogi'], default='adam',
                        help='optimizer to use')
    parser.add_argument('-s', '--schedule', action='store_true',
                        help="scheduled learning rate")
    parser.add_argument('-C', '--care', type=float, default=1,
                        help='scaling for learning schedule')
    parser.add_argument('--b1', type=float, default=0.9,
                        help="b1 parameter for adam")
    parser.add_argument('--b2', type=float, default=0.999,
                        help="b2 parameter for adam")
    parser.add_argument('--weight', type=str, default='jnp.ones(len(grads))',
                        help="weight for gradients")

    args = parser.parse_args()

    seed = args.seed
    if args.seed_time:
        seed = time.time_ns()
    torch.manual_seed(seed)

    with open(args.model, 'rb') as f:
        model = eval(f.read())
    V = model.dof

    skip = args.skip
    if args.skip == 30:
        skip = V

    '''
    if args.nnaffine:
        indftn = model.lattice.nearestneighbor
    '''

    if args.real:
        # Output real plane and quit
        contour = RealContour()
        contour.apply(nn.init.zeros_)
        with open(args.contour, 'wb') as f:
            pickle.dump(contour, f)
        sys.exit(0)

    loaded = False
    if not args.init and not args.fromfile:
        try:
            with open(args.contour, 'rb') as f:
                contour = pickle.load(f)
            loaded = True
        except FileNotFoundError:
            pass
    if args.fromfile:
        with open(args.fromfile, 'rb') as f:
            contour = pickle.load(f)
        loaded = True
    if not loaded:
        contour = Contour(V, [V]+[args.width*V]*args.layers)
        '''
        if model.periodic_contour:
            if args.affine:
                contour = PeriodicAffineContour(
                    V, [args.width*V] * args.layers, args.width, even_indices, odd_indices)
            else:
                contour = PeriodicContour(
                    V, [args.width*V] * args.layers, args.width)
        else:
            if args.affine:
                contour = AffineContour(
                    V, [args.width*V] * args.layers, even_indices, odd_indices)
            elif args.nnaffine:
                contour = NearestNeighborAffineContour(
                    V, [args.width*4] * args.layers, even_indices, odd_indices, indftn)
            else:
                contour = Contour(V, [args.width*V]*args.layers)
        '''
    
    '''
    if args.affine or args.nnaffine:
        even_indices = model.lattice.even()
        odd_indices = model.lattice.odd()

        @jax.jit
        def Seff(x, p):
            j = jax.jacfwd(lambda y: contour.apply(p, y))(x)
            logdet = jnp.log(j.diagonal().prod())
            xt = contour.apply(p, x)
            Seff = model.action(xt) - logdet
            return Seff
    '''

    def Seff(x):
        j = torch.autograd.functional.jacobian(contour, x)  # need to check
        s, logdet = torch.linalg.slogdet(j)
        xt = contour(x)  # need to check
        Seff = model.action(xt) - torch.log(s) - logdet
        return Seff

    # setup metropolis
    chain = metropolis.Chain(lambda x: Seff(
        x, contour).real, torch.zeros(V),  delta=1./torch.sqrt(torch.IntTensor([V])))
    '''
    if args.replica:
        chain = replica.ReplicaExchange(lambda x: Seff(x, contour_params), jnp.zeros(
            V), chain_key, delta=1./jnp.sqrt(V), max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
    else:
        chain = metropolis.Chain(lambda x: Seff(
            x, contour_params).real, jnp.zeros(V), chain_key, delta=1./jnp.sqrt(V))
    '''
    # cost function
    # Seff_grad = torch.autograd.grad(lambda y, p: -Seff(y, p).real, argnums=1)

    '''
    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care*1000),
            decay_rate=0.99,
            end_value=2e-5)
    else:
        sched = optax.constant_schedule(args.learningrate)
    
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(contour_params)
    opt_update_jit = jax.jit(opt.update)
    '''

    opt = optim.Adam(contour.parameters(), lr=args.learningrate)

    def save():
        with open(args.contour, 'wb') as f:
            pickle.dump(contour, f)

    '''
    def Grad_Mean(grads, weight):
        """
        Params:
            grads: Gradients
            weight: Weights
        """
        grads_w = [jax.tree_util.tree_map(
            lambda x: w*x, g) for w, g in zip(weight, grads)]
        w_mean = jnp.mean(weight)
        grad_mean = jax.tree_util.tree_map(
            lambda *x: jnp.mean(jnp.array(x), axis=0)/w_mean, *grads_w)
        return grad_mean
    '''

    def bootstrap(xs, ws=None, N=100, Bs=50):
        if Bs > len(xs):
            Bs = len(xs)
        B = len(xs)//Bs
        if ws is None:
            ws = xs*0 + 1
        # Block
        x, w = [], []
        for i in range(Bs):
            x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
            w.append(sum(ws[i*B:i*B+B]))
        x = np.array(x)
        w = np.array(w)
        # Regular bootstrap
        y = x * w
        m = (sum(y) / sum(w))
        ms = []
        for _ in range(N):
            s = np.random.choice(range(len(x)), len(x))
            ms.append((sum(y[s]) / sum(w[s])))
        ms = np.array(ms)
        return m, np.std(ms.real) + 1j*np.std(ms.imag)

    steps = int(10000 / args.nstochastic)
    weight = eval(args.weight)

    chain.calibrate()
    chain.step(N=args.thermalize*V)
    try:
        for t in range(args.tsteps):
            for s in range(steps):
                opt.zero_grad()
                chain.calibrate()
                grads = []
                for l in range(args.nstochastic):
                    chain.step(N=skip)
                    grads.append(Seff(chain.x).backward())

                grad = torch.mean(torch.stack(grads), axis=0)
                grad.backward()
                opt.step()
            '''
            # tracking the size of gradient
            grad_abs = 0.
            for i in range(2):
                grad_abs += np.linalg.norm(grad['params']
                                           ['Dense_'+str(i)]['kernel'])
                grad_abs += np.linalg.norm(grad['params']
                                           ['Dense_'+str(i)]['bias'])

                for j in range(args.layers):
                    grad_abs += np.linalg.norm(grad['params']
                                               ['MLP_'+str(i)]['Dense_'+str(j)]['kernel'])
                    grad_abs += np.linalg.norm(grad['params']
                                               ['MLP_'+str(i)]['Dense_'+str(j)]['bias'])
            '''
            # measurement once in a while
            phases = []
            acts = []
            for i in range(len(phases)):
                chain.step(N=skip)
                acts.append(Seff(chain.x, contour))
                phases.append(torch.exp(-1j*acts[i].imag))

            print(f'{np.mean(phases).real} {np.abs(np.mean(phases))} {bootstrap(np.array(phases))} ({np.mean(np.abs(chain.x))} {np.real(np.mean(acts))} {np.mean(acts)} {chain.acceptance_rate()})', flush=True)

            save()

    except KeyboardInterrupt:
        print()
        save()
