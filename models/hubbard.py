from dataclasses import dataclass
from typing import Tuple
from itertools import product

import scipy.special as special
from scipy.optimize import fsolve

import torch
import numpy as np
# import jax
# import jax.numpy as jnp


@dataclass  # 2D model, LxL lattice
class Lattice:
    #    geom: Tuple[int]
    L: int
    nt: int

    def __post_init__(self):
        #       self.D = len(self.geom)
        self.V = self.L**2
        self.dof = self.V * self.nt

    def idx(self, t, x1, x2):
        return (x2 % self.L) + self.L * (x1 % self.L) + self.L * self.L * (t % self.nt)

    def idx1(self, x1, x2):
        return (x2 % self.L) + self.L * (x1 % self.L)

    def idxreverse(self, num):
        t = num // (self.L**2)
        x1 = (num - t * (self.L ** 2)) // self.L
        x2 = (num - t * (self.L ** 2) - x1 * self.L) % self.L

        return torch.IntTensor([t, x1, x2])

    def sites(self):
        # Return a list of all sites.
        # need to find torch version
        return np.indices((self.nt, self.L, self.L))

    def sites1(self):
        # for exp discretiizaiton
        # need to find torch version
        return np.indices((self.nt, self.L, self.L, self.L, self.L))

    def spatial_sites(self):
        # Return a list of spatial sites
        return np.indices((self.L, self.L))  # need to find torch version

    def coords(self, i):
        t = i//self.L
        x = i % self.L
        return t, x

    # For Affine coupling layer
    '''
    def even(self):
        e_even = torch.zeros([self.nt, self.L//2, self.L//2])
        e_odd = torch.zeros([self.nt, self.L//2, self.L//2])

        def even_update_at(K, t, i, j):
            K = K.at[t, i, j].set(t * self.V + 2*(self.L * i + j))
            return K

        def odd_update_at(K, t, i, j):
            K = K.at[t, i, j].set(
                t * self.V + 2*(self.L * i + j + self.L//2) + 1)
            return K

        # need to find torch version
        ts, xs, ys = np.indices((self.nt, self.L//2, self.L//2))
        ts = torch.ravel(ts)
        xs = torch.ravel(xs)
        ys = torch.ravel(ys)

        def even_update_at_i(i, K):
            return even_update_at(K, ts[i], xs[i], ys[i])

        def odd_update_at_i(i, K):
            return odd_update_at(K, ts[i], xs[i], ys[i])

        e_even = jax.lax.fori_loop(0, len(ts), even_update_at_i, e_even)
        e_odd = jax.lax.fori_loop(0, len(ts), odd_update_at_i, e_odd)

        e = jnp.concatenate((e_even, e_odd), axis=None)

        return jnp.array(e, int).sort()

    def odd(self):
        o_even = jnp.zeros([self.nt, self.L//2, self.L//2])
        o_odd = jnp.zeros([self.nt, self.L//2, self.L//2])

        def even_update_at(K, t, i, j):
            K = K.at[t, i, j].set(t * self.V + 2*(self.L * i + j) + 1)
            return K

        def odd_update_at(K, t, i, j):
            K = K.at[t, i, j].set(t * self.V + 2*(self.L * i + j + self.L//2))
            return K

        ts, xs, ys = jnp.indices((self.nt, self.L//2, self.L//2))
        ts = jnp.ravel(ts)
        xs = jnp.ravel(xs)
        ys = jnp.ravel(ys)

        def even_update_at_i(i, K):
            return even_update_at(K, ts[i], xs[i], ys[i])

        def odd_update_at_i(i, K):
            return odd_update_at(K, ts[i], xs[i], ys[i])

        o_even = jax.lax.fori_loop(0, len(ts), even_update_at_i, o_even)
        o_odd = jax.lax.fori_loop(0, len(ts), odd_update_at_i, o_odd)

        o = jnp.concatenate((o_even, o_odd), axis=None)

        return jnp.array(o, int).sort()

    def nearestneighbor(self, num):
        index = self.idxreverse(num)

        x1 = (index+jnp.array([0, 0, 1])
              ) % jnp.array([self.nt, self.L, self.L])
        x2 = (index+jnp.array([0, 1, 0])
              ) % jnp.array([self.nt, self.L, self.L])
        x3 = (index+jnp.array([0, 0, -1])
              ) % jnp.array([self.nt, self.L, self.L])
        x4 = (index+jnp.array([0, -1, 0])
              ) % jnp.array([self.nt, self.L, self.L])

        def _idx(arr):
            return (arr[2] % self.L) + self.L * (arr[1] % self.L) + self.L * self.L * (arr[0] % self.nt)

        return jnp.array([_idx(x1), _idx(x2), _idx(x3), _idx(x4)], int)
    '''


@dataclass
class Hopping:
    lattice: Lattice
    kappa: float
    mu: float

    def hopping(self):
        size = self.lattice.L
        hop = torch.zeros((size**2, size**2))
        idx = Lattice(self.lattice.L, 1).idx1

        for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
            if (x1 == x2 and (y1 == (y2 + 1) % size or y1 == (y2 - 1 + size) % size)):
                hop[idx(x1, y1), idx(x2, y2)] = 1.0
            if (y1 == y2 and (x1 == (x2 + 1) % size or x1 == (x2 - 1 + size) % size)):
                hop[idx(x1, y1), idx(x2, y2)] = 1.0

        return hop

    def hopping2(self):
        size = self.lattice.L
        hop2 = torch.zeros((size**2, size**2))
        idx = Lattice(self.lattice.L, 1).idx1

        for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
            if (x1 == (x2 + 1) % size and (y1 == (y2 + 1) % size or y1 == (y2 - 1 + size) % size)):
                hop2[idx(x1, y1), idx(x2, y2)] = 1.0
            if (x1 == (x2 - 1 + size) % size and (y1 == (y2 + 1) % size or y1 == (y2 - 1 + size) % size)):
                hop2[idx(x1, y1), idx(x2, y2)] = 1.0

        return hop2

    def exp_h1(self):
        h1 = self.kappa * self.hopping() + self.mu * torch.eye(self.lattice.V) + \
            0.0 * self.hopping2()
        h1 = torch.matrix_exp(h1)
        return h1+0j

    def exp_h2(self):
        h2 = self.kappa * self.hopping()
        h2 = self.kappa * self.hopping() - self.mu * torch.eye(self.lattice.V) + \
            0.0 * self.hopping2()
        h2 = torch.matrix_exp(h2)
        return h2+0j

    def exp_h(self):
        h = self.kappa * self.hopping() + 0.0 * self.hopping2()
        h = torch.matrix_exp(h)
        return h+0j


@dataclass
class ImprovedModel:
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()

        self.h1_svd = torch.svd(self.h1)
        self.h2_svd = torch.svd(self.h2)

        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 - 1)) /
                           special.iv(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        fer_mat1 = torch.eye(self.lattice.V, dtype=torch.cfloat)

        for t in range(self.nt):
            temp_mat = torch.zeros(
                (self.lattice.V, self.lattice.V), dtype=torch.cfloat)

            for x in range(self.lattice.V):
                temp_mat[x, x] = torch.exp(
                    1j * torch.sin(A[t * self.lattice.V + x]))

            fer_mat1 = self.h1 @ temp_mat @ fer_mat1

        return torch.eye(self.lattice.V, dtype=torch.cfloat) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = torch.eye(self.lattice.V, dtype=torch.cfloat)

        for t in range(self.nt):
            temp_mat = torch.zeros(
                (self.lattice.V, self.lattice.V), dtype=torch.cfloat)

            for x in range(self.lattice.V):
                temp_mat[x, x] = torch.exp(
                    -1j * torch.sin(A[t * self.lattice.V + x]))

            fer_mat2 = self.h2 @ temp_mat @ fer_mat2

        return torch.eye(self.lattice.V, dtype=torch.cfloat) + fer_mat2

    def svd_mult(self, A, B):  # fact_mult
        m = torch.svd((torch.diag(A[1])+0j) @ (A[2].mH)
                      @ B[0] @ (torch.diag(B[1])+0j))
        u = A[0] @ m[0]
        s = m[1]
        v = B[2] @ m[2]
        return u, s, v

    def Hubbard1_svd(self, A):
        fer_mat1 = (torch.eye(self.lattice.V, dtype=torch.cfloat),
                    torch.ones(self.lattice.V), torch.eye(self.lattice.V, dtype=torch.cfloat))

        def update_at_t(t, A):
            fer_mat = torch.eye(self.lattice.V, dtype=torch.cfloat)
            for x in range(self.lattice.V):
                fer_mat[x, x] = torch.exp(
                    1j * torch.sin(A[t * self.lattice.V + x]))
            return torch.svd(fer_mat)

        for t in range(self.nt):
            fer_mat1 = self.svd_mult(update_at_t(t, A), fer_mat1)
            fer_mat1 = self.svd_mult(self.h1_svd, fer_mat1)

        # check the paper and write it compatibly with pytorch
        final_svd = torch.svd(
            fer_mat1[0].mH @ fer_mat1[2] + torch.diag(fer_mat1[1]))
        final_u = fer_mat1[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = fer_mat1[2] @ final_svd[2]

        return final_u, final_d, final_v

    def Hubbard2_svd(self, A):
        fer_mat2 = (torch.eye(self.lattice.V, dtype=torch.cfloat),
                    torch.ones(self.lattice.V), torch.eye(self.lattice.V, dtype=torch.cfloat))

        def update_at_t(t, A):
            fer_mat = torch.eye(self.lattice.V, dtype=torch.cfloat)
            for x in range(self.lattice.V):
                fer_mat[x, x] = torch.exp(
                    -1j * torch.sin(A[t * self.lattice.V + x]))
            return torch.svd(fer_mat)

        for t in range(self.nt):
            fer_mat2 = self.svd_mult(update_at_t(t, A), fer_mat2)
            fer_mat2 = self.svd_mult(self.h2_svd, fer_mat2)

        final_svd = torch.svd(
            fer_mat2[0].mH @ fer_mat2[2] + torch.diag(fer_mat2[1]))
        final_u = fer_mat2[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = fer_mat2[2] @ final_svd[2]

        return final_u, final_d, final_v

    def action_naive(self, A):
        s1, logdet1 = torch.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = torch.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * torch.sum(torch.cos(A)) - torch.log(s1) - logdet1 - torch.log(s2) - logdet2

    def action_svd(self, A):
        u1, d1, v1 = self.Hubbard1_svd(A)
        u1_s, u1_logdet = torch.linalg.slogdet(u1)
        d1_logdet = torch.sum(torch.log(d1))
        v1_s, v1_logdet = torch.linalg.slogdet(v1.mH)
        logdet1 = torch.log(u1_s) + u1_logdet + d1_logdet + \
            torch.log(v1_s) + v1_logdet

        u2, d2, v2 = self.Hubbard2_svd(A)
        u2_s, u2_logdet = torch.linalg.slogdet(u2)
        d2_logdet = torch.sum(torch.log(d2))
        v2_s, v2_logdet = torch.linalg.slogdet(v2.mH)
        logdet2 = torch.log(u2_s) + u2_logdet + d2_logdet + \
            torch.log(v2_s) + v2_logdet

        return -self.beta * torch.sum(torch.cos(A)) - logdet1 - logdet2

    def action(self, A):
        return self.action_naive(A)

    # def density(self, A):
    #     fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
    #     fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

    #     return jnp.trace(fer_mat2_inv - fer_mat1_inv) / self.lattice.V

    # def doubleoccupancy(self, A):
    #     d = 0 + 0j
    #     fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
    #     fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

    #     def do(i, d):
    #         return d + fer_mat2_inv[i, i] * (1.0 - fer_mat1_inv[i, i])

    #     d = jax.lax.fori_loop(0, self.lattice.V, do, d) / self.lattice.V

    #     return d

    # def staggered_magnetization(self, A):
    #     m = 0 + 0j
    #     fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
    #     fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

    #     idx = self.lattice.idx1
    #     size = self.lattice.L
    #     x = jnp.array(range(size))

    #     def update_at(x1, y1, x2, y2):
    #         a = idx(x1, y1)
    #         b = idx(x2, y2)

    #         m = - (-1.0)**(x1 + y1 + x2 + y2) * fer_mat1_inv[a, b] * \
    #             fer_mat1_inv[b, a] + fer_mat2_inv[a, b] * fer_mat2_inv[b, a]

    #         return m

    #     def update_at_diagonal(x1, y1):
    #         a = idx(x1, y1)
    #         return fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

    #     m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
    #                                                                         lambda j, vy: vy + jax.lax.fori_loop(0, size,
    #                                                                                                              lambda k, vz: vz + jax.lax.fori_loop(0, size, lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), m), m), m), m)
    #     m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(
    #         0, size, lambda j, vy: vy + update_at_diagonal(x[i], x[j]), m), m)

    #     ''' for representation
    #     for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
    #         a = idx(x1, y1)
    #         b = idx(x2, y2)

    #         m -= (-1.0)**(x1 + y1 + x2 + y2) * fer_mat1_inv(a, b) * \
    #         fer_mat1_inv(b, a) + fer_mat2_inv(a, b) * fer_mat2_inv(b, a)

    #         if (x1 == x2 and y1 == y2):
    #             m += fer_mat1_inv(a, a) * fer_mat2_inv(a, a)
    #     '''
    #     return m

    # def magnetization(self, A):
    #     m = 0 + 0j
    #     fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
    #     fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

    #     idx = self.lattice.idx1
    #     size = self.lattice.L
    #     x = jnp.array(range(size))

    #     def update_at(x1, y1, x2, y2):
    #         a = idx(x1, y1)
    #         b = idx(x2, y2)

    #         m = - fer_mat1_inv[a, b] * fer_mat1_inv[b, a] + \
    #             fer_mat2_inv[a, b] * fer_mat2_inv[b, a]

    #         return m

    #     def update_at_diagonal(x1, y1):
    #         a = idx(x1, y1)
    #         return fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

    #     m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
    #                                                                         lambda j, vy: vy + jax.lax.fori_loop(0, size,
    #                                                                                                              lambda k, vz: vz + jax.lax.fori_loop(0, size, lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), m), m), m), m)

    #     m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(
    #         0, size, lambda j, vy: vy + update_at_diagonal(x[i], x[j]), m), m)

    #     ''' for representation
    #     for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
    #         a = idx(x1, y1)
    #         b = idx(x2, y2)

    #         m -= fer_mat1_inv(a, b) * fer_mat1_inv(b, a) + \
    #         fer_mat2_inv(a, b) * fer_mat2_inv(b, a)

    #         if (x1 == x2 and y1 == y2):
    #             m += fer_mat1_inv(a, a) * fer_mat2_inv(a, a)
    #     '''
    #     return m

    # def n1(self, A):
    #     fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))

    #     return jnp.trace(fer_mat1_inv) / self.lattice.V

    # def n2(self, A):
    #     fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

    #     return jnp.trace(fer_mat2_inv) / self.lattice.V

    # def hamiltonian(self, A):
    #     h = 0 + 0j
    #     fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
    #     fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

    #     idx = self.lattice.idx1
    #     size = self.lattice.L
    #     x = jnp.array(range(size))

    #     def update_at(x1, y1, x2, y2):
    #         a = idx(x1, y1)
    #         b = idx(x2, y2)

    #         h = 0.5 * self.kappa * \
    #             self.hopping[a, b] * (fer_mat1_inv[a, b] + fer_mat2_inv[a, b])

    #         return h

    #     def update_at_diagonal(x1, y1):
    #         a = idx(x1, y1)
    #         return fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

    #     h = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
    #                                                                         lambda j, vy: vy + jax.lax.fori_loop(0, size,
    #                                                                                                              lambda k, vz: vz + jax.lax.fori_loop(0, size, lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), h), h), h), h)

    #     h = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(
    #         0, size, lambda j, vy: vy + update_at_diagonal(x[i], x[j]), h), h)

    #     return h

    # def observe(self, A):
    #     return jnp.array([self.density(A), self.doubleoccupancy(A), self.action(A), self.staggered_magnetization(A)])

    # def all(self, A):
    #     """
    #     Returns:
    #         Action and gradient
    #     """
    #     act, dact = jax.value_and_grad(self.action, holomorphic=True)(A+0j)
    #     return act, dact


'''
@dataclass
class ImprovedModel2(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h = self.Hopping.exp_h()

        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 - 1)) /
                           special.iv(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * jnp.sin(A[t * self.lattice.V + x]) + self.mu))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * jnp.sin(A[t * self.lattice.V + x]) - self.mu))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ConventionalModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(1, x + 0j) / (x *
                                                    special.iv(0, x + 0j))) - self.u
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1_old(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j
        H = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j

        for t in range(self.lattice.nt):
            for x1 in range(self.lattice.L):
                for x2 in range(self.lattice.L):
                    H = H.at[idx(x1, x2), idx(x1, x2)].set(-1.0 + self.u /
                                                           2.0 - self.mu - 1j * jnp.sin(A[t, x1, x2]))
                    H = H.at[idx(x1, x2), idx(x1 + 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 + 1)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1 - 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 - 1)].set(-self.kappa)

            fer_mat1 = H @ fer_mat1

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_old(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j
        H = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j

        for t in range(self.lattice.nt):
            for x1 in range(self.lattice.L):
                for x2 in range(self.lattice.L):
                    H = H.at[idx(x1, x2), idx(x1, x2)].set(-1.0 + self.u /
                                                           2.0 + self.mu + 1j * jnp.sin(A[t, x1, x2]))
                    H = H.at[idx(x1, x2), idx(x1 + 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 + 1)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1 - 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 - 1)].set(-self.kappa)

            fer_mat2 = H @ fer_mat2

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1_new(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 - self.mu - 1j * jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_new(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 + self.mu + 1j * jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1(self, A):
        return self.Hubbard1_new(A)

    def Hubbard2(self, A):
        return self.Hubbard2_new(A)

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2
'''


@dataclass
class ImprovedGaussianModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()

        self.h1_svd = torch.svd(self.h1+0j)
        self.h2_svd = torch.svd(self.h2+0j)

        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = torch.eye(self.lattice.V, dtype=torch.cfloat)

        for t in range(self.nt):
            temp_mat = torch.zeros(
                (self.lattice.V, self.lattice.V), dtype=torch.cfloat)

            for x in range(self.lattice.V):
                temp_mat[x, x] = torch.exp(1j * A[t * self.lattice.V + x])

            fer_mat1 = self.h1 @ temp_mat @ fer_mat1

        return torch.eye(self.lattice.V, dtype=torch.cfloat) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = torch.eye(self.lattice.V, dtype=torch.cfloat)

        for t in range(self.nt):
            temp_mat = torch.zeros(
                (self.lattice.V, self.lattice.V), dtype=torch.cfloat)

            for x in range(self.lattice.V):
                temp_mat[x, x] = torch.exp(-1j * A[t * self.lattice.V + x])

            fer_mat2 = self.h2 @ temp_mat @ fer_mat2

        return torch.eye(self.lattice.V, dtype=torch.cfloat) + fer_mat2

    def Hubbard1_svd(self, A):
        fer_mat1 = (torch.eye(self.lattice.V, dtype=torch.cfloat),
                    torch.ones(self.lattice.V, dtype=torch.cfloat), torch.eye(self.lattice.V, dtype=torch.cfloat))

        def update_at_t(t, A):
            fer_mat = torch.eye(self.lattice.V, dtype=torch.cfloat)
            for x in range(self.lattice.V):
                fer_mat[x, x] = torch.exp(1j * A[t * self.lattice.V + x])
            return torch.svd(fer_mat)

        for t in range(self.nt):
            fer_mat1 = self.svd_mult(update_at_t(t, A), fer_mat1)
            fer_mat1 = self.svd_mult(self.h1_svd, fer_mat1)

        final_svd = torch.svd(
            fer_mat1[0].mH @ fer_mat1[2] + torch.diag(fer_mat1[1]))
        final_u = fer_mat1[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = fer_mat1[2] @ final_svd[2]

        return final_u, final_d, final_v

    def Hubbard2_svd(self, A):
        fer_mat2 = (torch.eye(self.lattice.V, dtype=torch.cfloat),
                    torch.ones(self.lattice.V, dtype=torch.cfloat), torch.eye(self.lattice.V, dtype=torch.cfloat))

        def update_at_t(t, A):
            fer_mat = torch.eye(self.lattice.V, dtype=torch.cfloat)
            for x in range(self.lattice.V):
                fer_mat[x, x] = torch.exp(-1j * A[t * self.lattice.V + x])
            return torch.svd(fer_mat)

        for t in range(self.nt):
            fer_mat2 = self.svd_mult(update_at_t(t, A), fer_mat2)
            fer_mat2 = self.svd_mult(self.h2_svd, fer_mat2)

        final_svd = torch.svd(
            fer_mat2[0].mH @ fer_mat2[2] + torch.diag(fer_mat2[1]))
        final_u = fer_mat2[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = fer_mat2[2] @ final_svd[2]

        return final_u, final_d, final_v

    def action_naive(self, A):
        s1, logdet1 = torch.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = torch.linalg.slogdet(self.Hubbard2(A))
        return 1.0 / (2 * self.u) * A @ A - torch.log(s1) - logdet1 - torch.log(s2) - logdet2

    def action_svd(self, A):
        u1, d1, v1 = self.Hubbard1_svd(A)
        u1_s, u1_logdet = torch.linalg.slogdet(u1)
        d1_logdet = torch.sum(torch.log(d1))
        v1_s, v1_logdet = torch.linalg.slogdet(v1.mH)
        logdet1 = torch.log(u1_s) + u1_logdet + d1_logdet + \
            torch.log(v1_s) + v1_logdet

        u2, d2, v2 = self.Hubbard2_svd(A)
        u2_s, u2_logdet = torch.linalg.slogdet(u2)
        d2_logdet = torch.sum(torch.log(d2))
        v2_s, v2_logdet = torch.linalg.slogdet(v2.mH)
        logdet2 = torch.log(u2_s) + u2_logdet + d2_logdet + \
            torch.log(v2_s) + v2_logdet

        return 1.0 / (2 * self.u) * A @ A - logdet1 - logdet2

    def action(self, A):
        return self.action_naive(A)


'''
@dataclass
class ImprovedGaussianModel2(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h = self.Hopping.exp_h()

        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * A[t * self.lattice.V + x] + self.mu))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * A[t * self.lattice.V + x] - self.mu))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ConventionalGaussianModel(ImprovedGaussianModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 - self.mu - 1j * A[t, x1, x2])
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 + self.mu + 1j * A[t, x1, x2])
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class DiagonalModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = False

        self.hopping_mat = jnp.eye(self.lattice.V) - self.kappa * self.hopping
        self.hopping_inverse = jnp.linalg.inv(self.hopping_mat)
        self.hopping_s, self.hopping_logdet = jnp.linalg.slogdet(
            self.hopping_mat)

    def Hubbard1(self, A):
        fer_mat1 = self.hopping_mat + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                -jnp.exp(1j * A[t * self.lattice.V + x]) - self.mu)
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return temp_mat @ self.hopping_inverse

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = self.hopping_mat + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                -jnp.exp(-1j * A[t * self.lattice.V + x]) + self.mu)
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return temp_mat @ self.hopping_inverse

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - 2 * self.lattice.nt * (self.hopping_s + self.hopping_logdet) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ImprovedGaussianAlphaModel(ImprovedGaussianModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float
    alpha: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.lattice.dof = 2 * self.lattice.dof
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j
        phi = A[:self.lattice.dof//2]
        chi = A[self.lattice.dof//2:]

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * phi[t * self.lattice.V + x] + chi[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j
        phi = A[:self.lattice.dof//2]
        chi = A[self.lattice.dof//2:]

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * phi[t * self.lattice.V + x] + chi[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A[:self.lattice.dof//2] ** 2) / (2 * self.alpha * self.u) + jnp.sum((A[self.lattice.dof//2:] + (1 - self.alpha) * self.u)**2)/(2 * (1 - self.alpha) * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class HyperbolicModel(ImprovedGaussianModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def BetaFunction(self):
        def fn(x):
            return np.real(special.spherical_jn(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * self.beta * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * self.beta * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(jnp.tanh(A)))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(jnp.tanh(A)))
        return 2 * jnp.sum(jnp.log(jnp.cosh(A))) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ImprovedSpinModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 + 1)) /
                           special.iv(0, x + 0j)) - np.exp(self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.sum(jnp.sin(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2
'''


@dataclass
class ImprovedGaussianSpinModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()

        self.h1 = self.kappa * self.hopping
        self.h2 = self.kappa * self.hopping
        for i in range(self.lattice.L**2):
            self.h1[i, i] = self.mu-self.u
            self.h2[i, i] = -self.mu-self.u
        self.h1 = torch.matrix_exp(self.h1)
        self.h2 = torch.matrix_exp(self.h2)

        self.h1_svd = torch.svd(self.h1)
        self.h2_svd = torch.svd(self.h2)

        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = torch.eye(self.lattice.V)

        for t in range(self.nt):
            temp_mat = torch.zeros(
                (self.lattice.V, self.lattice.V))

            for x in range(self.lattice.V):
                temp_mat[x, x] = torch.exp(A[t * self.lattice.V + x])

            fer_mat1 = self.h1 @ temp_mat @ fer_mat1

        return torch.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = torch.eye(self.lattice.V)

        for t in range(self.nt):
            temp_mat = torch.zeros(
                (self.lattice.V, self.lattice.V))

            for x in range(self.lattice.V):
                temp_mat[x, x] = torch.exp(A[t * self.lattice.V + x])

            fer_mat2 = self.h2 @ temp_mat @ fer_mat2

        return torch.eye(self.lattice.V) + fer_mat2

    def svd_mult(self, A, B):  # fact_mult
        m = torch.svd((torch.diag(A[1])) @ (A[2].T)
                      @ B[0] @ (torch.diag(B[1])))
        u = A[0] @ m[0]
        s = m[1]
        v = B[2] @ m[2]
        return u, s, v

    def Hubbard1_svd(self, A):
        fer_mat1 = (torch.eye(self.lattice.V),
                    torch.ones(self.lattice.V), torch.eye(self.lattice.V))

        def update_at_t(t, A):
            fer_mat = torch.eye(self.lattice.V)
            for x in range(self.lattice.V):
                fer_mat[x, x] = torch.exp(A[t * self.lattice.V + x])
            return torch.svd(fer_mat)

        for t in range(self.nt):
            fer_mat1 = self.svd_mult(update_at_t(t, A), fer_mat1)
            fer_mat1 = self.svd_mult(self.h1_svd, fer_mat1)

        final_svd = torch.svd(
            fer_mat1[0].T @ fer_mat1[2] + torch.diag(fer_mat1[1]))
        final_u = fer_mat1[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = fer_mat1[2] @ final_svd[2]

        return final_u, final_d, final_v

    def Hubbard2_svd(self, A):
        fer_mat2 = (torch.eye(self.lattice.V),
                    torch.ones(self.lattice.V), torch.eye(self.lattice.V))

        def update_at_t(t, A):
            fer_mat = torch.eye(self.lattice.V)
            for x in range(self.lattice.V):
                fer_mat[x, x] = torch.exp(A[t * self.lattice.V + x])
            return torch.svd(fer_mat)

        for t in range(self.nt):
            fer_mat2 = self.svd_mult(update_at_t(t, A), fer_mat2)
            fer_mat2 = self.svd_mult(self.h2_svd, fer_mat2)

        final_svd = torch.svd(
            fer_mat2[0].T @ fer_mat2[2] + torch.diag(fer_mat2[1]))
        final_u = fer_mat2[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = fer_mat2[2] @ final_svd[2]

        return final_u, final_d, final_v

    def action_naive(self, A):
        s1, logdet1 = torch.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = torch.linalg.slogdet(self.Hubbard2(A))
        return 1.0 / (2 * self.u) * A @ A - torch.log(s1) - logdet1 - torch.log(s2) - logdet2

    def action_svd(self, A):
        u1, d1, v1 = self.Hubbard1_svd(A)
        u1_s, u1_logdet = torch.linalg.slogdet(u1)
        d1_logdet = torch.sum(torch.log(d1))
        v1_s, v1_logdet = torch.linalg.slogdet(v1.T)
        logdet1 = torch.log(u1_s) + u1_logdet + d1_logdet + \
            torch.log(v1_s) + v1_logdet

        u2, d2, v2 = self.Hubbard2_svd(A)
        u2_s, u2_logdet = torch.linalg.slogdet(u2)
        d2_logdet = torch.sum(torch.log(d2))
        v2_s, v2_logdet = torch.linalg.slogdet(v2.T)
        logdet2 = torch.log(u2_s) + u2_logdet + d2_logdet + \
            torch.log(v2_s) + v2_logdet

        return 1.0 / (2 * self.u) * A @ A - logdet1 - logdet2

    def action(self, A):
        return self.action_naive(A)


'''
@dataclass
class ImprovedGaussianSpinModel2(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h = self.Hopping.exp_h()

        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(A[t * self.lattice.V + x] + self.mu))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(A[t * self.lattice.V + x]) - self.mu)
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ConventionalSpinModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(1, x + 0j) / (x *
                                                    special.iv(0, x + 0j))) - self.u
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 - self.mu + jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 + self.mu + jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2
'''
