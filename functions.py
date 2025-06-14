import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools


def hamiltonian_generator_1_e(N, v, w):
    tot_sites = 2 * N

    H = np.zeros((tot_sites, tot_sites))

    for k in range(N):
        H[2*k-1,2*k] = w
        H[2*k,2*k-1] = w
        H[2*k,2*k+1] = v
        H[2*k+1,2*k] = v
    
    H[-1,0] = 0
    H[0,-1] = 0

    return H


def basis_generator_mul_e(tot_sites, N_e): # Checked OK

  basis = []

  for combination_indices in itertools.combinations(range(tot_sites), N_e):
    state = [0] * tot_sites
    for index in combination_indices:
      state[index] = 1
    basis.append(state)
  
  return np.array(basis, dtype=np.float64)


def hamiltonian_generator_mul_e(basis_set, tot_sites, v, w):
  d = len(basis_set)
  H = np.zeros((d, d))

  for i in range(d):
    for j in range(d):
      H[j][i] = np.dot(hamiltonian_on_ket(basis_set[i], tot_sites, v, w), basis_set[j])

  return H


def hamiltonian_on_ket(ket, tot_sites, v, w):
  res = np.zeros((tot_sites), dtype=np.float64)
  for i in range(tot_sites):
    if i % 2 == 0 and i - 1 > 0:
      if ket[i - 1] == 1:
        res[i] += w
      if ket[i + 1] == 1:
        res[i] += v
    elif i % 2 == 1 and i + 1 < tot_sites:
      if ket[i - 1] == 1:
        res[i] += v
      if ket[i + 1] == 1:
        res[i] += w
    elif i == 0 and ket[i + 1] == 1:
      res[i] += v
    elif i == tot_sites - 1 and ket[i - 1] == 1:
      res[i] += v
  return res


def normalize(vec):
  norm = np.linalg.norm(vec)
  if norm == 0:
    return vec
  return vec / norm
