'''
This code if for only a single value of "v" and "w"
N : number of unit cells
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as itr
from functions import *


N = 10
w = 2
v = 1
tot_sites = 2 * N
tol = 10**(-2)

topological_states_index = []
topological_states = []
hamiltonian = hamiltonian_generator_1_e(N, v, w)

dim = len(hamiltonian)

e_val_arr, e_vec_arr = np.linalg.eigh(hamiltonian)

e_val_arr_abs = np.abs(e_val_arr)

for i in range(dim):
    if e_val_arr_abs[i] < tol:
        topological_states_index.append(i)

# print(topological_states_index)

sites_arr = np.arange(tot_sites)

for i in topological_states_index:
    topological_states.append(e_vec_arr[:, i])


for i, wf in enumerate(topological_states):
    plt.plot(sites_arr, wf**2, label=f'{topological_states_index[i]}')
plt.legend()
plt.grid()
plt.show()

# # idx = e_val_arr_abs.argsort()
# # E_arr = e_val_arr_abs[idx]
# # E_vec_arr = e_vec_arr_T[:, idx]
# min_energy_idx = np.argmin(np.abs(e_val_arr))
# print(min_energy_idx)

# # print(E_arr)

# # topological_states.append(E_vec_arr[0])
# # topological_states.append(E_vec_arr[1])

# # for wf in topological_states:
# #     plt.plot(sites_arr, wf)
# #     plt.show()

# plt.plot(sites_arr, e_vec_arr[:, min_energy_idx])
# plt.show()


