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

hamiltonian = hamiltonian_generator_1_e(N, v, w)

print(hamiltonian)

e_val_arr_np, e_vec_arr_np = np.linalg.eigh(hamiltonian)
e_val_arr_sp, e_vec_arr_sp = sp.linalg.eigh(hamiltonian)

# print(e_val_arr_np)
# print(e_vec_arr_np)
# print(e_val_arr_sp)
# print(e_vec_arr_sp)


for i in range(len(e_val_arr_np)):
    print(f"{i} : {e_val_arr_np[i]} --------- {e_val_arr_sp[i]}")


# for i in range(len(e_vec_arr_np)):
#     for j in range(len(e_vec_arr_np[0])):
#         print(f"{e_vec_arr_np[i][j]} --------- {e_vec_arr_sp[i][j]}")

sites_arr = np.arange(tot_sites)
print(sites_arr)



# for i, wf in enumerate(e_vec_arr_np):
#     plt.plot(sites_arr, wf, label=f"{i}")
#     plt.legend()
#     plt.show()

# for i, wf in enumerate(e_vec_arr_sp):
#     plt.plot(sites_arr, wf, label=f"{i}")
#     plt.legend()
#     plt.show()

