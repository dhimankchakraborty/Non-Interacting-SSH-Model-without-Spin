'''
This code if for only a single value of "v" and "w"
N : number of unit cells
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as itr
import h5py
import os
import datetime
from functions import *


N = 10
w = 2
v = 1
tot_sites = 2 * N
# tol = 10**(-2)

topo_state_arr_idx_arr = []
topo_state_arr = []
hamiltonian = hamiltonian_generator_1_e(N, v, w)

dim = len(hamiltonian)

e_val_arr, e_vec_arr = np.linalg.eigh(hamiltonian)

e_val_arr_abs = np.abs(e_val_arr)

e_val_arr_abs_sorted = np.sort(e_val_arr_abs)
# print(e_val_arr_abs_sorted)

# for i in range(dim):
#     if e_val_arr_abs[i] < tol:
#         topo_state_arr_idx_arr.append(i)

# print((e_val_arr == e_val_arr_abs_sorted[1]).argmax())


topo_state_arr_idx_arr.append((e_val_arr_abs == e_val_arr_abs_sorted[0]).argmax())
topo_state_arr_idx_arr.append((e_val_arr_abs == e_val_arr_abs_sorted[1]).argmax())

print(topo_state_arr_idx_arr)

sites_arr = np.arange(tot_sites)

for i in topo_state_arr_idx_arr:
    topo_state_arr.append(e_vec_arr[:, i])


for i, wf in enumerate(topo_state_arr):
    plt.plot(sites_arr, wf, label=f'{topo_state_arr_idx_arr[i]}')
plt.legend()
plt.grid()
plt.show()


# data_file_name = "ssh-1-e-1-run.h5"
# data_file_output_directory = "data"
# group_name = f"{N}-{v}-{w}-{tol}-{datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")}"

# os.makedirs(data_file_output_directory, exist_ok=True)
# data_file_full_path = os.path.join(data_file_output_directory, data_file_name)

# file = h5py.File(data_file_full_path, 'a')

# group = file.create_group(group_name)

# group.attrs['Number of Unit Cells'] = N
# group.attrs['Total number of Sites'] = tot_sites
# group.attrs['v'] = v
# group.attrs['w'] = w
# group.attrs['Tolerence for Zero Energy State'] = tol
# group.attrs['Dimension of Hilbert Space'] = dim
# group.attrs['Date'] = datetime.datetime.now().strftime("%d-%m-%y")
# group.attrs['Time'] = datetime.datetime.now().strftime("%H:%M:%S")

# e_val_dset = group.create_dataset("e_val_arr", data=e_val_arr)
# e_val_dset.attrs['Shape'] = np.shape(e_val_arr)
# e_val_dset.attrs['Style'] = "I1" 
# e_val_dset.attrs['I1'] = "Denotes each eigenstate" 

# topo_state_dset = group.create_dataset("topo_state_arr", data=np.array(topo_state_arr))
# topo_state_dset.attrs['Shape'] = np.shape(topo_state_arr)
# topo_state_dset.attrs['Style'] = "I1xI2" 
# topo_state_dset.attrs['I1'] = "Denotes each zero energy eigenstate" 
# topo_state_dset.attrs['I2'] = "Denotes each element of zero energy eigenstate" 

# file.close()

