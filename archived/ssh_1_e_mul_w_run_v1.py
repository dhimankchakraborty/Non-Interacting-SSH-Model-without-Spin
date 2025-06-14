'''
This code if for only a single value of "v" and multiple values of "w"
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


w_step_no = 101
w_0 = 0
w_end = 3
w_arr = np.linspace(w_0, w_end, w_step_no)
w_arr = np.around(w_arr, decimals=3)
v = 1
N = 10
tot_sites = 2 * N
topo_state_no = 2


topo_state_arr = []
all_e_val_arr = []

for w in w_arr:
    topo_state_arr_idx_arr = []
    topo_state_arr.append([])
    hamiltonian = hamiltonian_generator_1_e(N, v, w)

    dim = len(hamiltonian)

    e_val_arr, e_vec_arr = np.linalg.eigh(hamiltonian)

    all_e_val_arr.append(e_val_arr)

    e_val_arr_abs = np.abs(e_val_arr)

    e_val_arr_abs_sorted = np.sort(e_val_arr_abs)

    for i in range(topo_state_no):
        topo_state_arr_idx_arr.append((e_val_arr_abs == e_val_arr_abs_sorted[i]).argmax())

    for i in topo_state_arr_idx_arr:
        topo_state_arr[-1].append(e_vec_arr[:, i])

all_e_val_arr = np.array(all_e_val_arr)
topo_state_arr = np.array(topo_state_arr)

for e_w_arr in all_e_val_arr.transpose():
    plt.plot(w_arr, e_w_arr)
plt.grid()
plt.show()

sites_arr = np.arange(tot_sites)

for j, state_0_en in enumerate(topo_state_arr):
    for i, wf in enumerate(state_0_en):
        plt.plot(sites_arr, wf, label=f'{topo_state_arr_idx_arr[i]}')
        plt.title(f"w = {w_arr[j]}")
    plt.legend()
    plt.grid()
    plt.show()


data_file_name = "ssh-1-e-mul-w-run.h5"
data_file_output_directory = "data"
group_name = f"{N}-{w_0}-{w_end}-{w_step_no}-{w}-{topo_state_no}-{datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")}"

os.makedirs(data_file_output_directory, exist_ok=True)
data_file_full_path = os.path.join(data_file_output_directory, data_file_name)

file = h5py.File(data_file_full_path, 'a')

group = file.create_group(group_name)

group.attrs['Number of Unit Cells'] = N
group.attrs['Total number of Sites'] = tot_sites
group.attrs['Start of w'] = w_0
group.attrs['End of w'] = w_end
group.attrs['Number of w steps'] = w_step_no
group.attrs['v'] = v
group.attrs['No of Recorded Topological States'] = topo_state_no
group.attrs['Dimension of Hilbert Space'] = dim
group.attrs['Date'] = datetime.datetime.now().strftime("%d-%m-%y")
group.attrs['Time'] = datetime.datetime.now().strftime("%H:%M:%S")

v_arr_dset = group.create_dataset("v_arr", data=w_arr)
v_arr_dset.attrs['Shape'] = np.shape(w_arr)
v_arr_dset.attrs['Style'] = "I1"
v_arr_dset.attrs['I1'] = "Denotes the value of each w"

e_val_dset = group.create_dataset("e_val_arr", data=all_e_val_arr)
e_val_dset.attrs['Shape'] = np.shape(all_e_val_arr)
e_val_dset.attrs['Style'] = "I1xI2"
e_val_dset.attrs['I1'] = "Denotes the set of eigen values of each w" 
e_val_dset.attrs['I2'] = "Denotes eigen value of each eigenstate" 

topo_state_dset = group.create_dataset("topo_state_arr", data=np.array(topo_state_arr))
topo_state_dset.attrs['Shape'] = np.shape(topo_state_arr)
topo_state_dset.attrs['Style'] = "I1xI2xI3" 
topo_state_dset.attrs['I1'] = "Denotes the set of eigen states of each w" 
topo_state_dset.attrs['I2'] = "Denotes each zero energy eigenstate" 
topo_state_dset.attrs['I3'] = "Denotes each element of a zero energy eigenstate" 

file.close()