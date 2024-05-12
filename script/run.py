# test.py
import cupy as cp
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('/software/pc24403/tfpcbpggsz/func')
sys.path.append('/software/pc24403/tfpcbpggsz/amp_ampgen')
from D0ToKSpipi2018 import *
import uproot as up
import pickle
import os
from multiprocessing import Pool

amp = PyD0ToKSpipi2018()
amp.init()

def get_mass(p1,p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)

def get_p4(decay="b2dpi", cut='', index=1):

    file_name = ''
    if cut == 'int':
        file_name = f'/software/pc24403/PCBPGGSZ/Int/weighted_{decay}.root:DalitzEventList'
    
    elif decay.split('_')[0] == 'b2dk' or decay.split('_')[0] == 'b2dpi':
        if cut == 'p':
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/lhcb_toy_{decay}_{index}_CPrange.root:BplusDalitzEventList'
        else:
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/oy/mass_fit/add_sw/lhcb_toy_{decay}_{index}.root:BminusDalitzEventList'

    tree = up.open(file_name)
  # Load the branches as arrays
    branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
    array = tree.arrays(branch_names)
       

    _p1 = cp.asarray([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
    _p2 = cp.asarray([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
    _p3 = cp.asarray([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
    # convert 4*1000 into a vectot<double>
    _p1 = cp.transpose(_p1)
    _p2 = cp.transpose(_p2)
    _p3 = cp.transpose(_p3)

    p1 = _p1
    p2 = _p2
    p3 = _p3

    p1bar = cp.hstack((_p1[:, :1], cp.negative(_p1[:, 1:])))
    p2bar = cp.hstack((_p2[:, :1], cp.negative(_p2[:, 1:])))
    p3bar = cp.hstack((_p3[:, :1], cp.negative(_p3[:, 1:])))


    return p1, p2, p3, p1bar, p2bar, p3bar


def load_int_amp(args):
    p1, p2, p3 = args

    return amp.AMP(p1.tolist(), p2.tolist(), p3.tolist())

def inital_amp():
    # Load the amplitude model

    print('Loading the amplitude model')
    

    Bdecays = ['b2dpi', 'b2dk']
    Types = ['DD', 'LL']
    Charges = ['p', 'm']
    for bdecay in Bdecays:
        for type in Types:
            for charge in Charges:
                start = time.time()

                decay = f'{bdecay}_{type}_{charge}'
                p1, p2, p3, p1bar, p2bar, p3bar = get_p4(decay=decay, cut='int')

                amplitude = []
                amplitudeBar = []

                p1_np = cp.asnumpy(p1)
                p2_np = cp.asnumpy(p2)
                p3_np = cp.asnumpy(p3)
                p1bar_np = cp.asnumpy(p1bar)
                p2bar_np = cp.asnumpy(p2bar)
                p3bar_np = cp.asnumpy(p3bar)

                #- p3
                data = [(p1_np[i], p2_np[i], p3_np[i]) for i in range(len(p1_np))]
                with Pool(processes=120) as pool:
                    amplitude.append(pool.map(load_int_amp, data))
                data_bar = [(p1bar_np[i], p3bar_np[i], p2bar_np[i]) for i in range(len(p1bar_np))]
                with Pool(processes=120) as pool:
                    amplitudeBar.append(pool.map(load_int_amp, data_bar))


                amp_array = cp.asarray(amplitude)
                ampbar_array = cp.asarray(amplitudeBar)

                print(cp.mean(amp_array))

                amp_array = amp_array
                ampbar_array = ampbar_array

                data_path = '/shared/scratch/pc24403/amp_ampgen'
                os.makedirs(data_path, exist_ok=True)
                cp.save(f'{data_path}/Int_{decay}_amp.npy', amp_array)
                cp.save(f'{data_path}/Int_{decay}_ampbar.npy', ampbar_array)

                end = time.time()
                print(f'Loading the {decay} amplitude model takes {end-start} seconds')


inital_amp()

    







