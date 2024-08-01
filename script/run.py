# test.py
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import sys
from tfpcbpggsz.amp import * 
import uproot as up
import os
from multiprocessing import Pool

Kspipi = PyD0ToKSpipi2018()
Kspipi.init()
def get_mass(p1,p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)

def get_p4(decay="b2dpi", cut='', index=1):

    file_name = ''
    if cut == 'int':
        file_name = f'/software/pc24403/PCBPGGSZ/Int/test/weighted_{decay}.root:DalitzEventList'
    if cut == 'Int':
        file_name = f'/software/pc24403/PCBPGGSZ/Int/flat_{decay}.root:DalitzEventList'
    
    #elif decay.split('_')[0] == 'b2dk' or decay.split('_')[0] == 'b2dpi':
    #    if cut == 'p':
    #        file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/lhcb_toy_{decay}_{index}_CPrange.root:BplusDalitzEventList'
    #    else:
    #        file_name = f'/software/pc24403/PCBPGGSZ/outputs/oy/mass_fit/add_sw/lhcb_toy_{decay}_{index}.root:BminusDalitzEventList'

    tree = up.open(file_name)
  # Load the branches as arrays
    branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
    array = tree.arrays(branch_names)
       

    _p1 = np.array([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
    _p2 = np.array([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
    _p3 = np.array([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
    # convert 4*1000 into a vectot<double>
    _p1 = np.transpose(_p1)
    _p2 = np.transpose(_p2)
    _p3 = np.transpose(_p3)

    p1 = _p1
    p2 = _p2
    p3 = _p3

    p1bar = np.hstack((_p1[:, :1], np.negative(_p1[:, 1:])))
    p2bar = np.hstack((_p2[:, :1], np.negative(_p2[:, 1:])))
    p3bar = np.hstack((_p3[:, :1], np.negative(_p3[:, 1:])))


    return p1, p2, p3, p1bar, p2bar, p3bar


def load_int_amp(args):
    p1, p2, p3 = args

    return Kspipi.AMP(p1.tolist(), p2.tolist(), p3.tolist())

def inital_amp():
    # Load the amplitude model

    print('Loading the amplitude model')
    

    Bdecays = ['b2dk']
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

                p1_np = np.array(p1)
                p2_np = np.array(p2)
                p3_np = np.array(p3)
                p1bar_np = np.array(p1bar)
                p2bar_np = np.array(p2bar)
                p3bar_np = np.array(p3bar)

                #- p3
                data = [(p1_np[i], p2_np[i], p3_np[i]) for i in range(len(p1_np))]
                amplitude = [load_int_amp(args) for args in data]
                data_bar = [(p1bar_np[i], p3bar_np[i], p2bar_np[i]) for i in range(len(p1bar_np))]
                amplitudeBar = [load_int_amp(args) for args in data_bar]


                amp_array = np.array(amplitude)
                ampbar_array = np.negative(np.array(amplitudeBar))

                print(np.mean(amp_array))

                amp_array = amp_array
                ampbar_array = ampbar_array

                data_path = '/shared/scratch/pc24403/amp_ampgen_big'
                os.makedirs(data_path, exist_ok=True)
                np.save(f'{data_path}/Int_{decay}_amp.npy', amp_array)
                np.save(f'{data_path}/Int_{decay}_ampbar.npy', ampbar_array)

                end = time.time()
                print(f'Loading the {decay} amplitude model takes {end-start} seconds')

def inital_amp_noeff():
    # Load the amplitude model

    print('Loading the amplitude model')
    

    Bdecays = ['b2dk', 'b2dpi']
    Types = ['DD', 'LL']
    Charges = ['p', 'm']
    for bdecay in Bdecays:
        for type in Types:
            for charge in Charges:
                start = time.time()

                decay = f'{bdecay}_{type}_{charge}'
                p1, p2, p3, p1bar, p2bar, p3bar = get_p4(decay=decay, cut='Int')

                amplitude = []
                amplitudeBar = []

                p1_np = np.array(p1)
                p2_np = np.array(p2)
                p3_np = np.array(p3)
                p1bar_np = np.array(p1bar)
                p2bar_np = np.array(p2bar)
                p3bar_np = np.array(p3bar)

                #- p3
                data = [(p1_np[i], p2_np[i], p3_np[i]) for i in range(len(p1_np))]
                with Pool(processes=120) as pool:
                    amplitude.append(pool.map(load_int_amp, data))
                data_bar = [(p1bar_np[i], p3bar_np[i], p2bar_np[i]) for i in range(len(p1bar_np))]
                with Pool(processes=120) as pool:
                    amplitudeBar.append(pool.map(load_int_amp, data_bar))


                amp_array = np.array(amplitude)
                ampbar_array = np.array(amplitudeBar)

                ampbar_array = np.negative(ampbar_array)

                #Divide the array with 1M for each file

                

                data_path = '/shared/scratch/pc24403/amp_ampgen_noeff'
                os.makedirs(data_path, exist_ok=True)

                np.save(f'{data_path}/Int_{decay}_amp.npy', amp_array)
                np.save(f'{data_path}/Int_{decay}_ampbar.npy', ampbar_array)

                end = time.time()
                print(f'Loading the {decay} amplitude model takes {end-start} seconds')


inital_amp()

    







