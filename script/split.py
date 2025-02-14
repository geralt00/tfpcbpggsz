import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import uproot as up
from multiprocessing import Pool
def get_p4():

    file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/root/tuple.root:DalitzEventList'
    
    tree = up.open(file_name)
  # Load the branches as arrays
    branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
    array = tree.arrays(branch_names)
       


    return array


large_arr = get_p4()

def do_split():
    
    Bdecays = ['b2dpi', 'b2dk']
    Types = ['DD', 'LL']
    Charges = ['p', 'm']
    index=0
    for bdecay in Bdecays:
        for type in Types:
            for charge in Charges:
                decay = f'{bdecay}_{type}_{charge}'
                print(f'Processing {decay}')
                length = 1000
                arr=[]
                if bdecay == 'b2dk':
                    length = 1000000
                    arr=large_arr[index*length:(index+1)*length]
                if bdecay == 'b2dpi':
                    length = 5000000
                    arr=large_arr[index*length:(index+1)*length]
                df = pd.DataFrame(arr,columns=[""])
                f = up.recreate(f'/software/pc24403/PCBPGGSZ/Int/flat_{decay}.root')
                f['DalitzEventList'] = df
                index+=1
do_split()
