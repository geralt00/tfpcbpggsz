from importlib.machinery import SourceFileLoader
import uproot as up
import numpy as np
import time
import iminuit
import matplotlib.pyplot as plt
import mplhep
import awkward as ak
import pandas as pd
import math

import tensorflow as tf
tf.get_logger().setLevel('INFO')

time1 = time.time()

from tfpcbpggsz.core import *
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.amp import *


# import argparse
# parser = argparse.ArgumentParser(description='Signal only Fit')
# parser.add_argument('--index', type=int, default=1, help='Index of the toy MC')
# parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

# parser.print_help()
# args = parser.parse_args()
# index=args.index
# update = True

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import sys
import os
# sys.path.append("/software/rj23972/safety_net/qmi-gamma-measurement/python/")
from tfpcbpggsz.Includes.common_classes import *
from tfpcbpggsz.Includes.selections import *
from tfpcbpggsz.Includes.ntuples import *
from tfpcbpggsz.Includes.variables import *
from tfpcbpggsz.Includes.common_constants import *
from tfpcbpggsz.Includes.functions import *
from tfpcbpggsz.Includes.VARDICT import VARDICT

ntuples = {}
ntuples["SDATA"] = {}
ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"] = Ntuple("SDATA","CB2DK_D2KSPIPI_DD","YRUN2", "MagAll")

for source in ntuples.keys():
    BDT_cut_efficiency[source] = {}
    total_eff[source] = {}
    for channel in ntuples[source].keys():
        paths = ntuples[source][channel].final_cuts_paths
        list_var = [ntuples[source][channel].variable_to_fit]
        cut = "BDT > "+str(BDT_cut)
        ntuples[source][channel].store_events(paths,list_var,cut)
        BDT_cut_efficiency[source][channel] = len(ntuples[source][channel].uproot_data[list_var[0]]) / ntuples[source][channel].final_cuts_eff["selected_events"]
        total_eff[source][channel] = ntuples[source][channel].preliminary_cuts_eff["efficiency"]*ntuples[source][channel].final_cuts_eff["efficiency"]*BDT_cut_efficiency[source][channel]
        pass
    pass

Bmass_vec = np.arange(5080, 5800, 5)

print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].components)
print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].all_mass_pdfs)
print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].variable_to_fit)

test = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].pdf_values_draw(Bmass_vec,VARDICT["SDATA"]["CB2DK_D2KSPIPI_DD"])

what_to_plot = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
plt.figure(figsize=(10,6))
plt.title(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].tex)
mplhep.histplot(np.histogram(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].uproot_data[v_what_to_plot.name],bins=np.shape(Bmass_vec)[0],range=v_what_to_plot.range_value), label="$B^+$")
plt.plot(Bmass_vec,test["total_mass_pdf"],label="Total")
plt.plot(Bmass_vec,test["Dpi_Kspipi_misID"],linestyle="--",label="Dpi_Kspipi_misID")
plt.plot(Bmass_vec,test["DK_Kspipi"],linestyle="--",label="DK_Kspipi")
plt.legend()
plt.xlabel(v_what_to_plot.tex)
plt.xlabel(v_what_to_plot.tex)
plt.yscale(v_what_to_plot.scale)
plt.tight_layout()
plt.savefig(what_to_plot+'.png')
plt.close()





ntuples["MC_Bu_D0K_KSpipi"] = {}

ntuples["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"]  = Ntuple("MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow","CB2DK_D2KSPIPI_DD","YRUN2", "MagAll")
ntuples["MC_Bu_D0pi_KSpipi"] = {}
ntuples["MC_Bu_D0pi_KSpipi"]["CB2DK_D2KSPIPI_DD"] = Ntuple("MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow","CB2DK_D2KSPIPI_DD","YRUN2", "MagAll")


shared_parameters = {
    "signal_mean_DK": [
        ["CB2DK_KSPIPI_DD", "DK_Kspipi", "cruij_m0"  ],
        ["CB2DK_KSPIPI_DD", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSPIPI_LL", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSPIPI_LL", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSKK_LL", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSKK_LL", "DK_Kspipi", "gauss_mean"],
    ],
    "signal_mean_Dpi": [
        ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "cruij_m0"  ],
        ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "gauss_mean"],
    ],
    "signal_width_DK_DD": [
        ["CB2DK_KSPIPI_DD", "DK_Kspipi", "cruij_sigmaL"  ],
        ["CB2DK_KSPIPI_DD", "DK_Kspipi", "cruij_sigmaR"  ],
        ["CB2DK_KSPIPI_DD", "DK_Kspipi", "gauss_sigma"],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "cruij_sigmaL"  ],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "cruij_sigmaR"  ],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "gauss_sigma"],
    ],
    # "signal_width_DK_LL": [
    #     ["CB2DK_KSPIPI_LL", "DK_Kspipi", "cruij_sigmaL"  ],
    #     ["CB2DK_KSPIPI_LL", "DK_Kspipi", "cruij_sigmaR"  ],
    #     ["CB2DK_KSPIPI_LL", "DK_Kspipi", "gauss_sigma"],
    #     ["CB2DK_KSKK_LL", "DK_Kspipi", "cruij_sigmaL"  ],
    #     ["CB2DK_KSKK_LL", "DK_Kspipi", "cruij_sigmaR"  ],
    #     ["CB2DK_KSKK_LL", "DK_Kspipi", "gauss_sigma"],
    # ],
    "signal_width_Dpi_DD": [
        ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "cruij_sigmaL"  ],
        ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "cruij_sigmaR"  ],
        ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "gauss_width"],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "cruij_sigmaL"  ],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "cruij_sigmaR"  ],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "gauss_width"],
    ],
    # "signal_width_Dpi_LL": [
    #     ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "cruij_sigmaL"  ],
    #     ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "cruij_sigmaR"  ],
    #     ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "gauss_width"],
    #     ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "cruij_sigmaL"  ],
    #     ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "cruij_sigmaR"  ],
    #     ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "gauss_width"],
    # ],
    "Dpi_Kspipi_yield_DD": [
        ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "yield"  ],
    ],
}


BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
total_eff = {}
BDT_cut_efficiency = {}

for source in ntuples.keys():
    BDT_cut_efficiency[source] = {}
    total_eff[source] = {}
    for channel in ntuples[source].keys():
        paths = ntuples[source][channel].final_cuts_paths
        list_var = [ntuples[source][channel].variable_to_fit]
        cut = "BDT > "+str(BDT_cut)
        ntuples[source][channel].store_events(paths,list_var,cut)
        BDT_cut_efficiency[source][channel] = len(ntuples[source][channel].uproot_data[list_var[0]]) / ntuples[source][channel].final_cuts_eff["selected_events"]
        total_eff[source][channel] = ntuples[source][channel].preliminary_cuts_eff["efficiency"]*ntuples[source][channel].final_cuts_eff["efficiency"]*BDT_cut_efficiency[source][channel]
        pass
    pass



# constraining the DK yields to the Dpi one
ratio_DK_to_Dpi = {}
ratio_DK_to_Dpi["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"] = BR_B2DK/BR_B2Dpi * total_eff["MC_Bu_D0pi_KSpipi"]["CB2DK_D2KSPIPI_DD"]/total_eff["MC_Bu_D0pi_KSpipi"]["CB2DPI_D2KSPIPI_DD"]
# constraining the misID Dpi to its non-misID counterpart
ratio_Dpi_misID_to_Dpi = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] = total_eff["MC_Bu_D0pi_KSpipi"]["CB2Dpi_D2KSPIPI_DD"]/total_eff["MC_Bu_D0pi_KSpipi"]["CB2DK_D2KSPIPI_DD"]
# constraining the misID DK to its non-misID counterpart
ratio_DK_misID_to_DK = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] = total_eff["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"]/total_eff["MC_Bu_D0K_KSpipi"]["CB2Dpi_D2KSPIPI_DD"]


constrained_parameters = [
    [ ["CB2DK_KSPIPI_DD", "DK_Kspipi", "yield"], ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "yield", ratio_DK_to_Dpi["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"] ] ],
    [ ["CB2DK_KSPIPI_DD", "Dpi_Kspipi_misID", "yield"], ["CB2DPI_KSPIPI_DD", "Dpi_Kspipi", "yield", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ],
    [ ["CB2Dpi_KSPIPI_DD", "DK_Kspipi_misID", "yield"], ["CB2DK_KSPIPI_DD", "DK_Kspipi", "yield", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] ] ],
]

### the mean B mass of the DK and Dpi signals are shared accross samples
# i.e. variables_to_fit["CB2DK_KSPIPI_DD"]["DK_Kspipi"]["cruij_m0"]



@tf.function
def get_total_nll(vec_to_fit):
    variables_to_fit = dict(VARDICT["SDATA"])
    for shared_par in shared_parameters.keys(): ### loop over the input parameters
        for sharing in shared_parameters[shared_par]:
            ### loop over all parameters that should share this value
            variables_to_fit[sharing[0]][sharing[1]][sharing[2]] = vec_to_fit[shared_par]
            pass
        pass
    for const in constrained_parameters:
        variables_to_fit[const[0][0]][const[0][1]][const[0][2]] = variables_to_fit[const[1][0]][const[1][1]][const[1][2]]*const[1][3]
        pass
    total_nll = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].get_nll(variables_to_fit["CB2DK_D2KSPIPI_DD"]) + ntuples["SDATA"]["CB2Dpi_D2KSPIPI_DD"].get_nll(variables_to_fit["CB2Dpi_D2KSPIPI_DD"])
    return total_nll


@tf.function
def nll(x):
    vec_to_fit = dict(shared_parameters.keys(), x)
    return get_total_nll(vec_to_fit, fixed_constraints, gaussian_constraints)


x = [0., 0., 0., 0., 0., 0.] # fit param

m = iminuit.Minuit(nll, x)
# Minuit(least_squares_np, (5, 5), name=("a", "b"))
# m.limits = [(0, None), (0, 10)]
mg = m.migrad()
with open(f'{logpath}/simfit_output_{index}.txt', 'w') as f:
    print(mg, file=f)
    means = mg.values
    errors = mg.errors
#    print("Means", means['x0'], means['x1'], means['x2'], means['x3'], file=f)
#    print("Errors", errors['x0'], errors['x1'], errors['x2'], errors['x3'], file=f)
    print("Means", means['x0'], means['x1'], means['x2'], means['x3'], means['x4'], means['x5'], file=f)
    print("Errors", errors['x0'], errors['x1'], errors['x2'], errors['x3'], errors['x4'], errors['x5'], file=f)


time4 = time.time()
print(f'Mass builder finished in {time2-time1} seconds')
print(f'Amplitude builder finished in {time3-time2} seconds')
print(f'Fit finished in {time4-time3} seconds')
print(f'Total time: {time4-time1} seconds')
