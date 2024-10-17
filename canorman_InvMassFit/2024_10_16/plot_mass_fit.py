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
import json

import tensorflow as tf
tf.get_logger().setLevel('INFO')

time1 = time.time()

from tfpcbpggsz.core import *
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.amp import *

def get_integral(x, pdf_values):
    # Flatten the tensors for sorting purposes
    x_flat = tf.reshape(x, [-1])
    y_flat = tf.reshape(pdf_values, [-1])    
    # Get the sorted indices and sort both x and y
    sorted_indices = tf.argsort(x_flat)
    sorted_x = tf.gather(x_flat, sorted_indices)
    sorted_y = tf.gather(y_flat, sorted_indices)
    # Perform integration using tfp.math.trapz along the second axis
    norm_const = tfp.math.trapz(sorted_y, sorted_x)
    return norm_const


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
from tfpcbpggsz.Includes.VARDICT import VARDICT, varDict

ntuples = {}
total_eff = {}
BDT_cut_efficiency = {}
pre_cuts_eff = {}
fin_cuts_eff = {}
ntuples = {}
paths = {}
list_var = {}
cut = {}

BDT_cut = 0.4
Bmass_vec = np.arange(5080, 5800, 5)
tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)
BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
str_BDT_cut = "(BDT_output > "+str(BDT_cut)+")"


components = {
    "SDATA":
    {
        "CB2DK_D2KSPIPI_DD": [ 
            ["DK_Kspipi", "Cruijff+Gaussian"],
            ["Dpi_Kspipi_misID", "SumCBShape"],
            ["Dst0K_D0pi0_Kspipi", "HORNSdini"],
            # ["DstpK_D0pip_Kspipi", "HORNSdini"],
            ["Dst0pi_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["Dstppi_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["Dst0K_D0gamma_Kspipi", "HILLdini"],
            # ["Dst0pi_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID"],
            # ["DKpi_Kspipi", "HORNSdini+Gaussian"],
            # ["Dpipi_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["Bs2DKpi_Kspipi_PartReco", "HORNSdini"],
            ["Combinatorial", "Exponential"],
        ],
        "CB2DPI_D2KSPIPI_DD": [ 
            ["Dpi_Kspipi", "Cruijff+Gaussian"],
            ["DK_Kspipi_misID", "CBShape"],
            ["Dst0pi_D0pi0_Kspipi", "HORNSdini"],
            # ["Dstppi_D0pip_Kspipi", "HORNSdini"],
            # ["Dst0K_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["DstpK_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID"],
            ["Dst0pi_D0gamma_Kspipi", "HILLdini"],
            # ["Dst0K_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID"],
            # ["Dpipi_Kspipi", "HORNSdini+HORNSdini"],
            ["Combinatorial", "Exponential"],
        ]
    },
    "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow":
    {
        "CB2DK_D2KSPIPI_DD": [
            ["DK_Kspipi", "Cruijff+Gaussian"],
        ],
        "CB2DPI_D2KSPIPI_DD": [
            ["Dpi_Kspipi_misID", "SumCBShape"],
        ],
    },
    "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow":
    {
        "CB2DPI_D2KSPIPI_DD": [
            ["Dpi_Kspipi", "Cruijff+Gaussian"],
        ],
        "CB2DK_D2KSPIPI_DD": [
            ["DK_Kspipi_misID", "CBShape"],
        ],
    }
}

components_tex = {
    "DK_Kspipi": r"$B^{\pm} \rightarrow D K^{\pm}$",
    "Dpi_Kspipi_misID": r"$B^{\pm} \rightarrow D \pi^{\pm}$",
    "Dst0K_D0pi0_Kspipi": r"$B^{\pm} \rightarrow (D^{*0} \rightarrow D [\pi^0]) K^{\pm}$",
    "DstpK_D0pip_Kspipi": r"$B^0 \rightarrow (D^{*+} \rightarrow D [\pi^+]) K^{\pm}$",    
    "Dst0pi_D0pi0_Kspipi_misID_PartReco": r"$B^{\pm} \rightarrow (D^{*0} \rightarrow D [\pi^0]) \pi^{\pm}$",
    "Dstppi_D0pip_Kspipi_misID_PartReco": r"$B^0 \rightarrow (D^{*+} \rightarrow D [\pi^+]) \pi^{\pm}$",
    "Dst0K_D0gamma_Kspipi": r"$B^{\pm} \rightarrow (D^{*} \rightarrow D [\gamma]) K^{\pm}$",
    "Dst0pi_D0gamma_Kspipi_misID_PartReco": r"$B^{\pm} \rightarrow (D^{*} \rightarrow D [\gamma]) \pi^{\pm}$",
    "DKpi_Kspipi": r"$B \rightarrow D K^{\pm} [\pi]$",
    "Dpipi_Kspipi_misID_PartReco": r"$B \rightarrow D \pi^{\pm} [\pi]$",
    "Bs2DKpi_Kspipi_PartReco": r"$B_s \rightarrow D K^{\pm} [\pi^\pm]$",
    "Combinatorial": "Combinatorial",
    "Dpi_Kspipi":      r"$B^{\pm} \rightarrow D \pi^{\pm}$",
    "DK_Kspipi_misID": r"$B^{\pm} \rightarrow D K^{\pm}$",
    "Dst0pi_D0pi0_Kspipi": r"$B^{\pm} \rightarrow (D^{*0} \rightarrow D [\pi^0]) \pi^{\pm}$",
    "Dstppi_D0pip_Kspipi": r"$B^0 \rightarrow (D^{*+} \rightarrow D [\pi^+]) \pi^{\pm}$",    
    "Dst0K_D0pi0_Kspipi_misID_PartReco": r"$B^{\pm} \rightarrow (D^{*0} \rightarrow D [\pi^0]) K^{\pm}$",
    "DstpK_D0pip_Kspipi_misID_PartReco": r"$B^0 \rightarrow (D^{*+} \rightarrow D [\pi^+]) K^{\pm}$",    
    "Dst0pi_D0gamma_Kspipi": r"$B^{\pm} \rightarrow (D^{*} \rightarrow D [\gamma]) \pi^{\pm}$",
    "Dst0K_D0gamma_Kspipi_misID_PartReco": r"$B^{\pm} \rightarrow (D^{*} \rightarrow D [\gamma]) K^{\pm}$",
    "Dpipi_Kspipi": r"$B \rightarrow D \pi^{\pm} [\pi]$",
    "Combinatorial": "Combinatorial",
}


list_sources  = ["MC_Bu_D0K_KSpipi", "MC_Bu_D0pi_KSpipi"]
list_channels = ["CB2DK_D2KSPIPI_DD", "CB2DPI_D2KSPIPI_DD"]

#### channel CB2DK_D2KSPIPI_DD
### DK_Kspipi
for channel in list_channels:
    ntuples[channel] = {}
    pre_cuts_eff[channel] = {}
    fin_cuts_eff[channel] = {}
    for source in list_sources:
        ntuples[channel][source]  = Ntuple(source+"_TightCut_LooserCuts_fixArrow",channel,"YRUN2", "MagAll")
        print(ntuples[channel][source])
        pre_cuts_eff[channel][source] = ntuples[channel][source].get_merged_cuts_eff("preliminary")
        fin_cuts_eff[channel][source] = ntuples[channel][source].get_merged_cuts_eff("final")

for channel in list_channels:
    BDT_cut_efficiency[channel] = {}
    total_eff[channel] = {}
    cut[channel] = {}
    paths[channel] = {}
    list_var[channel] = {}
    for source in list_sources:
        cut[channel][source] = str_BDT_cut + " & " + ntuples[channel][source].dict_final_cuts["Bach_PID"]
        ntuples[channel][source].initialise_mass_fit(components)
        paths[channel][source] = ntuples[channel][source].final_cuts_paths
        list_var[channel][source] = [ntuples[channel][source].variable_to_fit]
        ntuples[channel][source].store_events(paths[channel][source],list_var[channel][source],cut[channel][source])
        BDT_cut_efficiency[channel][source] = len(ntuples[channel][source].uproot_data[list_var[channel][source][0]]) / fin_cuts_eff[channel][source]["YRUN2"]["MagAll"]["selected_events"]
        total_eff[channel][source] = pre_cuts_eff[channel][source]["YRUN2"]["MagAll"]["efficiency"]*fin_cuts_eff[channel][source]["YRUN2"]["MagAll"]["efficiency"]*BDT_cut_efficiency[channel][source]
        pass
    pass

print(json.dumps(total_eff,indent=4))


start_values = {
    "signal_mean_DK":  varDict['signal_mean'],
    "signal_mean_Dpi": varDict['signal_mean'],
    "signal_width_DK_DD":  varDict['sigma_dk_DD'],
    "signal_width_Dpi_DD": varDict['sigma_dk_DD'],
    "Dst0K_D0pi0_Kspipi_yield_DD": 7000.,
    "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": 3000.,
    "Combinatorial_yield_DK_DD": 12000.,
    "Dpi_Kspipi_yield_DD": 120000.,
    "Dst0pi_D0pi0_Kspipi_yield_DD": 7000.,
    "Dst0pi_D0gamma_Kspipi_yield_DD": 7000.,
    "Combinatorial_yield_DPI_DD": 20000.,
}

shared_parameters = {
    "signal_mean_DK": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "cruij_m0"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSPIPI_LL", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSPIPI_LL", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSKK_LL", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSKK_LL", "DK_Kspipi", "gauss_mean"],
    ],
    "signal_mean_Dpi": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "cruij_m0"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "gauss_mean"],
    ],
    "signal_width_DK_DD": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "cruij_sigmaL"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "cruij_sigmaR"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "gauss_sigma"],
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
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "cruij_sigmaL"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "cruij_sigmaR"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "gauss_width"],
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
    "Dst0K_D0pi0_Kspipi_yield_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "yield"],
    ],
    "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi_misID_PartReco", "yield"],
    ],
    # "DK_Kspipi_yield_DD": [ ## this should cause an error
    #     ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "yield"  ],
    # ],
    "Combinatorial_yield_DK_DD": [
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "yield"  ],
    ],
    "Dpi_Kspipi_yield_DD": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield"  ],
    ],
    "Dst0pi_D0pi0_Kspipi_yield_DD": [ 
        ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi", "yield"  ],
    ],
    "Dst0pi_D0gamma_Kspipi_yield_DD": [ 
        ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0gamma_Kspipi", "yield"  ],
    ],
    "Combinatorial_yield_DPI_DD": [
        ["CB2DPI_D2KSPIPI_DD", "Combinatorial", "yield"  ],
    ],
}
        
        
# constraining the DK yields to the Dpi one
ratio_DK_to_Dpi = {}
ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] = BR_B2DK/BR_B2Dpi * total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]/total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]
# constraining the misID Dpi to its non-misID counterpart
ratio_Dpi_misID_to_Dpi = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] = total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]/total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]
# constraining the misID DK to its non-misID counterpart
ratio_DK_misID_to_DK = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] = total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]/total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
        

constrained_parameters = [
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi"       , "yield"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield", ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ],
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "yield"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ],
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "yield"], ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "yield", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] ] ],
]

with open(f'simfit_output.json') as f:
    vec_results = json.load(f)
    pass


variables_post_fit = dict(VARDICT["SDATA"])
for shared_par in vec_results.keys(): ### loop over the input parameters
    for sharing in shared_parameters[shared_par]:
        ### loop over all parameters that should share this value
        # print(sharing)
        variables_post_fit[sharing[0]][sharing[1]][sharing[2]] = vec_results[shared_par]
        pass
    pass



### data
pdf_values_prefit = {}
for channel in list_channels:
    pdf_values_prefit[channel] = {}
    ntuples[channel]["SDATA"] = Ntuple("SDATA",channel,"YRUN2", "MagAll")
    ntuples[channel]["SDATA"].initialise_mass_fit(components)
    print(ntuples[channel]["SDATA"])
    print(ntuples[channel]["SDATA"].components)
    print(ntuples[channel]["SDATA"].variable_to_fit)
    cut[channel]["SDATA"] = str_BDT_cut + " & " + ntuples[channel]["SDATA"].dict_final_cuts["Bach_PID"]
    paths[channel]["SDATA"] = ntuples[channel]["SDATA"].final_cuts_paths
    list_var[channel]["SDATA"] = [ntuples[channel]["SDATA"].variable_to_fit]
    ntuples[channel]["SDATA"].store_events(paths[channel]["SDATA"],list_var[channel]["SDATA"],cut[channel]["SDATA"])
    pdf_values_prefit[channel]["SDATA"] = ntuples[channel]["SDATA"].pdf_values_draw(Bmass_vec,variables_post_fit[channel])
scale = 5
fig, ax = plt.subplots(1,2,figsize=(15, 8))
plt.suptitle(ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].source.tex + " " + ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].year.tex)
##### DK
what_to_plot = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
data_to_plot = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].uproot_data.query( "("+what_to_plot+" < 5800) & ("+what_to_plot+" > 5080)" )[v_what_to_plot.name]
mplhep.histplot(np.histogram(data_to_plot,bins=int((5800-5080)/5),range=[5080,5800]), label="Run 2 data", ax=ax[0])
ax[0].plot(Bmass_vec,scale*pdf_values_prefit["CB2DK_D2KSPIPI_DD"]["SDATA"]["total_mass_pdf"],label="Total")
for comp in ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].components:
    ax[0].plot(Bmass_vec,pdf_values_prefit["CB2DK_D2KSPIPI_DD"]["SDATA"][comp[0]]*scale,linestyle="--",label=comp[0]+"\n"+comp[1])
ax[0].legend(title=ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].channel.tex)
ax[0].set_xlim(5080,5800)
ax[0].set_xlabel(v_what_to_plot.tex)
ax[0].set_xlabel(v_what_to_plot.tex)
##### DPI
what_to_plot = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
data_to_plot = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].uproot_data.query( "("+what_to_plot+" < 5800) & ("+what_to_plot+" > 5080)" )[v_what_to_plot.name]
mplhep.histplot(np.histogram(data_to_plot,bins=int((5800-5080)/5),range=[5080,5800]), label="Run 2 data", ax=ax[1])
ax[1].plot(Bmass_vec,scale*pdf_values_prefit["CB2DPI_D2KSPIPI_DD"]["SDATA"]["total_mass_pdf"],label="Total")
for comp in ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].components:
    ax[1].plot(Bmass_vec,pdf_values_prefit["CB2DPI_D2KSPIPI_DD"]["SDATA"][comp[0]]*scale,linestyle="--",label=comp[0]+"\n"+comp[1])
ax[1].legend(title=ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].channel.tex)
ax[1].set_xlim(5080,5800)
ax[1].set_xlabel(v_what_to_plot.tex)
ax[1].set_xlabel(v_what_to_plot.tex)
plt.tight_layout()
plt.savefig(what_to_plot+'.png')
plt.savefig(what_to_plot+'.pdf')
plt.close()




# with open(f'simfit_output.txt', 'w') as f:
#     print(mg, file=f)
#     means = mg.values
#     errors = mg.errors
#     print("Means", means['x0'], means['x1'], means['x2'], means['x3'], means['x4'], means['x5'], file=f)
#     print("Errors", errors['x0'], errors['x1'], errors['x2'], errors['x3'], errors['x4'], errors['x5'], file=f)
#     #    print("Means", means['x0'], means['x1'], means['x2'], means['x3'], file=f)
#     #    print("Errors", errors['x0'], errors['x1'], errors['x2'], errors['x3'], file=f)
    
    
#     time4 = time.time()
#     print(f'Mass builder finished in {time2-time1} seconds')
#     print(f'Amplitude builder finished in {time3-time2} seconds')
#     print(f'Fit finished in {time4-time3} seconds')
#     print(f'Total time: {time4-time1} seconds')
#     pass
# if False:
#     pass
