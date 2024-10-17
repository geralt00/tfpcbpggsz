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

"""Default configuration options for `matplotlib`"""
from matplotlib import rc, rcParams
rc('text', usetex=True)
rc('font',**{'family':'serif',
             'serif':['Computer Modern Roman','cmu serif']+rcParams['font.serif']})
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['axes.labelsize'] = 14
rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 8
rcParams["figure.figsize"] = (12, 6)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
kwargs_data = {
    "xerr"        : 2.5,
    "histtype"    : 'errorbar',
    "linestyle"   : "None",
    "color"       : "black",
    "markersize" : 2,
    "capsize" : 1.5,    
}

time1 = time.time()
#### scale = bin width for plotting purposes
scale = 5

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
# index= args.index
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
from tfpcbpggsz.Includes.Measurement import Measurement
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
Bmass_vec = np.arange(5080, 5800, 1)
tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)
BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
str_BDT_cut = "(BDT_output > "+str(BDT_cut)+")"

################ define the components in each channel and source
components = {
    "SDATA":
    {
        "CB2DK_D2KSPIPI_DD": [ 
            ["DK_Kspipi", "Cruijff+Gaussian"],
            ["Dpi_Kspipi_misID", "SumCBShape"],
            ["Dst0K_D0pi0_Kspipi", "HORNSdini"],
            ["DstpK_D0pip_Kspipi", "HORNSdini"],
            ["Dst0pi_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID"],
            ["Dstppi_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID"],
            ["Dst0K_D0gamma_Kspipi", "HILLdini"],
            ["Dst0pi_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID"],
            ["DKpi_Kspipi", "HORNSdini+Gaussian"],
            ["Dpipi_Kspipi_misID_PartReco", "HORNSdini_misID"],
            ["Bs2DKpi_Kspipi_PartReco", "HORNSdini"],
            ["Combinatorial", "Exponential"],
        ],
        "CB2DPI_D2KSPIPI_DD": [ 
            ["Dpi_Kspipi", "Cruijff+Gaussian"],
            ["DK_Kspipi_misID", "CBShape"],
            ["Dst0pi_D0pi0_Kspipi", "HORNSdini"],
            ["Dstppi_D0pip_Kspipi", "HORNSdini"],
            ["Dst0K_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID"],
            ["DstpK_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID"],
            ["Dst0pi_D0gamma_Kspipi", "HILLdini"],
            ["Dst0K_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID"],
            ["Dpipi_Kspipi", "HORNSdini+HORNSdini"],
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


################ their tex formatting
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

################ the sources and fits we want the efficiencies of
list_sources  = ["MC_Bu_D0K_KSpipi", "MC_Bu_D0pi_KSpipi"]
list_channels = ["CB2DK_D2KSPIPI_DD", "CB2DPI_D2KSPIPI_DD"]

#### get MC ntuples and their efficiencies
for channel in list_channels:
    ntuples[channel] = {}
    pre_cuts_eff[channel] = {}
    fin_cuts_eff[channel] = {}
    for source in list_sources:
        ntuples[channel][source]  = Ntuple(source+"_TightCut_LooserCuts_fixArrow",channel,"YRUN2", "MagAll")
        print(ntuples[channel][source])
        pre_cuts_eff[channel][source] = ntuples[channel][source].get_merged_cuts_eff("preliminary")
        fin_cuts_eff[channel][source] = ntuples[channel][source].get_merged_cuts_eff("final")
########### here we get BDT and PID eff that are not yet in the final ntuples
for channel in list_channels:
    BDT_cut_efficiency[channel] = {}
    total_eff[channel] = {}
    cut[channel] = {}
    paths[channel] = {}
    list_var[channel] = {}
    for source in list_sources:
        ntuples[channel][source].initialise_mass_fit(components)
        cut[channel][source] = str_BDT_cut + " & " + ntuples[channel][source].dict_final_cuts["Bach_PID"] + " & (" + ntuples[channel][source].variable_to_fit + " < 5800 ) & (" + ntuples[channel][source].variable_to_fit + " > 5080 )"
        paths[channel][source] = ntuples[channel][source].final_cuts_paths
        list_var[channel][source] = [ntuples[channel][source].variable_to_fit]
        ntuples[channel][source].store_events(paths[channel][source],list_var[channel][source],cut[channel][source])
        BDT_cut_efficiency[channel][source] = len(ntuples[channel][source].uproot_data[list_var[channel][source][0]]) / fin_cuts_eff[channel][source]["YRUN2"]["MagAll"]["selected_events"]
        total_eff[channel][source] = pre_cuts_eff[channel][source]["YRUN2"]["MagAll"]["efficiency"]*fin_cuts_eff[channel][source]["YRUN2"]["MagAll"]["efficiency"]*BDT_cut_efficiency[channel][source]
        pass
    pass

print(json.dumps(total_eff,indent=4))
            
######## initialise data ntuples
for channel in list_channels:
    # i_s = INDEX_SOURCE_TO_VARDICT["SDATA"]
    i_c = INDEX_CHANNEL_TO_VARDICT[channel]
    ntuples[channel]["SDATA"] = Ntuple("SDATA",channel,"YRUN2", "MagAll")
    ntuples[channel]["SDATA"].initialise_mass_fit(components)
    print(ntuples[channel]["SDATA"])
    print(ntuples[channel]["SDATA"].components)
    print(ntuples[channel]["SDATA"].variable_to_fit)
    cut[channel]["SDATA"] = str_BDT_cut + " & " + ntuples[channel]["SDATA"].dict_final_cuts["Bach_PID"] + " & (" + ntuples[channel][source].variable_to_fit + " < 5800 ) & (" + ntuples[channel][source].variable_to_fit + " > 5080 )"
    paths[channel]["SDATA"] = ntuples[channel]["SDATA"].final_cuts_paths
    list_var[channel]["SDATA"] = [ntuples[channel]["SDATA"].variable_to_fit]
    ntuples[channel]["SDATA"].store_events(paths[channel]["SDATA"],list_var[channel]["SDATA"],cut[channel]["SDATA"])


                
######## define starting values of the variables in the fit
start_values = {
    "ratio_BR_DK_to_Dpi" :                         0.07,
    "signal_mean_DK":                              varDict['signal_mean']+50,
    "signal_mean_Dpi":                             varDict['signal_mean']+50,
    "signal_width_DK_DD":                          varDict['sigma_dk_DD']+50,
    "signal_width_Dpi_DD":                         varDict['sigma_dk_DD']+50,
    "Dst0K_D0pi0_Kspipi_yield_DD":                 3000.,
    "DstpK_D0pip_Kspipi_yield_DD":                 3000.,
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": 3000.,
    "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": 3000.,
    "Dst0K_D0gamma_Kspipi_yield_DD"              : 3000.,
    "Combinatorial_yield_DK_DD":                   12000.,
    "Dpi_Kspipi_yield_DD":                         120000.,
    "Dst0pi_D0pi0_Kspipi_yield_DD":                7000.,
    "Dst0pi_D0gamma_Kspipi_yield_DD":              7000.,
    "Combinatorial_yield_DPI_DD":                  20000.,                     
    "Combinatorial_c_DK_DD":                       -0.003,
}

####### define limits 
limit_values = {
    "ratio_BR_DK_to_Dpi" :                         [0.0001,1],
    "signal_mean_DK":                              [5080,5800],
    "signal_mean_Dpi":                             [5080,5800],
    "signal_width_DK_DD":                          [1,100],
    "signal_width_Dpi_DD":                         [1,100],
    "Dst0K_D0pi0_Kspipi_yield_DD":                 [0, 20000],
    "DstpK_D0pip_Kspipi_yield_DD":                 [0, 20000],
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": [0, 20000],
    "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": [0, 20000],
    "Dst0K_D0gamma_Kspipi_yield_DD"              : [0, 20000],
    "Combinatorial_yield_DK_DD":                   [0, 50000],
    "Dpi_Kspipi_yield_DD":                         [0,400000],
    "Dst0pi_D0pi0_Kspipi_yield_DD":                [0,400000],
    "Dst0pi_D0gamma_Kspipi_yield_DD":              [0,400000],
    "Combinatorial_yield_DPI_DD":                  [0,400000],
    "Combinatorial_c_DK_DD":                       [-0.01,-0.0005],
}

###### define which free fit parameters is applied to which "real" variables
dict_shared_parameters = {
    "ratio_BR_DK_to_Dpi" : [
        ["SHARED_THROUGH_CHANNELS", "parameters", "ratio_BR_DK_to_Dpi"]
    ],
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
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "gauss_sigma"],
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
    "DstpK_D0pip_Kspipi_yield_DD": [
        ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "yield"],
    ],
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": [
    #     ["CB2DK_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi_misID_PartReco", "yield"],
    # ],
    "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dstppi_D0pip_Kspipi_misID_PartReco", "yield"],
    ],
    "Dst0K_D0gamma_Kspipi_yield_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dst0K_D0gamma_Kspipi", "yield"],
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
    "Combinatorial_c_DK_DD": [ # Exponential 11
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "c"],
    ],
}

# https://gitlab.cern.ch/lhcb-b2oc/analyses/GGSZ-B2Dh/-/blob/master/common_inputs/pid_efficiencies/pid_4.0_Run1and2.settings.txt?ref_type=heads
# https://gitlab.cern.ch/lhcb-b2oc/analyses/GGSZ-B2Dh/-/blob/master/common_inputs/selection_efficiencies/selection_eff.settings.txt?ref_type=heads
# 0.97*942.908711914/(0.86*928.554348882)

############# compute ratio of efficiencies etc to constrain parameters
# constraining the DK yields to the Dpi one
ratio_DK_to_Dpi = {}
ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] = total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]/total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
####### Note: in the previous ana, this was 0.076 #
# https://gitlab.cern.ch/lhcb-b2oc/analyses/GGSZ-B2Dh/-/blob/master/fits/B2DhGGSZ/src/B2DhModelDefault.cpp?ref_type=heads#L53
# constraining the misID Dpi to its non-misID counterpart
ratio_Dpi_misID_to_Dpi = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] = total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]/total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]
# ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dst0pi_D0pi0_Kspipi_misID_PartReco"] = total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]/total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
# constraining the misID DK to its non-misID counterpart
ratio_DK_misID_to_DK = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] = total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]/total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
# constraining the misID DK to its non-misID counterpart - Part Reco
        
######### define which real variable is constrained from which other var
# (only implemented with multiplicative factor)
dict_constrained_parameters = [
    ## constrain DK from Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi"       , "yield"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield",
                                                            [ "SHARED_THROUGH_CHANNELS", "parameters", "ratio_BR_DK_to_Dpi", ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ] ],
    ## constrain misID in DK from goodID in Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "yield"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] ] ],
    ## constrain misID DPI from goodID in DK
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "yield"], ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "yield", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] ] ],
]
print(json.dumps(dict_constrained_parameters,indent=4))


ratio_Dst0K_to_DstpK = Measurement(1,0.1)
######### define which real variable is constrained from which other var
# (only implemented with multiplicative factor)
dict_gaussian_constraints = [
        [ ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "yield"], ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "yield", ratio_Dst0K_to_DstpK.value, ratio_Dst0K_to_DstpK.error] ],
]
print(json.dumps(dict_gaussian_constraints,indent=4))


######### simply the list of variables that will be fitted
parameters_to_fit = list(start_values.keys())

########### and now, necessary magic to go from dictionaries to "TensorFlowable" lists
## shared parameters
shared_parameters = [] # list_variables
for i in range(len(parameters_to_fit)):
    par_name = parameters_to_fit[i]
    print(par_name)
    shared_parameters.append([])# each parameter is applied to a list of different "real" variables
    list_sharing_parameters = dict_shared_parameters[par_name]
    print(list_sharing_parameters)
    for var in list_sharing_parameters:
        i_channel   = list(VARDICT["SDATA"].keys()).index(var[0])
        i_component = list(VARDICT["SDATA"][var[0]].keys()).index(var[1])
        i_variable  = list(VARDICT["SDATA"][var[0]][var[1]].keys()).index(var[2])
        shared_parameters[i].append([i_channel, i_component, i_variable])
        pass
    print(shared_parameters[i])
    pass
#### now, shared_parameters contain the indices of the variables
# of the free parameters to be fitted
# such that, with x the vector of fit parameters,
# x[0] = all the parameters in the list shared_parameters[0]
print(shared_parameters)

## constrain
constrained_parameters = [] # list_variables
for i in range(len(dict_constrained_parameters)):
    constrained_parameters.append([])
    to_const = dict_constrained_parameters[i][0]
    const    = dict_constrained_parameters[i][1]
    print(to_const, " is constrained to ", const)
    to_const_channel   = list(VARDICT["SDATA"].keys()).index(to_const[0])
    to_const_component = list(VARDICT["SDATA"][to_const[0]].keys()).index(to_const[1])
    to_const_variable  = list(VARDICT["SDATA"][to_const[0]][to_const[1]].keys()).index(to_const[2])
    const_channel   = list(VARDICT["SDATA"].keys()).index(const[0])
    const_component = list(VARDICT["SDATA"][const[0]].keys()).index(const[1])
    const_variable  = list(VARDICT["SDATA"][const[0]][const[1]].keys()).index(const[2])
    const_value     = const[3]
    if (type(const_value) == list):
        ii_const_channel   = list(VARDICT["SDATA"].keys()).index(const_value[0])
        ii_const_component = list(VARDICT["SDATA"][const_value[0]].keys()).index(const_value[1])
        ii_const_variable  = list(VARDICT["SDATA"][const_value[0]][const_value[1]].keys()).index(const_value[2])
        const_value = [ii_const_channel, ii_const_component, ii_const_variable, const_value[3]]
        pass
    constrained_parameters[i] = [
        [to_const_channel, to_const_component, to_const_variable],
        [const_channel, const_component, const_variable, const_value],
    ]
    pass
#### now, constrained_parameters contain the list of variables
# to be constrained from other variables
# first element are the indices of the variable to constrain
# second element are the indices of the variable constraining + the multiplicative factor
print(constrained_parameters)
# tf_constrained_parameters = tf.constant(constrained_parameters)


## constrain
gaussian_constraints = [] # list_variables
for i in range(len(dict_gaussian_constraints)):
    gaussian_constraints.append([])
    to_const = dict_gaussian_constraints[i][0]
    const    = dict_gaussian_constraints[i][1]
    print(to_const, " is gaussian constrained to ", const)
    to_const_channel   = list(VARDICT["SDATA"].keys()).index(to_const[0])
    to_const_component = list(VARDICT["SDATA"][to_const[0]].keys()).index(to_const[1])
    to_const_variable  = list(VARDICT["SDATA"][to_const[0]][to_const[1]].keys()).index(to_const[2])
    const_channel   = list(VARDICT["SDATA"].keys()).index(const[0])
    const_component = list(VARDICT["SDATA"][const[0]].keys()).index(const[1])
    const_variable  = list(VARDICT["SDATA"][const[0]][const[1]].keys()).index(const[2])
    const_value     = const[3]
    const_error     = const[4]
    gaussian_constraints[i] = [
        [to_const_channel, to_const_component, to_const_variable],
        [const_channel, const_component, const_variable, const_value, const_error],
    ]
    pass
#### now, gaussian_constraints contain the list of variables
# to be gaussian constrained from other variables times (x +- err)
print(gaussian_constraints)
# tf_constrained_parameters = tf.constant(constrained_parameters)

#### TensorFlowing our dictionaries
# tf_start_values = tf.convert_to_tensor(list(start_values.values()))
# tf_limit_values = tf.convert_to_tensor(list(limit_values.values()))
### this is tiring: I need to make list_vardict rectangular
## this won't be aproblem since all variables
# are represented by 3 numbers [channel, component, variable]
# so we just add 0 to variables that don't exist

### list_vardict contains only the variables of the components we are looking at
# and in the same order as the one in components
# i.e. the same order as the one defines in the Ntuple object
NUMBER_OF_CHANNELS = len(list_channels) + 1

list_vardict = [] # first index is source
for i in range(NUMBER_OF_CHANNELS):
    channel = list(VARDICT["SDATA"].keys())[i]
    print("channel")
    print(channel)
    list_vardict.append([]) # second index is channel
    for component in VARDICT["SDATA"][channel]:
        ### third index is component
        print("component")
        list_vardict[-1].append([])
        print(component)
        for variables_values in VARDICT["SDATA"][channel][component].values():
            list_vardict[-1][-1].append(variables_values)
            pass
        pass
    print("end channel")
    print(" ")
    pass

MAX_ONE = max([len(list_vardict[i]) for i in range(len(list_vardict))])
tmp_list = []
for i in range(len(list_vardict)):
    tmp_list += [len(list_vardict[i][j]) for j in range(len(list_vardict[i])) ]
    pass
MAX_TWO = max(tmp_list)
print(MAX_ONE, MAX_TWO)
for i in range(len(list_vardict)): ## loop channel
    while (len(list_vardict[i]) < MAX_ONE): 
        list_vardict[i].append([])
        pass
    for j in range(len(list_vardict[i])): ## loop component
        while (len(list_vardict[i][j]) < MAX_TWO): 
            list_vardict[i][j].append(0)
            pass
        pass
    pass

# tf_list_vardict = tf.constant(list_vardict)

#### sanity check: we don't want a parameter that is free in the fit to also be constrained
def sanity_checks():
    if not ( (len(list(dict_shared_parameters.keys())) == len(list(start_values.keys()))) and (len(list(dict_shared_parameters.keys())) == len(list(limit_values.keys()))) ):
        print("ERROR ---------- There's a problem in the input dictionaries, please check")
        return False
    for const in dict_constrained_parameters:
        for shared in dict_shared_parameters.values():
            if (const[0] in shared):
                print(const[0])
                if (sanity_checks(const[0]) == False) :
                    print("whopopopopo not so fast")
                    print("const: ", const)
                    print(" this line constrains a parameter that is also in shared_parameters i.e. is free in the fit --- please checl")
                    return False
                ### that means that a parameter that is constrained
                ## is also registered as a free parameter of the fit:
                # that shouldn't happened and needs to be fixed
                pass
            pass
        pass
    return True

if (sanity_checks() == False):
    print("ERROR ---------- SANITY CHECKS NOT PASSED --- CHECK")
    exit()
else:
    print(" On a encore eu d'la chance !")
    
@tf.function
def get_total_nll(x): # , tensor_to_fit):
    #### the indices for tensor_to_fit are defined in INDEX_CHANNEL_TO_????
    # print(tensor_to_fit[0])
    # total_nll = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].get_nll(x,tensor_to_fit) + ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].get_nll(tensor_to_fit[1])
    total_nll = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].get_nll(x, list_vardict, shared_parameters, constrained_parameters, components) + ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].get_nll(x, list_vardict, shared_parameters, constrained_parameters, components, gaussian_constraints=gaussian_constraints)
    return total_nll
    
    
@tf.function
def nll(x):
    ### here we do some formatting: we want to translate the vector of parameters to
    # a multi D tensor with proper indices
    # tensor_to_fit = tf.identity(tf_list_vardict))
    #### first, we grab the i'th value of x and
    # give it to all parameters at position i in shared_parameters
    # print(tensor_to_fit)
    # for i in range(len(shared_parameters)):
    #     for par in shared_parameters[i]:
    #         # print(par)
    #         previous = tensor_to_fit[par[0]][par[1]][par[2]]
    #         tensor_to_fit[par[0]][par[1]][par[2]].assign(x[i])
    #         pass
    #     pass
    #### then, we need to apply the constraints
    # print("COINSTRAING")
    # print("COINSTRAING")    
    # for i in range(len(constrained_parameters)):
    #     to_const = constrained_parameters[i][0]
    #     const    = constrained_parameters[i][1]
    #     print("Before: ", tensor_to_fit[to_const[0]][to_const[1]][to_const[2]])
    #     # print(to_const)
    #     # print(const)
    #     value_const = tensor_to_fit[const[0]][const[1]][const[2]]
    #     value_to_const = value_const*const[3]
    #     tensor_to_fit[to_const[0]][to_const[1]][to_const[2]] = value_to_const
    #     print("After : ", tensor_to_fit[to_const[0]][to_const[1]][to_const[2]])
    #     pass
    # tf.print(x)
    return get_total_nll(x) # , tensor_to_fit)


# print("nll for x = start_values : ",nll(x).numpy())    
# x = [1000. for i in range(len(parameters_to_fit))] # fit param
# print("nll for x =         1000 : ",nll(x).numpy())
# x = [0. for i in range(len(parameters_to_fit))] # fit param
# print("nll for x =           0. : ",nll(x).numpy())
    

############### pre-fit
variables_pre_fit = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].get_mass_pdf_functions(list_vardict, params=list(start_values.values()), shared_parameters=shared_parameters, constrained_parameters=constrained_parameters)
#### pdf_values
### data
pdf_values_prefit = {}
for channel in list_channels:
    i_c = INDEX_CHANNEL_TO_VARDICT[channel]
    pdf_values_prefit[channel] = {}
    pdf_values_prefit[channel]["SDATA"] = ntuples[channel]["SDATA"].pdf_values_draw(tf_Bmass_vec,variables_pre_fit[i_c])
############### plot pre-fit
fig, ax = plt.subplots(1,2)
plt.suptitle(ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].source.tex + " " + ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].year.tex)
##### DK
what_to_plot = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
data_to_plot = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].uproot_data.query( "("+what_to_plot+" < 5800) & ("+what_to_plot+" > 5080)" )[v_what_to_plot.name]
mplhep.histplot(np.histogram(data_to_plot,bins=int((5800-5080)/5),range=[5080,5800]), label="Run 2 data", ax=ax[0],  **kwargs_data)
ax[0].plot(Bmass_vec,scale*pdf_values_prefit["CB2DK_D2KSPIPI_DD"]["SDATA"]["total_mass_pdf"],label="Total")
for comp in ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].components:
    ax[0].plot(Bmass_vec,pdf_values_prefit["CB2DK_D2KSPIPI_DD"]["SDATA"][comp[0]]*scale,linestyle="--",label=components_tex[comp[0]]+"\n"+comp[1])
ax[0].legend(title=ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].channel.tex)
ax[0].set_xlim(5080,5800)
ax[0].set_xlabel(v_what_to_plot.tex)
ax[0].set_xlabel(v_what_to_plot.tex)
##### DPI
what_to_plot = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
data_to_plot = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].uproot_data.query( "("+what_to_plot+" < 5800) & ("+what_to_plot+" > 5080)" )[v_what_to_plot.name]
mplhep.histplot(np.histogram(data_to_plot,bins=int((5800-5080)/5),range=[5080,5800]), label="Run 2 data", ax=ax[1], **kwargs_data)
ax[1].plot(Bmass_vec,scale*pdf_values_prefit["CB2DPI_D2KSPIPI_DD"]["SDATA"]["total_mass_pdf"],label="Total")
for comp in ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].components:
    ax[1].plot(Bmass_vec,pdf_values_prefit["CB2DPI_D2KSPIPI_DD"]["SDATA"][comp[0]]*scale,linestyle="--",label=components_tex[comp[0]]+"\n"+comp[1])
ax[1].legend(title=ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].channel.tex)
ax[1].set_xlim(5080,5800)
ax[1].set_xlabel(v_what_to_plot.tex)
ax[1].set_xlabel(v_what_to_plot.tex)
plt.tight_layout()
plt.savefig("prefit_"+what_to_plot+'.png')
plt.savefig("prefit_"+what_to_plot+'.pdf')
plt.close()

x = list(start_values.values())
m = iminuit.Minuit(nll, x, name=parameters_to_fit)
m.limits = list(limit_values.values())
mg = m.migrad()
# Minuit(least_squares_np, (5, 5), name=("a", "b"))
# m.limits = [(0, None), (0, 10)]

print(mg)
means  = mg.values
errors = mg.errors
hesse  = mg.hesse()
cov    = hesse.covariance
corr   = cov.correlation()
## i have to loop over the entries if this dict to set the pandas df myself
corr_array = np.zeros(len(parameters_to_fit)*len(parameters_to_fit)).reshape(len(parameters_to_fit),len(parameters_to_fit))
for i in range(len(parameters_to_fit)):
    corr_array[i] = corr[parameters_to_fit[i]]
    pass
pd_cov = pd.DataFrame(corr_array,index=parameters_to_fit,columns=parameters_to_fit)
for i in range(len(means)):
    print(f'{parameters_to_fit[i]:<50}', ": ", means[i] ," +- ", errors[i])
    pass
# print("Means   ", means)
# print("Errors  ", errors)
vec_results = dict(zip(parameters_to_fit,means))
print(json.dumps(vec_results,indent=4))
with open(f'simfit_output.json', 'w') as f:
    json.dump(vec_results,f,indent=4)
    pass


# columns_to_keep = ["Wilson_C9_tau", "Wilson_C9", "Wilson_C10", "J_psi_phase", "psi_2S_phase"]
# pull_df_corr = pull_df[columns_to_keep].corr()   # correlation matrix in pandas dataframe
import seaborn as sns
fig = plt.figure(figsize=(10, 8))  
sns.heatmap(pd_cov, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig("covariance.png")
# pages.append(fig)

variables_post_fit = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].get_mass_pdf_functions(list_vardict, params=means, shared_parameters=shared_parameters, constrained_parameters=constrained_parameters)


### data
pdf_values_postfit = {}
for channel in list_channels:
    i_c = INDEX_CHANNEL_TO_VARDICT[channel]
    pdf_values_postfit[channel] = {}
    pdf_values_postfit[channel]["SDATA"] = ntuples[channel]["SDATA"].pdf_values_draw(tf_Bmass_vec,variables_post_fit[i_c])
    # pdf_values_postfit[channel]["SDATA"] = ntuples[channel]["SDATA"].pdf_values_draw(tf_Bmass_vec,VARDICT["SDATA"][i_c])
    pass

scale = 5
fig, ax = plt.subplots(1,2)
plt.suptitle(ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].source.tex + " " + ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].year.tex)
##### DK
what_to_plot = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
data_to_plot = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].uproot_data.query( "("+what_to_plot+" < 5800) & ("+what_to_plot+" > 5080)" )[v_what_to_plot.name]
mplhep.histplot(np.histogram(data_to_plot,bins=int((5800-5080)/5),range=[5080,5800]), label="Run 2 data", ax=ax[0], **kwargs_data)
ax[0].plot(Bmass_vec,scale*pdf_values_postfit["CB2DK_D2KSPIPI_DD"]["SDATA"]["total_mass_pdf"],label="Total")
for comp in ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].components:
    ax[0].plot(Bmass_vec,pdf_values_postfit["CB2DK_D2KSPIPI_DD"]["SDATA"][comp[0]]*scale,linestyle="--",label=components_tex[comp[0]]+"\n"+comp[1])
ax[0].legend(title="Channel: "+ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].channel.tex)
ax[0].set_xlim(5080,5800)
ax[0].set_xlabel(v_what_to_plot.tex)
ax[0].set_xlabel(v_what_to_plot.tex)
##### DPI
what_to_plot = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].variable_to_fit
v_what_to_plot = DICT_VARIABLES_TEX[what_to_plot]
data_to_plot = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].uproot_data.query( "("+what_to_plot+" < 5800) & ("+what_to_plot+" > 5080)" )[v_what_to_plot.name]
mplhep.histplot(np.histogram(data_to_plot,bins=int((5800-5080)/5),range=[5080,5800]), label="Run 2 data", ax=ax[1], **kwargs_data)
ax[1].plot(Bmass_vec,scale*pdf_values_postfit["CB2DPI_D2KSPIPI_DD"]["SDATA"]["total_mass_pdf"],label="Total")
for comp in ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].components:
    ax[1].plot(Bmass_vec,pdf_values_postfit["CB2DPI_D2KSPIPI_DD"]["SDATA"][comp[0]]*scale,linestyle="--",label=components_tex[comp[0]]+"\n"+comp[1])
ax[1].legend(title="Channel: "+ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"].channel.tex)
ax[1].set_xlim(5080,5800)
ax[1].set_xlabel(v_what_to_plot.tex)
ax[1].set_xlabel(v_what_to_plot.tex)
plt.tight_layout()
plt.savefig("postfit_"+what_to_plot+'.png')
plt.savefig("postfit_"+what_to_plot+'.pdf')
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
