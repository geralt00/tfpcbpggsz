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
import seaborn as sns
import matplotlib as mpl
import gc

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
rcParams["figure.figsize"] = (8, 6)
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

from tfpcbpggsz.core import *
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.amp_masses import *

Kspipi = PyD0ToKspipi2018()
Kspipi.init()

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
basic_list_var = ["Bu_ID", "zp_p", "zm_pp", "m_Kspip", "m_Kspim"]
for particle in ["Ks", "h1", "h2"]:
    basic_list_var.append(f"{particle}_PE")
    basic_list_var.append(f"{particle}_PX")
    basic_list_var.append(f"{particle}_PY")
    basic_list_var.append(f"{particle}_PZ")
    pass


BDT_cut = 0.4
min_mass = 5200
# min_mass = 5080
max_mass = 5800
Bmass_vec = np.arange(min_mass, max_mass, 1)
tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)
####
min_dalitz = 0.4
max_dalitz = 3.0
nbins=100
m_Kspip_range = [min_dalitz, max_dalitz]
Dalitz_mass_vec = np.arange(min_dalitz, max_dalitz, (max_dalitz-min_dalitz)/1000. )
tf_Dalitz_mass_vec = tf.cast(Dalitz_mass_vec, tf.float64)
####
BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
str_BDT_cut = "(BDT_output > "+str(BDT_cut)+")"
#### scaling for comparison histogram vs pdfs
scale = (max_mass - min_mass) / float(nbins)
projection_scaling = (m_Kspip_range[1]-m_Kspip_range[0]) / float(nbins)


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
        ntuples[channel][source].initialise_fit(components)
        cut[channel][source] = str_BDT_cut + " & " + ntuples[channel][source].dict_final_cuts["Bach_PID"] + " & (" + ntuples[channel][source].variable_to_fit + f" < {max_mass} ) & (" + ntuples[channel][source].variable_to_fit + f" > {min_mass} )"
        paths[channel][source] = ntuples[channel][source].final_cuts_paths
        list_var[channel][source] = [ntuples[channel][source].variable_to_fit] + basic_list_var
        ntuples[channel][source].store_events(paths[channel][source],list_var[channel][source],cut[channel][source], Kspipi)
        BDT_cut_efficiency[channel][source] = len(ntuples[channel][source].uproot_data[list_var[channel][source][0]]) / fin_cuts_eff[channel][source]["YRUN2"]["MagAll"]["selected_events"]
        total_eff[channel][source] = pre_cuts_eff[channel][source]["YRUN2"]["MagAll"]["efficiency"]*fin_cuts_eff[channel][source]["YRUN2"]["MagAll"]["efficiency"]*BDT_cut_efficiency[channel][source]
        # ntuples[channel][source].store_amplitudes(Kspipi)
        pass
    pass

print(json.dumps(total_eff,indent=4))



######## initialise data ntuples
for channel in list_channels:
    # i_s = INDEX_SOURCE_TO_VARDICT["SDATA"]
    i_c = INDEX_CHANNEL_TO_VARDICT[channel]
    ntuples[channel]["SDATA"] = Ntuple("SDATA",channel,"YRUN2", "MagAll")
    ntuples[channel]["SDATA"].initialise_fit(components)
    print(ntuples[channel]["SDATA"])
    print(ntuples[channel]["SDATA"].components)
    print(ntuples[channel]["SDATA"].variable_to_fit)
    cut[channel]["SDATA"] = str_BDT_cut + " & " + ntuples[channel]["SDATA"].dict_final_cuts["Bach_PID"] + " & (" + ntuples[channel][source].variable_to_fit + f" < {max_mass} ) & (" + ntuples[channel][source].variable_to_fit + f" > {min_mass} )"
    paths[channel]["SDATA"] = ntuples[channel]["SDATA"].final_cuts_paths
    list_var[channel]["SDATA"] = [ntuples[channel]["SDATA"].variable_to_fit] + basic_list_var
    ntuples[channel]["SDATA"].store_events(paths[channel]["SDATA"],list_var[channel]["SDATA"],cut[channel]["SDATA"], Kspipi)
    # ntuples[channel]["SDATA"].store_amplitudes(Kspipi)
    pass


######## define starting values of the variables in the fit
start_values = {
    "ratio_BR_DK_to_Dpi" :                         0.07,
    "signal_mean_DK":                              varDict['signal_mean']+50,
    "signal_mean_Dpi":                             varDict['signal_mean']+50,
    "signal_width_DK_DD":                          varDict['sigma_dk_DD']+50,
    "signal_width_Dpi_DD":                         varDict['sigma_dk_DD']+50,
    # "Dst0K_D0pi0_Kspipi_yield_DD":                 3000.,
    # "DstpK_D0pip_Kspipi_yield_DD":                 3000.,
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": 3000.,
    # "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": 3000.,
    # "Dst0K_D0gamma_Kspipi_yield_DD"              : 3000.,
    "Combinatorial_yield_DK_DD_Bplus" :                   1200.,
    "Combinatorial_yield_DK_DD_Bminus":                   12000.,
    "Dpi_Kspipi_yield_DD_Bplus" :                         12000.,
    "Dpi_Kspipi_yield_DD_Bminus":                         120000.,
    # "Dst0pi_D0pi0_Kspipi_yield_DD":                7000.,
    # "Dst0pi_D0gamma_Kspipi_yield_DD":              7000.,
    "Combinatorial_yield_DPI_DD_Bplus" :                  2000.,                     
    "Combinatorial_yield_DPI_DD_Bminus":                  20000.,                     
    "Combinatorial_c_DK_DD":                       -0.003,
    "xplus"                                        : 1.00,
    "yplus"                                        : 0.00,
    "xminus"                                       : 1.00,
    "yminus"                                       : 0.00,
    # "xxi"                                          : 0.00,
    # "yxi"                                          : 0.00,
}

####### define limits 
limit_values = {
    "ratio_BR_DK_to_Dpi" :                         [0.0001,1],
    "signal_mean_DK":                              [5080,5800],
    "signal_mean_Dpi":                             [5080,5800],
    "signal_width_DK_DD":                          [1,100],
    "signal_width_Dpi_DD":                         [1,100],
    # "Dst0K_D0pi0_Kspipi_yield_DD":                 [0, 20000],
    # "DstpK_D0pip_Kspipi_yield_DD":                 [0, 20000],
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": [0, 20000],
    # "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": [0, 20000],
    # "Dst0K_D0gamma_Kspipi_yield_DD"              : [0, 20000],
    "Combinatorial_yield_DK_DD_Bplus" :                   [0, 50000],
    "Combinatorial_yield_DK_DD_Bminus":                   [0, 50000],
    "Dpi_Kspipi_yield_DD_Bplus" :                         [0,400000],
    "Dpi_Kspipi_yield_DD_BMinus":                         [0,400000],
    # "Dst0pi_D0pi0_Kspipi_yield_DD":                [0,400000],
    # "Dst0pi_D0gamma_Kspipi_yield_DD":              [0,400000],
    "Combinatorial_yield_DPI_DD_Bplus" :                  [0,400000],
    "Combinatorial_yield_DPI_DD_Bminus":                  [0,400000],
    "Combinatorial_c_DK_DD":                       [-0.01,-0.0005],
    "xplus"                                        : [-1., 1.],
    "yplus"                                        : [-1., 1.],
    "xminus"                                       : [-1., 1.],
    "yminus"                                       : [-1., 1.],
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
    # "Dst0K_D0pi0_Kspipi_yield_DD": [
    #     ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "yield_Bplus"],
    # ],
    # "DstpK_D0pip_Kspipi_yield_DD": [
    #     ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "yield_Bplus"],
    # ],
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": [
    #     ["CB2DK_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi_misID_PartReco", "yield"],
    # ],
    # "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": [
    #     ["CB2DK_D2KSPIPI_DD", "Dstppi_D0pip_Kspipi_misID_PartReco", "yield_Bplus"],
    # ],
    # "Dst0K_D0gamma_Kspipi_yield_DD": [
    #     ["CB2DK_D2KSPIPI_DD", "Dst0K_D0gamma_Kspipi", "yield_Bplus"],
    # ],
    # "DK_Kspipi_yield_DD": [ ## this should cause an error
    #     ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "yield"  ],
    # ],
    "Combinatorial_yield_DK_DD_Bplus": [
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "yield_Bplus"  ],
    ],
    "Combinatorial_yield_DK_DD_Bminus": [
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "yield_Bminus"  ],
    ],
    "Dpi_Kspipi_yield_DD_Bplus": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield_Bplus"  ],
    ],
    "Dpi_Kspipi_yield_DD_Bminus": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield_Bminus"  ],
    ],
    # "Dst0pi_D0pi0_Kspipi_yield_DD": [ 
    #     ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi", "yield_Bplus"  ],
    # ],
    # "Dst0pi_D0gamma_Kspipi_yield_DD": [ 
    #     ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0gamma_Kspipi", "yield_Bplus"  ],
    # ],
    "Combinatorial_yield_DPI_DD_Bplus": [
        ["CB2DPI_D2KSPIPI_DD", "Combinatorial", "yield_Bplus"  ],
    ],
    "Combinatorial_yield_DPI_DD_Bminus": [
        ["CB2DPI_D2KSPIPI_DD", "Combinatorial", "yield_Bminus"  ],
    ],
    "Combinatorial_c_DK_DD": [ # Exponential 11
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "c"],
    ],
    "xplus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "xplus"],
    ],
    "yplus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "yplus"],
    ],
    "xminus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "xminus"],
    ],
    "yminus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "yminus"],
    ],
}


############################ ALL KIND OF RATIOS OF EFFICIENCIES
############# compute ratio of efficiencies etc to constrain parameters
# constraining the DK yields to the Dpi one
ratio_DK_to_Dpi = {}
ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] = total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]/total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
# constraining the misID Dpi to its non-misID counterpart
ratio_Dpi_misID_to_Dpi = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] = total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]/total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0pi_KSpipi"]
# constraining the misID DK to its non-misID counterpart
ratio_DK_misID_to_DK = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] = total_eff["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]/total_eff["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
# constraining the misID DK to its non-misID counterpart - Part Reco
        
######### FIXED CONSTRAINTS
# (only implemented with multiplicative factor)
dict_constrained_parameters = [
    ## constrain DK from Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi"       , "yield_Bplus" ], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield_Bplus",
                                                            [ "SHARED_THROUGH_CHANNELS", "parameters", "ratio_BR_DK_to_Dpi", ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ] ],
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi"       , "yield_Bminus"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield_Bminus",
                                                            [ "SHARED_THROUGH_CHANNELS", "parameters", "ratio_BR_DK_to_Dpi", ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ] ],
    ## constrain misID in DK from goodID in Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "yield_Bplus"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield_Bplus", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] ] ],
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "yield_Bminus"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "yield_Bminus", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] ] ],
    ## constrain misID DPI from goodID in DK
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "yield_Bplus"], ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "yield_Bplus", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] ] ],
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "yield_Bminus"], ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "yield_Bminus", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] ] ],
]
print(json.dumps(dict_constrained_parameters,indent=4))

####################### GAUSSIAN CONSTRAINTS
ratio_Dst0K_to_DstpK = Measurement(1,0.1)
######### define which real variable is constrained from which other var
# (only implemented with multiplicative factor)
dict_gaussian_constraints = [
        # [ ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "yield_Bplus"], ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "yield_Bplus", ratio_Dst0K_to_DstpK.value, ratio_Dst0K_to_DstpK.error] ],
]
print(json.dumps(dict_gaussian_constraints,indent=4))


#### This is how we can change some initial parameters
fixed_variables = dict(**VARDICT["SDATA"])
fixed_variables["SHARED_THROUGH_CHANNELS"] = {
    "parameters":
    {
        "ratio_BR_DK_to_Dpi": 0.07,
        "xplus"             : 1.00,
        "yplus"             : 0.00,
        "xminus"            : 1.00,
        "yminus"            : 0.00,
        "xxi"               : 0.00,
        "yxi"               : 0.00,
    }
}

#### initialisation of the fit:
# - changes dictionaries to lists 
NLL = NLLComputation(
    start_values,
    limit_values,
    dict_shared_parameters,
    dict_constrained_parameters,
    dict_gaussian_constraints,
    list_channels,
    fixed_variables,
    ntuples,
    components
)
parameters_to_fit = NLL.parameters_to_fit

preFit_list_variables = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].get_list_variables(
    NLL.fixed_variables,
    params=list(start_values.values()),
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)


############
#### normalisation
ntuple_normalisation = ntuples["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
# ntuple_normalisation = ntuples["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
norm_ampD0    = ntuple_normalisation.AmpD0
norm_ampD0bar = ntuple_normalisation.AmpD0bar
norm_zp_p     = ntuple_normalisation.zp_p
norm_zm_pp    = ntuple_normalisation.zm_pp

######### data tuple
ntuple_test          = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"]
# ntuple_test          = ntuples["CB2DPI_D2KSPIPI_DD"]["SDATA"]
# this one defines the dalitz and mass pdfs, and tores the normalisation events
ntuple_test.define_dalitz_pdfs(norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp)
print(ntuple_test.dalitz_pdfs.keys()  )
print(ntuple_test.dalitz_pdfs["Bplus"])
print(ntuple_test.mass_pdfs.keys()    )
print(ntuple_test.mass_pdfs["Bplus"]  )
# ntuple_test.get_list_variables(NLL.fixed_variables)
# this bit runs get_mass_pdf and get_dalitz_pdf
# for all components and B signs

####################### some tools for plotting: 2D grid of invariant Dalitz masses
Dalitz_Kspip_mat, Dalitz_Kspim_mat = np.meshgrid(Dalitz_mass_vec,Dalitz_mass_vec)
## SRD variables
RD_var = func_var_rotated(Dalitz_Kspip_mat, Dalitz_Kspim_mat, QMI_zpmax_Kspi, QMI_zpmin_Kspi, QMI_zmmax_Kspi, QMI_zmmin_Kspi)
SRD_var = func_var_rotated_stretched(RD_var)
zp_p  = SRD_var[0]
zm_pp = SRD_var[1]
# compute the amplitudes of this 2D mesh
ampD0    = np.zeros(zp_p.shape , dtype=complex)
ampD0bar = np.zeros(zm_pp.shape, dtype=complex)
for row in range(zp_p.shape[0]):
    if (row%100==0): print("Processed ", row)
    for col in range(zp_p.shape[1]):
        tmp_amps = Kspipi.get_amp(
            zp_p[row][col] ,
            zm_pp[row][col]
        )
        ampD0[row][col]    = tmp_amps[0]
        ampD0bar[row][col] = tmp_amps[1]
        pass
    pass


################ plot the amplitudes to check
cs = plt.contourf(Dalitz_Kspip_mat, Dalitz_Kspim_mat, np.abs(ampD0), levels=100)
# , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
plt.xlabel("Kspip")
plt.ylabel("Kspim")
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()
plt.savefig("abs_ampD0.png")
plt.close("all")

cs = plt.contourf(zp_p, zm_pp, np.abs(ampD0), levels=100)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel(r"$z_{+}^{\prime}$")
plt.ylabel(r"$z_{-}^{\prime\prime}$")
# , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()
plt.savefig("SRD_abs_ampD0.png")
plt.close("all")

cs = plt.contourf(Dalitz_Kspip_mat, Dalitz_Kspim_mat, np.abs(ampD0bar), levels=100)
# , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
plt.xlabel("Kspip")
plt.ylabel("Kspim")
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()
plt.savefig("abs_ampD0bar.png")
plt.close("all")


def draw_dalitz_pdf(ntuple, ampD0, ampD0bar, zp_p, zm_pp, variables=None):
    """
    this function is only for plotting purposes. It computes the pdfs 
    for each component and the sum.
    """
    if (variables == None):
        pass
    else:
        ntuple.initialise_fixed_pdfs(variables)
        pass
    pdfs_values = {}
    ## loop over the B signs
    for Bsign in BSIGNS.keys():
        pdfs_values[Bsign] = {}
        pdfs_values[Bsign]["total_pdf"] = tf.cast(np.zeros(
            np.shape(ampD0)
        ).astype(np.float64), tf.float64)
        print(Bsign)
        ## loop over the components of the ntuple
        for i in range(len(ntuple.dalitz_pdfs[Bsign])):
            comp_pdf = ntuple.dalitz_pdfs[Bsign][i]
            isSignalDK   = (comp_pdf.component in SIGNAL_COMPONENTS_DK )
            isSignalDPI  = (comp_pdf.component in SIGNAL_COMPONENTS_DPI)
            if ( (isSignalDK == True) or (isSignalDPI == True)):
                ## if comp is DK or DPI, the pdf is a function of the amplitudes
                tmp_pdf_values = comp_pdf.pdf(ampD0, ampD0bar)
            else:
                ## otherwise it's a function of the SRD variables
                print(comp_pdf.component)
                tmp_pdf_values = comp_pdf.pdf(zp_p, zm_pp)
                print(tmp_pdf_values)
                pass
            #### multiply each pdf by its yield
            index_yields  = INDEX_YIELDS[Bsign]
            comp_yield    = variables[ntuple.i_c][i][index_yields]
            tmp_pdf_values = tmp_pdf_values*comp_yield
            pdfs_values[Bsign][comp_pdf.component]     = tmp_pdf_values
            pdfs_values[Bsign]["total_pdf"]           += tmp_pdf_values
            pass # loop comps
        pass # loop signs
    return pdfs_values


#### compute the pdfs
dalitz_pdfs_values = draw_dalitz_pdf(
    ntuple_test,
    ampD0   ,
    ampD0bar,
    zp_p    ,
    zm_pp   ,
    preFit_list_variables
)
print(dalitz_pdfs_values)

### plotting these pdfs
for Bsign in dalitz_pdfs_values.keys():
    for comp in dalitz_pdfs_values[Bsign].keys():
        try:
            cs = plt.contourf(
                Dalitz_Kspip_mat,
                Dalitz_Kspim_mat,
                dalitz_pdfs_values[Bsign][comp],
                levels=100) # ,
                # norm="log")
        except TypeError:
            continue
        # , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
        plt.xlabel("Kspip")
        plt.ylabel("Kspim")
        cbar = plt.colorbar(cs)
        # cs.cmap.set_over('red')
        # cs.cmap.set_under('white')
        # cs.changed()
        plt.tight_layout()
        plt.savefig(f"preFit_plots_DK/{Bsign}_{comp}.png")
        plt.close("all")
        pass
    pass


def draw_projection_pdf(ntuple, pdfs_values):
    """
    This function integrates separately over the two dimensions
    to get both projections of the Dalitz pdfs.
    """
    Kspip_projection = {}
    Kspim_projection = {}
    for Bsign in BSIGNS.keys():
        Kspip_projection[Bsign] = {}
        Kspim_projection[Bsign] = {}
        #### the two indices correspond to x and y axis
        # i.e. z+ and z- or m(Kspip) and m(Kspim)
        Kspip_projection[Bsign]["total_pdf"] = np.zeros(
            pdfs_values[Bsign]["total_pdf"].shape[0]
        ).astype(np.float64)
        Kspim_projection[Bsign]["total_pdf"] = np.zeros(
            pdfs_values[Bsign]["total_pdf"].shape[0]
        ).astype(np.float64)
        print(Bsign)
        for i in range(len(ntuple.dalitz_pdfs[Bsign])):
            comp_pdf = ntuple.dalitz_pdfs[Bsign][i]
            print(comp_pdf.component)
            if(pdfs_values[Bsign][comp_pdf.component].shape == []):
                print("This component is empty ---- skipping")
                continue
            Kspip_projection[Bsign][comp_pdf.component] = np.zeros(
                pdfs_values[Bsign]["total_pdf"].shape[0]
            ).astype(np.float64)
            Kspim_projection[Bsign][comp_pdf.component] = np.zeros(
                pdfs_values[Bsign]["total_pdf"].shape[0]
            ).astype(np.float64)
            tmp_numpy_pdfs = pdfs_values[Bsign][comp_pdf.component].numpy()
            for i_Kspi in range(Kspip_projection[Bsign][comp_pdf.component].shape[0]):
                Kspip_projection[Bsign][comp_pdf.component][i_Kspi] = float(tf.reduce_mean(
                    tmp_numpy_pdfs.transpose()[i_Kspi]
                ))
                Kspip_projection[Bsign]["total_pdf"][i_Kspi] += float(tf.reduce_mean(
                    tmp_numpy_pdfs.transpose()[i_Kspi]
                ))
                #### Kspim
                Kspim_projection[Bsign][comp_pdf.component][i_Kspi] = float(tf.reduce_mean(
                    tmp_numpy_pdfs[i_Kspi]
                ))
                Kspim_projection[Bsign]["total_pdf"][i_Kspi] += float(tf.reduce_mean(
                    tmp_numpy_pdfs[i_Kspi]
                ))
                # if (Kspip_projection[Bsign][comp_pdf.component][i_Kspip] > 0):
                #     print(Kspip_projection[Bsign][comp_pdf.component][i_Kspip])
                pass # loop axes
            pass # loop comps
        pass # loop signs
    return Kspip_projection, Kspim_projection
########## get the projections
Kspip_projection, Kspim_projection = draw_projection_pdf(ntuple_test, dalitz_pdfs_values)


data = {
    "Bplus" : ntuple_test.Bplus_data ,
    "Bminus": ntuple_test.Bminus_data,
}

#####
for Bsign in BSIGNS.keys():
    tmp_data = data[Bsign]
    plt.plot(
        Dalitz_mass_vec,
        Kspip_projection[Bsign]["total_pdf"]*projection_scaling,
        label="total_pdf"
    )
    for i in range(len(ntuple_test.dalitz_pdfs[Bsign])):
        comp_pdf = ntuple_test.dalitz_pdfs[Bsign][i]
        index_yields  = INDEX_YIELDS[Bsign]
        if (preFit_list_variables[ntuple_test.i_c][i][index_yields] == 0): continue
        plt.fill_between(
            Dalitz_mass_vec,
            Kspip_projection[Bsign][comp_pdf.component]*projection_scaling,
            alpha=0.5,
            label=comp_pdf.component
        )
        pass
    # plt.fill_between(
    #     Dalitz_mass_vec,
    #     Kspip_projection[Bsign]["Dpi_Kspipi_misID"]*projection_scaling,
    #     alpha=0.5,
    #     label="Dpi_Kspipi_misID"
    # )
    mplhep.histplot(
        np.histogram(tmp_data["m_Kspip"], bins=100),
        label=ntuple_test.tex
    )
    plt.legend(title=Bsign)
    plt.xlabel("Kspip")
    plt.ylabel(f"Events / ({round(projection_scaling*1000)} MeV)")
    plt.tight_layout()
    plt.savefig(f"preFit_plots_DK/{Bsign}_total_pdf_Kspip_projection.png")
    plt.close("all")
    plt.plot(
        Dalitz_mass_vec,
        Kspim_projection[Bsign]["total_pdf"]*projection_scaling,
        label="total_pdf"
    )
    for i in range(len(ntuple_test.dalitz_pdfs[Bsign])):
        comp_pdf = ntuple_test.dalitz_pdfs[Bsign][i]
        index_yields  = INDEX_YIELDS[Bsign]
        if (preFit_list_variables[ntuple_test.i_c][i][index_yields] == 0): continue
        plt.fill_between(
            Dalitz_mass_vec,
            Kspim_projection[Bsign][comp_pdf.component]*projection_scaling,
            alpha=0.5,
            label=comp_pdf.component
        )
        pass
    mplhep.histplot(
        np.histogram(tmp_data["m_Kspim"], bins=100),
        label=ntuple_test.tex
    )
    plt.legend(title=Bsign)
    plt.xlabel("Kspim")
    plt.ylabel(f"Events / ({round(projection_scaling*1000)} MeV)")
    plt.tight_layout()
    plt.savefig(f"preFit_plots_DK/{Bsign}_total_pdf_Kspim_projection.png")
    plt.close("all")
    pass


####### and now the mass pdfs
mass_pdfs_values = ntuple_test.draw_mass_pdfs(
    tf_Bmass_vec,
    preFit_list_variables
)

for Bsign in BSIGNS.keys():
    tmp_data = data[Bsign]
    mplhep.histplot(
        np.histogram(tmp_data[ntuple_test.variable_to_fit], bins=100),
        label=ntuple_test.tex
    )
    plt.plot(
        Bmass_vec,
        scale*mass_pdfs_values[Bsign]["total_mass_pdf"],
        label="Total"
    )
    for comp in ntuple_test.components:
        plt.plot(
            Bmass_vec,
            mass_pdfs_values[Bsign][comp[0]]*scale,linestyle="--",
            label=components_tex[comp[0]]+"\n"+comp[1]
        )
        pass
    plt.xlabel("Constrained $D\pi$ mass")
    plt.ylabel(f"Events / ({round(scale*nbins)} MeV)")
    plt.tight_layout()
    plt.savefig(f"preFit_plots_DK/{Bsign}_mass_distribution.png")
    plt.close("all")
    pass


@tf.function
def nll(x):
    return NLL.get_total_nll(x) # , tensor_to_fit)

ntuple_normalisation_DK_DD  = ntuples["CB2DK_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
ntuple_normalisation_DPI_DD = ntuples["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
dict_norm_ampD0    = {
    "CB2DK_D2KSPIPI_DD" :  ntuple_normalisation_DK_DD.AmpD0,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.AmpD0,
}
dict_norm_ampD0bar = {
    "CB2DK_D2KSPIPI_DD" :  ntuple_normalisation_DK_DD.AmpD0bar,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.AmpD0bar,
}
dict_norm_zp_p = {
    "CB2DK_D2KSPIPI_DD" :  ntuple_normalisation_DK_DD.zp_p,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.zp_p,
}
dict_norm_zm_pp = {
    "CB2DK_D2KSPIPI_DD" :  ntuple_normalisation_DK_DD.zm_pp,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.zm_pp,
}


NLL.initialise_dalitz_fit(
    dict_norm_ampD0,
    dict_norm_ampD0bar,
    dict_norm_zp_p,
    dict_norm_zm_pp
)

x = tf.cast(list(start_values.values()),tf.float64)
print("start computing")
print("test nll(x) : ", nll(x))


m = iminuit.Minuit(nll, x, name=parameters_to_fit)
m.limits = list(limit_values.values())
mg = m.migrad()


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
fig = plt.figure(figsize=(10, 8))  
sns.heatmap(pd_cov, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig("covariance.png")
plt.close("all")
# pages.append(fig)

postFit_list_variables = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].get_list_variables(
    NLL.fixed_variables,
    params=means,
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)

#### compute the pdfs
postFit_dalitz_pdfs_values = draw_dalitz_pdf(
    ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"],
    ampD0   ,
    ampD0bar,
    zp_p    ,
    zm_pp   ,
    variables=postFit_list_variables
)

print(postFit_dalitz_pdfs_values)

### plotting these pdfs
for Bsign in postFit_dalitz_pdfs_values.keys():
    for comp in postFit_dalitz_pdfs_values[Bsign].keys():
        try:
            cs = plt.contourf(
                Dalitz_Kspip_mat,
                Dalitz_Kspim_mat,
                postFit_dalitz_pdfs_values[Bsign][comp],
                levels=100) # ,
                # norm="log")
        except TypeError:
            continue
        # , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
        plt.xlabel("Kspip")
        plt.ylabel("Kspim")
        cbar = plt.colorbar(cs)
        # cs.cmap.set_over('red')
        # cs.cmap.set_under('white')
        # cs.changed()
        plt.tight_layout()
        plt.savefig(f"postFit_plots_DK/{Bsign}_{comp}.png")
        plt.close("all")
        pass
    pass

Kspip_projection, Kspim_projection = draw_projection_pdf(
    ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"],
    postFit_dalitz_pdfs_values
)

##### dalitz projection
data = {
    "Bplus" : ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].Bplus_data ,
    "Bminus": ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].Bminus_data,
}

#####
for Bsign in BSIGNS.keys():
    tmp_data = data[Bsign]
    plt.plot(
        Dalitz_mass_vec,
        Kspip_projection[Bsign]["total_pdf"]*projection_scaling,
        label="total_pdf"
    )
    for i in range(len(ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].dalitz_pdfs[Bsign])):
        comp_pdf = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].dalitz_pdfs[Bsign][i]
        index_yields  = INDEX_YIELDS[Bsign]
        if (NLL.fixed_variables[ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].i_c][i][index_yields] == 0): continue
        plt.fill_between(
            Dalitz_mass_vec,
            Kspip_projection[Bsign][comp_pdf.component]*projection_scaling,
            alpha=0.5,
            label=comp_pdf.component
        )
        pass
    # plt.fill_between(
    #     Dalitz_mass_vec,
    #     Kspip_projection[Bsign]["Dpi_Kspipi_misID"]*projection_scaling,
    #     alpha=0.5,
    #     label="Dpi_Kspipi_misID"
    # )
    mplhep.histplot(
        np.histogram(tmp_data["m_Kspip"], bins=100),
        label=ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].tex
    )
    plt.legend(title=Bsign)
    plt.xlabel("Kspip")
    plt.ylabel(f"Events / ({round(projection_scaling*1000)} MeV)")
    plt.tight_layout()
    plt.savefig(f"postFit_plots_DK/{Bsign}_total_pdf_Kspip_projection.png")
    plt.close("all")
    plt.plot(
        Dalitz_mass_vec,
        Kspim_projection[Bsign]["total_pdf"]*projection_scaling,
        label="total_pdf"
    )
    for i in range(len(ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].dalitz_pdfs[Bsign])):
        comp_pdf = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].dalitz_pdfs[Bsign][i]
        index_yields  = INDEX_YIELDS[Bsign]
        if (NLL.fixed_variables[ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].i_c][i][index_yields] == 0): continue
        plt.fill_between(
            Dalitz_mass_vec,
            Kspim_projection[Bsign][comp_pdf.component]*projection_scaling,
            alpha=0.5,
            label=comp_pdf.component
        )
        pass
    mplhep.histplot(
        np.histogram(tmp_data["m_Kspim"], bins=100),
        label=ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].tex
    )
    plt.legend(title=Bsign)
    plt.xlabel("Kspim")
    plt.ylabel(f"Events / ({round(projection_scaling*1000)} MeV)")
    plt.tight_layout()
    plt.savefig(f"postFit_plots_DK/{Bsign}_total_pdf_Kspim_projection.png")
    plt.close("all")
    pass



####### and now the mass pdfs
mass_pdfs_values = ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].draw_mass_pdfs(
    tf_Bmass_vec,
    postFit_list_variables
)

for Bsign in BSIGNS.keys():
    tmp_data = data[Bsign]
    mplhep.histplot(
        np.histogram(tmp_data[ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].variable_to_fit], bins=100),
        label=ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].tex
    )
    plt.plot(
        Bmass_vec,
        scale*mass_pdfs_values[Bsign]["total_mass_pdf"],
        label="Total"
    )
    for comp in ntuples["CB2DK_D2KSPIPI_DD"]["SDATA"].components:
        plt.plot(
            Bmass_vec,
            mass_pdfs_values[Bsign][comp[0]]*scale,linestyle="--",
            label=components_tex[comp[0]]+"\n"+comp[1]
        )
        pass
    plt.xlabel("Constrained $D\pi$ mass")
    plt.ylabel(f"Events / ({round(scale*nbins)} MeV)")
    plt.tight_layout()
    plt.savefig(f"postFit_plots_DK/{Bsign}_mass_distribution.png")
    plt.close("all")
    pass
