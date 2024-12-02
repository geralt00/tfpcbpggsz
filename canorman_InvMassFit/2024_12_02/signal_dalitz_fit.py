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
from tfpcbpggsz.Includes.VARDICT_DALITZ import VARDICT, varDict

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
num_mass_values = 100.
Dalitz_mass_vec = np.arange(min_dalitz, max_dalitz, (max_dalitz-min_dalitz)/num_mass_values )
tf_Dalitz_mass_vec = tf.cast(Dalitz_mass_vec, tf.float64)
####
BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
str_BDT_cut = "(BDT_output > "+str(BDT_cut)+")"
#### scaling for comparison histogram vs pdfs
mass_scaling   = (max_mass - min_mass) / float(nbins)
dalitz_scaling = (m_Kspip_range[1]-m_Kspip_range[0]) / float(nbins)


################ define the components in each channel and source
components = {
    "SDATA":
    {
        "CB2DK_D2KSPIPI_DD": [ 
            ["DK_Kspipi", "Cruijff+Gaussian", "B2Dh_D2Kspipi"],
            ["Dpi_Kspipi_misID", "SumCBShape", "B2Dh_D2Kspipi"],
            ["Dst0K_D0pi0_Kspipi", "HORNSdini", "Legendre_2_2"],
            ["DstpK_D0pip_Kspipi", "HORNSdini", "Legendre_2_2"],
            ["Dst0pi_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID", "Legendre_2_2"],
            ["Dstppi_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID", "Legendre_2_2"],
            ["Dst0K_D0gamma_Kspipi", "HILLdini", "Legendre_2_2"],
            ["Dst0pi_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID", "Legendre_2_2"],
            ["DKpi_Kspipi", "HORNSdini+Gaussian", "Legendre_2_2"],
            ["Dpipi_Kspipi_misID_PartReco", "HORNSdini_misID", "Legendre_2_2"],
            ["Bs2DKpi_Kspipi_PartReco", "HORNSdini", "Legendre_2_2"],
            ["Combinatorial", "Exponential", "Legendre_2_2"],
        ],
        "CB2DPI_D2KSPIPI_DD": [ 
            ["Dpi_Kspipi", "Cruijff+Gaussian", "B2Dh_D2Kspipi"],
            ["DK_Kspipi_misID", "CBShape", "B2Dh_D2Kspipi"],
            ["Dst0pi_D0pi0_Kspipi", "HORNSdini", "Legendre_2_2"],
            ["Dstppi_D0pip_Kspipi", "HORNSdini", "Legendre_2_2"],
            ["Dst0K_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID", "Legendre_2_2"],
            ["DstpK_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID", "Legendre_2_2"],
            ["Dst0pi_D0gamma_Kspipi", "HILLdini", "Legendre_2_2"],
            ["Dst0K_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID", "Legendre_2_2"],
            ["Dpipi_Kspipi", "HORNSdini+HORNSdini", "Legendre_2_2"],
            ["Combinatorial", "Exponential", "Legendre_2_2"],
        ]
    },
    "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow":
    {
        "CB2DK_D2KSPIPI_DD": [
            ["DK_Kspipi_PHSP", "Cruijff+Gaussian", "Legendre_2_2"],
        ],
        "CB2DPI_D2KSPIPI_DD": [
            ["Dpi_Kspipi_misID_PHSP", "Cruijff+Gaussian", "Legendre_2_2"],
        ],
    },
    "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow":
    {
        "CB2DPI_D2KSPIPI_DD": [
            ["Dpi_Kspipi_PHSP", "Cruijff+Gaussian", "Legendre_2_2"],
        ],
        "CB2DK_D2KSPIPI_DD": [
            ["DK_Kspipi_misID_PHSP", "CBShape", "Legendre_2_2"],
        ],
    },
    "COMBINATORIAL_BACKGROUND_DK":
    {
        "CB2DK_D2KSPIPI_DD": [
            ["Combinatorial", "Exponential", "Legendre_2_2"],
        ],
    }
}


################ their tex formatting
components_tex = {
    "DK_Kspipi": r"$B^{\pm} \rightarrow D K^{\pm}$",
    "DK_Kspipi_PHSP": r"$B^{\pm} \rightarrow D K^{\pm}$ PHSP",
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
list_sources  = ["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow" , "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow", "SDATA"]
list_channels = ["CB2DK_D2KSPIPI_DD", "CB2DPI_D2KSPIPI_DD"]

for channel in list_channels:
    os.makedirs(f"{channel}", exist_ok=True)
    os.makedirs(f"{channel}_preFit", exist_ok=True)
    pass

#### get MC ntuples and their efficiencies
for source in list_sources:
    ntuples[source] = {}
    pre_cuts_eff[source] = {}
    fin_cuts_eff[source] = {}
    for channel in list_channels:
        ntuples[source][channel]  = Ntuple(f"{source}",channel,"YRUN2", "MagAll")
        print(ntuples[source][channel])
        pre_cuts_eff[source][channel] = ntuples[source][channel].get_merged_cuts_eff("preliminary")
        fin_cuts_eff[source][channel] = ntuples[source][channel].get_merged_cuts_eff("final")
        pass
    pass

########### here we get BDT and PID eff that are not yet in the final ntuples
for source in list_sources:
    BDT_cut_efficiency[source] = {}
    total_eff[source] = {}
    cut[source] = {}
    paths[source] = {}
    list_var[source] = {}
    for channel in list_channels:
        ntuples[source][channel].initialise_fit(components[source][channel])
        cut[source][channel] = str_BDT_cut + " & " + ntuples[source][channel].dict_final_cuts["Bach_PID"] + " & (" + ntuples[source][channel].variable_to_fit + f" < {max_mass} ) & (" + ntuples[source][channel].variable_to_fit + f" > {min_mass} )"
        paths[source][channel] = ntuples[source][channel].final_cuts_paths
        list_var[source][channel] = [ntuples[source][channel].variable_to_fit] + basic_list_var
        ntuples[source][channel].store_events(paths[source][channel],list_var[source][channel],cut[source][channel], Kspipi)
        BDT_cut_efficiency[source][channel] = len(ntuples[source][channel].uproot_data[list_var[source][channel][0]]) / fin_cuts_eff[source][channel]["YRUN2"]["MagAll"]["selected_events"]
        total_eff[source][channel] = pre_cuts_eff[source][channel]["YRUN2"]["MagAll"]["efficiency"]*fin_cuts_eff[source][channel]["YRUN2"]["MagAll"]["efficiency"]*BDT_cut_efficiency[source][channel]
        # ntuples[source][channel].store_amplitudes(Kspipi)
        pass
    pass

print(json.dumps(total_eff,indent=4))

######### data tuple
ntuple_test = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"]


######## define starting values of the variables in the fit
start_values = {
    "ratio_BR_DK_to_Dpi" :                          0.044,
    "signal_mean_DK":                              5279.0, # varDict['signal_mean']+50,
    "signal_mean_Dpi":                             5279.0, # varDict['signal_mean']+50,
    "signal_width_DK_DD":                           14.49, # varDict['sigma_dk_DD']+50,
    "signal_width_Dpi_DD":                          15.82, # varDict['sigma_dk_DD']+50,
    # "Dst0K_D0pi0_Kspipi_yield_DD":                 3000.,
    # "DstpK_D0pip_Kspipi_yield_DD":                 3000.,
    # "Dst0pi_D0pi0_Kspipi_misID_PartReco_yield_DD": 3000.,
    # "Dstppi_D0pip_Kspipi_misID_PartReco_yield_DD": 3000.,
    # "Dst0K_D0gamma_Kspipi_yield_DD"              : 3000.,
    "Combinatorial_yield_DK_DD_Bplus" :                    3600., # 1200.,
    "Combinatorial_yield_DK_DD_Bminus":                    3640., # 12000.,
    "Dpi_Kspipi_yield_DD_Bplus" :                         73580., # 12000.,
    "Dpi_Kspipi_yield_DD_Bminus":                         72400., # 120000.,
    # "Dst0pi_D0pi0_Kspipi_yield_DD":                7000.,
    # "Dst0pi_D0gamma_Kspipi_yield_DD":              7000.,
    "Combinatorial_yield_DPI_DD_Bplus" :                  8600., # 2000.,                     
    "Combinatorial_yield_DPI_DD_Bminus":                  8700., # 20000.,                     
    "Combinatorial_c_DK_DD":                       -0.005,
    "xplus"                                        : 1.00,
    "yplus"                                        : 0.00,
    "xminus"                                       : 1.00,
    "yminus"                                       : 0.00,
    # "xxi"                                          : 1.00,
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
    # "xxi"                                          : [-1., 1.],
    # "yxi"                                          : [-1., 1.],
}

###### define which free fit parameters is applied to which "real" variables
dict_shared_parameters = {
    "ratio_BR_DK_to_Dpi" : [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "ratio_BR_DK_to_Dpi"]
    ],
    "signal_mean_DK": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_m0"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "gauss_mean"],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSKK_DD", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSPIPI_LL", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSPIPI_LL", "DK_Kspipi", "gauss_mean"],
        # ["CB2DK_KSKK_LL", "DK_Kspipi", "cruij_m0"  ],
        # ["CB2DK_KSKK_LL", "DK_Kspipi", "gauss_mean"],
    ],
    "signal_mean_Dpi": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_m0"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "gauss_mean"],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSKK_DD", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSPIPI_LL", "Dpi_Kspipi", "gauss_mean"],
        # ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "cruij_m0"  ],
        # ["CB2DPI_KSKK_LL", "Dpi_Kspipi", "gauss_mean"],
    ],
    "signal_width_DK_DD": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_sigmaL"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_sigmaR"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "gauss_sigma"],
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
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_sigmaL"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_sigmaR"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "gauss_sigma"],
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
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "mass", "yield_Bplus"  ],
    ],
    "Combinatorial_yield_DK_DD_Bminus": [
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "mass", "yield_Bminus"  ],
    ],
    "Dpi_Kspipi_yield_DD_Bplus": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus"  ],
    ],
    "Dpi_Kspipi_yield_DD_Bminus": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bminus"  ],
    ],
    # "Dst0pi_D0pi0_Kspipi_yield_DD": [ 
    #     ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi", "yield_Bplus"  ],
    # ],
    # "Dst0pi_D0gamma_Kspipi_yield_DD": [ 
    #     ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0gamma_Kspipi", "yield_Bplus"  ],
    # ],
    "Combinatorial_yield_DPI_DD_Bplus": [
        ["CB2DPI_D2KSPIPI_DD", "Combinatorial", "mass", "yield_Bplus"  ],
    ],
    "Combinatorial_yield_DPI_DD_Bminus": [
        ["CB2DPI_D2KSPIPI_DD", "Combinatorial", "mass", "yield_Bminus"  ],
    ],
    "Combinatorial_c_DK_DD": [ # Exponential 11
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "mass", "c"],
    ],
    "xplus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "xplus"],
    ],
    "yplus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "yplus"],
    ],
    "xminus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "xminus"],
    ],
    "yminus": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "yminus"],
    ],
    # "xxi": [
    #     ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "xxi"],
    # ],
    # "yxi": [
    #     ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "yxi"],
    # ],
}


############################ ALL KIND OF RATIOS OF EFFICIENCIES
############# compute ratio of efficiencies etc to constrain parameters
# constraining the DK yields to the Dpi one
ratio_DK_to_Dpi = {}
ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] = total_eff["MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DPI_D2KSPIPI_DD"]/total_eff["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DK_D2KSPIPI_DD"]
# constraining the misID Dpi to its non-misID counterpart
ratio_Dpi_misID_to_Dpi = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"] = {}
ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] = total_eff["MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DK_D2KSPIPI_DD"]/total_eff["MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DPI_D2KSPIPI_DD"]
# constraining the misID DK to its non-misID counterpart
ratio_DK_misID_to_DK = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"] = {}
ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] = total_eff["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DPI_D2KSPIPI_DD"]/total_eff["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DK_D2KSPIPI_DD"]
# constraining the misID DK to its non-misID counterpart - Part Reco
        
######### FIXED CONSTRAINTS
# (only implemented with multiplicative factor)
dict_constrained_parameters = [
    ## constrain DK from Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi"       , "mass", "yield_Bplus" ], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus",
                                                            [ "SHARED_THROUGH_CHANNELS", "parameters", "mass", "ratio_BR_DK_to_Dpi", ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ] ],
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi"       , "mass", "yield_Bminus"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bminus",
                                                            [ "SHARED_THROUGH_CHANNELS", "parameters", "mass", "ratio_BR_DK_to_Dpi", ratio_DK_to_Dpi["CB2DK_D2KSPIPI_DD"] ] ] ],
    ## constrain misID in DK from goodID in Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "mass", "yield_Bplus"] , ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] ] ],
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "mass", "yield_Bminus"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bminus", ratio_Dpi_misID_to_Dpi["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"] ] ],
    ## constrain misID DPI from goodID in DK
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "mass", "yield_Bplus"] , ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "mass", "yield_Bplus", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] ] ],
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "mass", "yield_Bminus"], ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "mass", "yield_Bminus", ratio_DK_misID_to_DK["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"] ] ],
]
print(json.dumps(dict_constrained_parameters,indent=4))

####################### GAUSSIAN CONSTRAINTS
ratio_Dst0K_to_DstpK = Measurement(1,0.1)
######### define which real variable is constrained from which other var
# (only implemented with multiplicative factor)
dict_gaussian_constraints = [
        # [ ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "mass", "yield_Bplus"], ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "yield_Bplus", "mass", ratio_Dst0K_to_DstpK.value, ratio_Dst0K_to_DstpK.error] ],
]
print(json.dumps(dict_gaussian_constraints,indent=4))


#### This is how we can change some initial parameters
fixed_variables = dict(**VARDICT["SDATA"])
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["ratio_BR_DK_to_Dpi"] =  0.07
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["xplus"             ] =  0.10
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["yplus"             ] =  0.00
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["xminus"            ] =  0.10
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["yminus"            ] =  0.00
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["xxi"               ] =  1.00
fixed_variables["SHARED_THROUGH_CHANNELS"]["parameters"]["mass"]["yxi"               ] =  0.00

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
    ntuples["SDATA"]
)
parameters_to_fit = NLL.parameters_to_fit

preFit_list_variables = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=list(start_values.values()),
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)


####################### some tools for plotting: 2D grid of invariant Dalitz masses
Dalitz_Kspip_mat, Dalitz_Kspim_mat = np.meshgrid(Dalitz_mass_vec,Dalitz_mass_vec)
# Dalitz_Kspip_mat = tf.cast(Dalitz_Kspip_mat, tf.float64)
# Dalitz_Kspim_mat = tf.cast(Dalitz_Kspim_mat, tf.float64)
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


############
#### normalisation
ntuple_normalisation = ntuples["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DK_D2KSPIPI_DD"]
# ntuple_normalisation = ntuples["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi"]
norm_ampD0    = ntuple_normalisation.AmpD0
norm_ampD0bar = ntuple_normalisation.AmpD0bar
norm_zp_p     = ntuple_normalisation.zp_p
norm_zm_pp    = ntuple_normalisation.zm_pp
##############
flat_norm_ampD0    = ampD0.reshape(ampD0.shape[0]*ampD0.shape[0])
flat_norm_ampD0bar = ampD0bar.reshape(ampD0bar.shape[0]*ampD0bar.shape[0])
flat_norm_zp_p     = zp_p.reshape(zp_p.shape[0]*zp_p.shape[0])
flat_norm_zm_pp    = zm_pp.reshape(zm_pp.shape[0]*zm_pp.shape[0])

norm_ampD0    = {
    "Bplus"  : flat_norm_ampD0,
    "Bminus" : flat_norm_ampD0,
}
norm_ampD0bar    = {
    "Bplus"  : flat_norm_ampD0bar,
    "Bminus" : flat_norm_ampD0bar,
}
norm_zp_p    = {
    "Bplus"  : flat_norm_zp_p,
    "Bminus" : flat_norm_zp_p,
}
norm_zm_pp    = {
    "Bplus"  : flat_norm_zm_pp,
    "Bminus" : flat_norm_zm_pp,
}

for channel in list_channels:
    # ntuple_test          = ntuples["CB2DPI_D2KSPIPI_DD"]["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]
    # this one defines the dalitz and mass pdfs, and tores the normalisation events
    ntuples["SDATA"][channel].define_dalitz_pdfs(norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp)
    # print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].dalitz_pdfs.keys()  )
    # print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].dalitz_pdfs["Bplus"])
    # print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].mass_pdfs.keys()    )
    # print(ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].mass_pdfs["Bplus"]  )
    # ntuple_test.get_list_variables(NLL.fixed_variables)
    # this bit runs get_mass_pdf and get_dalitz_pdf
    # for all components and B signs
    pass


def draw_dalitz_pdf(ntuple, ampD0, ampD0bar, zp_p, zm_pp, variables=None):
    """
    this function is only for plotting purposes. It computes the pdfs 
    for each component and the sum.
    """
    if (variables == None):
        variables = ntuple.list_variables
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
                print(np.max(tmp_pdf_values))
            else:
                ## otherwise it's a function of the SRD variables
                print(comp_pdf.component)
                tmp_pdf_values = comp_pdf.pdf(zp_p, zm_pp)
                print(np.max(tmp_pdf_values))
                pass
            #### multiply each pdf by its yield
            index_yields  = INDEX_YIELDS[Bsign]
            comp_yield    = variables[ntuple.i_c][i][2][index_yields]
            tmp_pdf_values = tmp_pdf_values*dalitz_scaling*dalitz_scaling*comp_yield
            pdfs_values[Bsign][comp_pdf.component]     = tmp_pdf_values
            pdfs_values[Bsign]["total_pdf"]           += tmp_pdf_values
            pass # loop comps
        pass # loop signs
    return pdfs_values

dalitz_pdfs_values = {}
for channel in list_channels:
    #### compute the pdfs
    dalitz_pdfs_values[channel] = draw_dalitz_pdf(
        ntuples["SDATA"][channel],
        ampD0   ,
        ampD0bar,
        zp_p    ,
        zm_pp   ,
        variables=preFit_list_variables
    )
    pass

# print(dalitz_pdfs_values)



### plotting these pdfs
for channel in list_channels:
    for Bsign in dalitz_pdfs_values[channel].keys():
        for comp in dalitz_pdfs_values[channel][Bsign].keys():
            try:
                cs = plt.contourf(
                    Dalitz_Kspip_mat,
                    Dalitz_Kspim_mat,
                    dalitz_pdfs_values[channel][Bsign][comp],
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
            plt.savefig(f"{channel}_preFit/{Bsign}_{comp}.png")
            plt.close("all")
            pass
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
                ))*float(nbins)
                Kspip_projection[Bsign]["total_pdf"][i_Kspi] += float(tf.reduce_mean(
                    tmp_numpy_pdfs.transpose()[i_Kspi]
                ))*float(nbins)
                #### Kspim
                Kspim_projection[Bsign][comp_pdf.component][i_Kspi] = float(tf.reduce_mean(
                    tmp_numpy_pdfs[i_Kspi]
                ))*float(nbins)
                Kspim_projection[Bsign]["total_pdf"][i_Kspi] += float(tf.reduce_mean(
                    tmp_numpy_pdfs[i_Kspi]
                ))*float(nbins)
                # if (Kspip_projection[Bsign][comp_pdf.component][i_Kspip] > 0):
                #     print(Kspip_projection[Bsign][comp_pdf.component][i_Kspip])
                pass # loop axes
            pass # loop comps
        pass # loop signs
    return Kspip_projection, Kspim_projection


########## get the projections
Kspi_projection = {}
B_data          = {}
for channel in list_channels:
    Kspi_projection[channel] = draw_projection_pdf(
        ntuples["SDATA"][channel],
        dalitz_pdfs_values[channel]
    )
    B_data[channel] = {
        "Bplus" : ntuples["SDATA"][channel].Bplus_data ,
        "Bminus": ntuples["SDATA"][channel].Bminus_data,
    }
    pass    

for channel in list_channels:
    ntuple_plot = ntuples["SDATA"][channel]
    #####
    for Bsign in BSIGNS.keys():
        tmp_data = B_data[channel][Bsign]
        plt.plot(
            Dalitz_mass_vec,
            Kspi_projection[channel][0][Bsign]["total_pdf"],
            label="total_pdf"
        )
        for i in range(len(ntuple_plot.dalitz_pdfs[Bsign])):
            comp_pdf = ntuple_plot.dalitz_pdfs[Bsign][i]
            index_yields  = INDEX_YIELDS[Bsign]
            if (preFit_list_variables[ntuple_plot.i_c][i][2][index_yields] == 0): continue
            plt.fill_between(
                Dalitz_mass_vec,
                Kspi_projection[channel][0][Bsign][comp_pdf.component],
                alpha=0.5,
                label=comp_pdf.component
            )
            pass
        # plt.fill_between(
        #     Dalitz_mass_vec,
        #     Kspip_projection[Bsign]["Dpi_Kspipi_misID"],
        #     alpha=0.5,
        #     label="Dpi_Kspipi_misID"
        # )
        mplhep.histplot(
            np.histogram(tmp_data["m_Kspip"], bins=nbins, range=m_Kspip_range),
            label=ntuple_plot.tex
            # , density=True
        )
        plt.legend(title=Bsign)
        plt.xlabel("Kspip")
        plt.ylabel(f"Events / ({round(dalitz_scaling*1000)} MeV)")
        plt.tight_layout()
        plt.savefig(f"{channel}_preFit/{Bsign}_total_pdf_Kspip_projection.png")
        plt.close("all")
        plt.plot(
            Dalitz_mass_vec,
            Kspi_projection[channel][1][Bsign]["total_pdf"],
            label="total_pdf"
        )
        for i in range(len(ntuple_plot.dalitz_pdfs[Bsign])):
            comp_pdf = ntuple_plot.dalitz_pdfs[Bsign][i]
            index_yields  = INDEX_YIELDS[Bsign]
            if (preFit_list_variables[ntuple_plot.i_c][i][2][index_yields] == 0): continue
            plt.fill_between(
                Dalitz_mass_vec,
                Kspi_projection[channel][1][Bsign][comp_pdf.component],
                alpha=0.5,
                label=comp_pdf.component
            )
            pass
        mplhep.histplot(
            np.histogram(tmp_data["m_Kspim"], bins=nbins, range=m_Kspip_range),
            label=ntuple_plot.tex
            # , density=True
        )
        plt.legend(title=Bsign)
        plt.xlabel("Kspim")
        plt.ylabel(f"Events / ({round(dalitz_scaling*1000)} MeV)")
        plt.tight_layout()
        plt.savefig(f"{channel}_preFit/{Bsign}_total_pdf_Kspim_projection.png")
        plt.close("all")
        pass
    pass


####### and now the mass pdfs
mass_pdfs_values = {}
for channel in list_channels:
    mass_pdfs_values[channel] = ntuples["SDATA"][channel].draw_mass_pdfs(
        tf_Bmass_vec,
        preFit_list_variables
    )
    for Bsign in BSIGNS.keys():
        tmp_data = B_data[channel][Bsign]
        mplhep.histplot(
            np.histogram(tmp_data[ntuples["SDATA"][channel].variable_to_fit], bins=nbins),
            label=ntuples["SDATA"][channel].tex
        )
        plt.plot(
            Bmass_vec,
            mass_scaling*mass_pdfs_values[channel][Bsign]["total_mass_pdf"],
            label="Total"
        )
        for comp in ntuples["SDATA"][channel].components:
            plt.plot(
                Bmass_vec,
                mass_pdfs_values[channel][Bsign][comp[0]]*mass_scaling,linestyle="--",
                label=components_tex[comp[0]]+"\n"+comp[1]
            )
            pass
        plt.xlabel("Constrained $D\pi$ mass")
        plt.ylabel(f"Events / ({round(mass_scaling)} MeV)")
        plt.tight_layout()
        plt.savefig(f"{channel}_preFit/{Bsign}_mass_distribution.png")
        plt.close("all")
        pass
    pass


@tf.function
def nll(x):
    return NLL.get_total_nll(x) # , tensor_to_fit)

# ntuple_normalisation_DK_DD  = ntuple_test
ntuple_normalisation_DPI_DD = ntuples["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]["CB2DPI_D2KSPIPI_DD"]
dict_norm_ampD0    = {
    "CB2DK_D2KSPIPI_DD" :  norm_ampD0, # ntuple_normalisation_DK_DD.AmpD0,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.AmpD0,
}
dict_norm_ampD0bar = {
    "CB2DK_D2KSPIPI_DD" :  norm_ampD0bar, # ntuple_normalisation_DK_DD.AmpD0bar,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.AmpD0bar,
}
dict_norm_zp_p = {
    "CB2DK_D2KSPIPI_DD" :  norm_zp_p, # ntuple_normalisation_DK_DD.zp_p,
    "CB2DPI_D2KSPIPI_DD": ntuple_normalisation_DPI_DD.zp_p,
}
dict_norm_zm_pp = {
    "CB2DK_D2KSPIPI_DD" :  norm_zm_pp, # ntuple_normalisation_DK_DD.zm_pp,
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



# m = iminuit.Minuit(nll, x, name=parameters_to_fit)
# m.limits = list(limit_values.values())
# mg = m.migrad()


# print(mg)
# means  = mg.values
# errors = mg.errors
# hesse  = mg.hesse()
# cov    = hesse.covariance
# corr   = cov.correlation()
# ## i have to loop over the entries if this dict to set the pandas df myself
# corr_array = np.zeros(len(parameters_to_fit)*len(parameters_to_fit)).reshape(len(parameters_to_fit),len(parameters_to_fit))
# for i in range(len(parameters_to_fit)):
#     corr_array[i] = corr[parameters_to_fit[i]]
#     pass
# pd_cov = pd.DataFrame(corr_array,index=parameters_to_fit,columns=parameters_to_fit)
# for i in range(len(means)):
#     print(f'{parameters_to_fit[i]:<50}', ": ", means[i] ," +- ", errors[i])
#     pass
# # print("Means   ", means)
# # print("Errors  ", errors)
# vec_results = dict(zip(parameters_to_fit,means))
# print(json.dumps(vec_results,indent=4))
# with open(f'simfit_output.json', 'w') as f:
#     json.dump(vec_results,f,indent=4)
#     pass


# # columns_to_keep = ["Wilson_C9_tau", "Wilson_C9", "Wilson_C10", "J_psi_phase", "psi_2S_phase"]
# # pull_df_corr = pull_df[columns_to_keep].corr()   # correlation matrix in pandas dataframe
# fig = plt.figure(figsize=(10, 8))  
# sns.heatmap(pd_cov, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.tight_layout()
# plt.savefig("covariance.png")
# plt.close("all")
# # pages.append(fig)

# postFit_list_variables = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].get_list_variables(
#     NLL.fixed_variables,
#     params=means,
#     shared_parameters=NLL.shared_parameters,
#     constrained_parameters=NLL.constrained_parameters
# )

# #### compute the pdfs
# postFit_dalitz_pdfs_values = {}
# for channel in list_channels:
#     # print(1)
#     postFit_dalitz_pdfs_values[channel] = draw_dalitz_pdf(
#         ntuples["SDATA"][channel],
#         ampD0   ,
#         ampD0bar,
#         zp_p    ,
#         zm_pp   ,
#         variables=postFit_list_variables
#     )
#     # print(2)
#     ### plotting these pdfs
#     for Bsign in postFit_dalitz_pdfs_values[channel].keys():
#         # print(3)
#         for comp in postFit_dalitz_pdfs_values[channel][Bsign].keys():
#             # print(4)
#             try:
#                 cs = plt.contourf(
#                     Dalitz_Kspip_mat,
#                     Dalitz_Kspim_mat,
#                     postFit_dalitz_pdfs_values[channel][Bsign][comp],
#                     levels=100) # ,
#                 # norm="log")
#             except TypeError:
#                 continue
#             # , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
#             plt.xlabel("Kspip")
#             plt.ylabel("Kspim")
#             cbar = plt.colorbar(cs)
#             # cs.cmap.set_over('red')
#             # cs.cmap.set_under('white')
#             # cs.changed()
#             plt.tight_layout()
#             plt.savefig(f"{channel}/{Bsign}_{comp}.png")
#             plt.close("all")
#             pass
#         pass
#     pass

# ########## get the projections
# postFit_Kspi_projection = {}
# for channel in list_channels:
#     postFit_Kspi_projection[channel] = draw_projection_pdf(
#         ntuples["SDATA"][channel],
#         postFit_dalitz_pdfs_values[channel]
#     )
#     #####
#     for Bsign in BSIGNS.keys():
#         # print(1)
#         tmp_data = B_data[channel][Bsign]
#         ### draw 2D distribution we are fitting
#         # plt.scatter(tmp_data["m_Kspip"], tmp_data["m_Kspim"])
#         mplhep.hist2dplot(np.histogram2d(tmp_data["m_Kspip"], tmp_data["m_Kspim"], bins=[nbins,nbins], range= [m_Kspip_range, m_Kspip_range]))
#         plt.savefig(f"{channel}/{Bsign}_input_dalitz.png")
#         plt.close("all")
#         plt.plot(
#             Dalitz_mass_vec,
#             postFit_Kspi_projection[channel][0][Bsign]["total_pdf"],
#             label="total_pdf"
#         )
#         for i in range(len(ntuples["SDATA"][channel].dalitz_pdfs[Bsign])):
#             comp_pdf = ntuples["SDATA"][channel].dalitz_pdfs[Bsign][i]
#             print(comp_pdf.component)
#             index_yields  = INDEX_YIELDS[Bsign]
#             if (NLL.fixed_variables[ntuples["SDATA"][channel].i_c][i][index_yields] == 0): continue
#             plt.fill_between(
#                 Dalitz_mass_vec,
#                 postFit_Kspi_projection[channel][0][Bsign][comp_pdf.component],
#                 alpha=0.5,
#                 label=comp_pdf.component
#             )
#             pass
#         # plt.fill_between(
#         #     Dalitz_mass_vec,
#         #     Kspip_projection[Bsign]["Dpi_Kspipi_misID"],
#         #     alpha=0.5,
#         #     label="Dpi_Kspipi_misID"
#         # )
#         mplhep.histplot(
#             np.histogram(tmp_data["m_Kspip"], bins=nbins, range=m_Kspip_range),
#             label=ntuples["SDATA"][channel].tex
#         )
#         plt.legend(title=Bsign)
#         plt.xlabel("Kspip")
#         plt.ylabel(f"Events / ({round(dalitz_scaling*1000)} MeV)")
#         plt.tight_layout()
#         plt.savefig(f"{channel}/{Bsign}_total_pdf_Kspip_projection.png")
#         plt.close("all")
#         plt.plot(
#             Dalitz_mass_vec,
#             postFit_Kspi_projection[channel][1][Bsign]["total_pdf"],
#             label="total_pdf"
#         )
#         for i in range(len(ntuples["SDATA"][channel].dalitz_pdfs[Bsign])):
#             comp_pdf = ntuples["SDATA"][channel].dalitz_pdfs[Bsign][i]
#             index_yields  = INDEX_YIELDS[Bsign]
#             if (NLL.fixed_variables[ntuples["SDATA"][channel].i_c][i][index_yields] == 0): continue
#             plt.fill_between(
#                 Dalitz_mass_vec,
#                 postFit_Kspi_projection[channel][1][Bsign][comp_pdf.component],
#                 alpha=0.5,
#                 label=comp_pdf.component
#             )
#             pass
#         mplhep.histplot(
#             np.histogram(tmp_data["m_Kspim"], bins=nbins, range=m_Kspip_range),
#             label=ntuples["SDATA"][channel].tex
#         )
#         plt.legend(title=Bsign)
#         plt.xlabel("Kspim")
#         plt.ylabel(f"Events / ({round(dalitz_scaling*1000)} MeV)")
#         plt.tight_layout()
#         plt.savefig(f"{channel}/{Bsign}_total_pdf_Kspim_projection.png")
#         plt.close("all")
#         pass
#     pass



# ####### and now the mass pdfs
# postFit_mass_pdfs_values = {}
# for channel in list_channels:
#     postFit_mass_pdfs_values[channel] = ntuples["SDATA"][channel].draw_mass_pdfs(
#         tf_Bmass_vec,
#         postFit_list_variables
#     )
#     for Bsign in BSIGNS.keys():
#         tmp_data = B_data[channel][Bsign]
#         mplhep.histplot(
#             np.histogram(tmp_data[ntuples["SDATA"][channel].variable_to_fit], bins=nbins),
#             label=ntuples["SDATA"][channel].tex
#         )
#         plt.plot(
#             Bmass_vec,
#             mass_scaling*postFit_mass_pdfs_values[channel][Bsign]["total_mass_pdf"],
#             label="Total"
#         )
#         for comp in ntuples["SDATA"][channel].components:
#             plt.plot(
#                 Bmass_vec,
#                 postFit_mass_pdfs_values[channel][Bsign][comp[0]]*mass_scaling,linestyle="--",
#                 label=components_tex[comp[0]]+"\n"+comp[1]
#             )
#             pass
#         plt.xlabel("Constrained $D\pi$ mass")
#         plt.ylabel(f"Events / ({round(mass_scaling)} MeV)")
#         plt.tight_layout()
#         plt.savefig(f"{channel}/{Bsign}_mass_distribution.png")
#         plt.close("all")
#         pass
#     pass


# print(mg)
