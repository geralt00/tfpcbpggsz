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
import os

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

date = "2025_07_15"

####### NOT CONDOR
plot_dir=f"{date}/invariant_mass_fit"
######

# ########## CONDOR
# plot_dir=f"/shared/scratch/rj23972/safety_net/tfpcbpggsz/canorman_InvMassFit/{plot_dir}"
# #########

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(f"{plot_dir}/preFit", exist_ok=True)


time1 = time.time()
#### scale = bin width for plotting purposes
scale = 5

from tfpcbpggsz.core import *
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.amp import *


from tfpcbpggsz.amp_up import *
Kspipi_up = PyD0ToKSpipi2018()
Kspipi_up.init()


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
paths = {}
list_var = {}
cut = {}

BDT_cut = 0.4
max_mass = 5800
min_mass = 5080
Bmass_vec = np.arange(min_mass, max_mass, 1)
B_mass_range = [min_mass, max_mass]
tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)
BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
str_BDT_cut = "(BDT_output > "+str(BDT_cut)+")"
nbins = 100
mass_scaling   = (max_mass - min_mass) / float(nbins)
mplhep.style.use("LHCb2")
kwargs_data = {
    # "xerr"        : mass_scaling/2.,
    "histtype"    : 'errorbar',
    "linestyle"   : "None",
    "color"       : "black",
    # "markersize"  : 3,
    # "capsize"     : 1.5,    
}


################ define the components in each channel and source
components = {
    "CB2DK_D2KSPIPI_DD": [ 
        ["DK_Kspipi", "Cruijff+Gaussian", "Flat"],
        ["Dpi_Kspipi_misID", "SumCBShape", "Flat"],
        ["Dst0K_D0pi0_Kspipi", "HORNSdini", "Flat"],
        ["DstpK_D0pip_Kspipi", "HORNSdini", "Flat"],
        ["Dst0pi_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID", "Flat"],
        ["Dstppi_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID", "Flat"],
        ["Dst0K_D0gamma_Kspipi", "HILLdini", "Flat"],
        ["Dst0pi_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID", "Flat"],
        ["DKpi_Kspipi", "HORNSdini+Gaussian", "Flat"],
        ["Dpipi_Kspipi_misID_PartReco", "HORNSdini_misID", "Flat"],
        ["Bs2DKpi_Kspipi_PartReco", "HORNSdini", "Flat"],
        ["Combinatorial", "Exponential", "Flat"],
    ],
    "CB2DPI_D2KSPIPI_DD": [ 
        ["Dpi_Kspipi", "Cruijff+Gaussian", "Flat"],
        ["DK_Kspipi_misID", "CBShape", "Flat"],
        ["Dst0pi_D0pi0_Kspipi", "HORNSdini", "Flat"],
        ["Dstppi_D0pip_Kspipi", "HORNSdini", "Flat"],
        ["Dst0K_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID", "Flat"],
        ["DstpK_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID", "Flat"],
        ["Dst0pi_D0gamma_Kspipi", "HILLdini", "Flat"],
        ["Dst0K_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID", "Flat"],
        ["Dpipi_Kspipi", "HORNSdini+HORNSdini", "Flat"],
        ["Combinatorial", "Exponential", "Flat"],
    ]
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


####### 
input_variables = {}
for channel in list_channels:
    input_variables[channel]       = {} ## loop over channel
    for comp in components[channel]: ## loop over components
        input_variables[channel][comp[0]] = VARDICT["SDATA"][channel][comp[0]]
        # print(comp[0])
        pass
    pass

input_variables["SHARED_THROUGH_CHANNELS"] = VARDICT["SDATA"]["SHARED_THROUGH_CHANNELS"]


###########
######## define starting values of the variables in the fit
inputs = {
    "gamma" : deg_to_rad(GAMMA), "rb" : RB_DK, "dB" : deg_to_rad(DELTAB_DK),
    "rb_dpi": RB_DPI, "dB_dpi": deg_to_rad(DELTAB_DPI)
}
inputs["xplus"], inputs["yplus"], inputs["xminus"], inputs["yminus"], inputs["xxi"], inputs["yxi"] = get_xy_xi(
    (inputs["gamma"], inputs["rb"], inputs["dB"], inputs["rb_dpi"], inputs["dB_dpi"])
)

############ initialise ntuples for fitting
basic_list_var = ["Bu_ID", "zp_p", "zm_pp", "m_Kspip", "m_Kspim"]
# for particle in ["KS","pim","pip"]:
#     for mom in ["PE", "PX", "PY", "PZ"]:
#         basic_list_var += [f"{particle}_{mom}"]
#         pass
#     pass
for particle in ["Ks","h1","h2"]:
    for mom in ["PE", "PX", "PY", "PZ"]:
        basic_list_var += [f"{particle}_{mom}"]
        pass
    pass

#### get MC ntuples and their efficiencies
for source in list_sources:
    ntuples[source] = {}
    pre_cuts_eff[source] = {}
    fin_cuts_eff[source] = {}
    for channel in list_channels:
        ntuples[source][channel]  = Ntuple(source+"_TightCut_LooserCuts_fixArrow",channel,"YRUN2", "MagAll")
        print(ntuples[source][channel])
        pre_cuts_eff[source][channel] = ntuples[source][channel].get_merged_cuts_eff("preliminary")
        fin_cuts_eff[source][channel] = ntuples[source][channel].get_merged_cuts_eff("final")
########### here we get BDT and PID eff that are not yet in the final ntuples
for source in list_sources:
    BDT_cut_efficiency[source] = {}
    total_eff[source] = {}
    cut[source] = {}
    paths[source] = {}
    list_var[source] = {}
    for channel in list_channels:
        cut[source][channel] = str_BDT_cut + " & " + ntuples[source][channel].dict_final_cuts["Bach_PID"] + " & (" + ntuples[source][channel].variable_to_fit + " < 5800 ) & (" + ntuples[source][channel].variable_to_fit + " > 5080 )"
        paths[source][channel] = ntuples[source][channel].final_cuts_paths
        list_var[source][channel] = [ntuples[source][channel].variable_to_fit] + basic_list_var
        ntuples[source][channel].store_events(paths[source][channel],
                                              list_var[source][channel],
                                              cut[source][channel],
                                              Kspipi_up)
        BDT_cut_efficiency[source][channel] = len(ntuples[source][channel].uproot_data[list_var[source][channel][0]]) / fin_cuts_eff[source][channel]["YRUN2"]["MagAll"]["selected_events"]
        total_eff[source][channel] = pre_cuts_eff[source][channel]["YRUN2"]["MagAll"]["efficiency"]*fin_cuts_eff[source][channel]["YRUN2"]["MagAll"]["efficiency"]*BDT_cut_efficiency[source][channel]
        pass
    pass

ntuples["SDATA"] = {}
cut["SDATA"]     = {}
paths["SDATA"]   = {}
list_var["SDATA"]= {}
######## initialise data ntuples
for channel in list_channels:
    # i_s = INDEX_SOURCE_TO_VARDICT["SDATA"]
    i_c = INDEX_CHANNEL_TO_VARDICT[channel]
    ntuples["SDATA"][channel] = Ntuple("SDATA",channel,"YRUN2", "MagAll")
    index_channel = list(input_variables.keys()).index(channel)
    ntuples["SDATA"][channel].initialise_fit(components[channel], index_channel)
    ntuples["SDATA"][channel].define_mass_pdfs()
    # print(ntuples["SDATA"][channel])
    # print(ntuples["SDATA"][channel].components)
    # print(ntuples["SDATA"][channel].variable_to_fit)
    cut["SDATA"][channel] = str_BDT_cut + " & " + ntuples["SDATA"][channel].dict_final_cuts["Bach_PID"] + " & (" + ntuples[source][channel].variable_to_fit + " < 5800 ) & (" + ntuples[source][channel].variable_to_fit + " > 5080 )"
    paths["SDATA"][channel] = ntuples["SDATA"][channel].final_cuts_paths
    list_var["SDATA"][channel] = [ntuples["SDATA"][channel].variable_to_fit] + basic_list_var
    ntuples["SDATA"][channel].store_events(paths["SDATA"][channel],
                                           list_var["SDATA"][channel],
                                           cut["SDATA"][channel],
                                           Kspipi_up)


                
######## define starting values of the variables in the fit
start_values = {
    "ratio_BR_DK_to_Dpi" :                         0.07,
    "signal_mean_DK":                              varDict['signal_mean']+50,
    "signal_mean_Dpi":                             varDict['signal_mean']+50,
    "signal_width_DK_DD":                          varDict['sigma_dk_DD']+50,
    "signal_width_Dpi_DD":                         varDict['sigma_dk_DD']+50,
    "Dst0K_D0pi0_Kspipi_yield_Bplus_DD" :                 3000.,
    "DstpK_D0pip_Kspipi_yield_Bplus_DD":                 3000.,
    "Dstppi_D0pip_Kspipi_misID_PartReco_yield_Bplus_DD": 3000.,
    "Dst0K_D0gamma_Kspipi_yield_Bplus_DD"              : 3000.,
    "Combinatorial_yield_Bplus_DK_DD":                   12000.,
    "Dpi_Kspipi_yield_Bplus_DD":                         120000.,
    "Dst0pi_D0pi0_Kspipi_yield_Bplus_DD":                7000.,
    "Dst0pi_D0gamma_Kspipi_yield_Bplus_DD":              7000.,
    "Combinatorial_yield_Bplus_DPI_DD":                  20000.,                     
    "Combinatorial_c_DK_DD":                       -0.003,
}

####### define limits 
limit_values = {
    "ratio_BR_DK_to_Dpi" :                         [0.0001,1],
    "signal_mean_DK":                              [5080,5800],
    "signal_mean_Dpi":                             [5080,5800],
    "signal_width_DK_DD":                          [1,100],
    "signal_width_Dpi_DD":                         [1,100],
    "Dst0K_D0pi0_Kspipi_yield_Bplus_DD":                 [0, 20000],
    "DstpK_D0pip_Kspipi_yield_Bplus_DD":                 [0, 20000],
    "Dstppi_D0pip_Kspipi_misID_PartReco_yield_Bplus_DD": [0, 20000],
    "Dst0K_D0gamma_Kspipi_yield_Bplus_DD"              : [0, 20000],
    "Combinatorial_yield_Bplus_DK_DD":                   [0, 50000],
    "Dpi_Kspipi_yield_Bplus_DD":                         [0,400000],
    "Dst0pi_D0pi0_Kspipi_yield_Bplus_DD":                [0,400000],
    "Dst0pi_D0gamma_Kspipi_yield_Bplus_DD":              [0,400000],
    "Combinatorial_yield_Bplus_DPI_DD":                  [0,400000],
    "Combinatorial_c_DK_DD":                       [-0.01,-0.0005],
}

###### define which free fit parameters is applied to which "real" variables
dict_shared_parameters = {
    "ratio_BR_DK_to_Dpi" : [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "ratio_BR_DK_to_Dpi"]
    ],
    "signal_mean_DK": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_m0"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "gauss_mean"],
    ],
    "signal_mean_Dpi": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_m0"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "gauss_mean"],
    ],
    "signal_width_DK_DD": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_sigmaL"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_sigmaR"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "gauss_sigma"],
    ],
    "signal_width_Dpi_DD": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_sigmaL"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_sigmaR"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "gauss_sigma"],
    ],
    "Dst0K_D0pi0_Kspipi_yield_Bplus_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "mass", "yield_Bplus" ],
    ],
    "DstpK_D0pip_Kspipi_yield_Bplus_DD": [
        ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "mass", "yield_Bplus" ],
    ],
    "Dstppi_D0pip_Kspipi_misID_PartReco_yield_Bplus_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dstppi_D0pip_Kspipi_misID_PartReco", "mass", "yield_Bplus"],
    ],
    "Dst0K_D0gamma_Kspipi_yield_Bplus_DD": [
        ["CB2DK_D2KSPIPI_DD", "Dst0K_D0gamma_Kspipi", "mass", "yield_Bplus"],
    ],
    "Combinatorial_yield_Bplus_DK_DD": [
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "mass", "yield_Bplus"  ],
    ],
    "Dpi_Kspipi_yield_Bplus_DD": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus"  ],
    ],
    "Dst0pi_D0pi0_Kspipi_yield_Bplus_DD": [ 
        ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0pi0_Kspipi", "mass", "yield_Bplus"  ],
    ],
    "Dst0pi_D0gamma_Kspipi_yield_Bplus_DD": [ 
        ["CB2DPI_D2KSPIPI_DD", "Dst0pi_D0gamma_Kspipi", "mass", "yield_Bplus"  ],
    ],
    "Combinatorial_yield_Bplus_DPI_DD": [
        ["CB2DPI_D2KSPIPI_DD", "Combinatorial", "mass", "yield_Bplus"  ],
    ],
    "Combinatorial_c_DK_DD": [
        ["CB2DK_D2KSPIPI_DD", "Combinatorial", "mass", "c"],
    ],
}

# https://gitlab.cern.ch/lhcb-b2oc/analyses/GGSZ-B2Dh/-/blob/master/common_inputs/pid_efficiencies/pid_4.0_Run1and2.settings.txt?ref_type=heads
# https://gitlab.cern.ch/lhcb-b2oc/analyses/GGSZ-B2Dh/-/blob/master/common_inputs/selection_efficiencies/selection_eff.settings.txt?ref_type=heads
# 0.97*942.908711914/(0.86*928.554348882)

################################## compute ratio of efficiencies etc to constrain parameters

############ constraining the DK yields to the Dpi one
ratio_DK_to_Dpi        = total_eff["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"]/total_eff["MC_Bu_D0pi_KSpipi"]["CB2DPI_D2KSPIPI_DD"]
############ constraining the misID Dpi to its non-misID counterpart
ratio_Dpi_misID_to_Dpi = total_eff["MC_Bu_D0pi_KSpipi"]["CB2DK_D2KSPIPI_DD"]/total_eff["MC_Bu_D0pi_KSpipi"]["CB2DPI_D2KSPIPI_DD"]
############ constraining the misID DK to its non-misID counterpart
ratio_DK_misID_to_DK   = total_eff["MC_Bu_D0K_KSpipi"]["CB2DPI_D2KSPIPI_DD"]/total_eff["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"]
        
######### define which real variable is constrained from which other var
# (only implemented with multiplicative factor)
dict_constrained_parameters = [
    ## constrain DK from Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi", "mass", "yield_Bplus"], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus",
                                                                   [ "SHARED_THROUGH_CHANNELS", "parameters", "mass", "ratio_BR_DK_to_Dpi"   , ratio_DK_to_Dpi ] ] ],
    ## constrain misID in DK from goodID in Dpi
    [ ["CB2DK_D2KSPIPI_DD" , "Dpi_Kspipi_misID", "mass", "yield_Bplus" ], ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus" , ratio_Dpi_misID_to_Dpi ] ],
    ## constrain misID DPI from goodID in DK
    [ ["CB2DPI_D2KSPIPI_DD", "DK_Kspipi_misID" , "mass", "yield_Bplus"], ["CB2DK_D2KSPIPI_DD" , "DK_Kspipi" , "mass", "yield_Bplus"  , ratio_DK_misID_to_DK ] ],
]
# print(json.dumps(dict_constrained_parameters,indent=4))


ratio_Dst0K_to_DstpK = Measurement(1,0.1)
######### define which real variable is constrained from which other var
# (only implemented with multiplicative factor)
dict_gaussian_constraints = [
        [ ["CB2DK_D2KSPIPI_DD", "DstpK_D0pip_Kspipi", "mass", "yield_Bplus" ], ["CB2DK_D2KSPIPI_DD", "Dst0K_D0pi0_Kspipi", "mass", "yield_Bplus", ratio_Dst0K_to_DstpK.value, ratio_Dst0K_to_DstpK.error] ],
]
# print(json.dumps(dict_gaussian_constraints,indent=4))



NLL = NLLComputation(start_values,
                     limit_values,
                     dict_shared_parameters,
                     dict_constrained_parameters,
                     dict_gaussian_constraints,
                     list_channels,
                     input_variables,
                     ntuples["SDATA"])

parameters_to_fit = NLL.parameters_to_fit

preFit_list_variables = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=list(start_values.values()),
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)



####### and now the mass pdfs
mass_pdfs_values = {}
for channel in list_channels:
    mass_pdfs_values[channel] = ntuples["SDATA"][channel].draw_combined_mass_pdfs(
        tf_Bmass_vec,
        preFit_list_variables
    )
    mplhep.histplot(np.histogram(ntuples["SDATA"][channel].Bu_M["both"],
                                 bins=nbins,
                                 range=B_mass_range),
                    label=ntuples["SDATA"][channel].channel.tex+" data",
                    **kwargs_data
                    )
    plt.plot(Bmass_vec,
             mass_scaling*mass_pdfs_values[channel]["both"]["total_mass_pdf"],
             label="Total"
             )
    for comp in ntuples["SDATA"][channel].components:
        plt.plot(
            Bmass_vec,
            mass_pdfs_values[channel]["both"][comp[0]]*mass_scaling,linestyle="--",
            label=components_tex[comp[0]]+"\n"+comp[1]
        )
        pass
    plt.xlabel("Constrained $DK$ mass")
    plt.ylabel(f"Events / ({round(mass_scaling)} MeV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/preFit/{channel}_Bplusminus_mass_distribution.png")
    plt.close("all")
    pass



######
@tf.function
def nll(x):
    return NLL.get_total_mass_nll(x) # , tensor_to_fit) something


x = tf.cast(list(start_values.values()),tf.float64)
print("start computing")
print("test nll(x) : ", nll(x))

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
###########i have to loop over the entries if this dict to set the pandas df myself
corr_array = np.zeros(len(parameters_to_fit)*len(parameters_to_fit)).reshape(len(parameters_to_fit),len(parameters_to_fit))
cov_array = np.zeros(len(parameters_to_fit)*len(parameters_to_fit)).reshape(len(parameters_to_fit),len(parameters_to_fit))
for i in range(len(parameters_to_fit)):
    corr_array[i] = corr[parameters_to_fit[i]]
    cov_array[i]  = cov[parameters_to_fit[i]]
    pass

pd_corr = pd.DataFrame(corr_array,index=parameters_to_fit,columns=parameters_to_fit)
pd_cov  = pd.DataFrame(cov_array,index=parameters_to_fit,columns=parameters_to_fit)
for i in range(len(means)):
    print(f'{parameters_to_fit[i]:<50}', ": ", means[i] ," +- ", errors[i])
    pass

# print("Means   ", means)
# print("Errors  ", errors)
means_results = dict(zip(parameters_to_fit,means))
errors_results = dict(zip(parameters_to_fit,errors))

print(json.dumps(means_results,indent=4))
print(json.dumps(errors_results,indent=4))


if (mg.valid==True):
    with open(f"{plot_dir}/means_results.json", "w") as f:
        json.dump(means_results, f, indent=4)
        pass

    with open(f"{plot_dir}/errors_results.json", "w") as f:
        json.dump(errors_results, f, indent=4)
        pass

    pass

import seaborn as sns
fig = plt.figure(figsize=(30, 24))  
sns.heatmap(pd_cov, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig(f"{plot_dir}/covariance.png")
plt.close("all")

import seaborn as sns
fig = plt.figure(figsize=(30, 24))  
sns.heatmap(pd_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig(f"{plot_dir}/correlation.png")
plt.close("all")


postFit_list_variables = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=means,
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)


####### and now the mass pdfs
postFit_mass_pdfs_values = {}
for channel in list_channels:
    postFit_mass_pdfs_values[channel] = ntuples["SDATA"][channel].draw_combined_mass_pdfs(
        tf_Bmass_vec,
        postFit_list_variables
    )
    mplhep.histplot(np.histogram(ntuples["SDATA"][channel].Bu_M["both"],
                                 bins=nbins,
                                 range=B_mass_range),
                    label=ntuples["SDATA"][channel].channel.tex,
                    **kwargs_data
                    )
    plt.plot(Bmass_vec,
             mass_scaling*postFit_mass_pdfs_values[channel]["both"]["total_mass_pdf"],
             label="Total"
             )
    for comp in ntuples["SDATA"][channel].components:
        plt.plot(
            Bmass_vec,
            postFit_mass_pdfs_values[channel]["both"][comp[0]]*mass_scaling,linestyle="--",
            label=components_tex[comp[0]]+"\n"+comp[1]
        )
        pass
    plt.xlabel(ntuples["SDATA"][channel].variable_to_fit)
    plt.ylabel(f"Events / ({round(mass_scaling)} MeV)")
    plt.title(rf"{channel} $B^\pm$")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{channel}_Bplusminus_mass_distribution.png")
    plt.savefig(f"{plot_dir}/{channel}_Bplusminus_mass_distribution.pdf")
    plt.close("all")
    pass

        
