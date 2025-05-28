from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
import uproot as up
import pandas as pd
import json
import shutil
import argparse
# from numpy import trapz
from scipy.integrate import trapz
import scipy.optimize as opt

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_test import *
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from plothist import plot_hist, make_hist 
from tfpcbpggsz.ulti import get_xy_xi, deg_to_rad, rad_to_deg

from tfpcbpggsz.Includes.VARDICT_DALITZ import VARDICT, varDict
from tfpcbpggsz.Includes.common_classes import DICT_VARIABLES_TEX, zp_p_tex  ,zm_pp_tex, Ntuple

from tfpcbpggsz.core import *
from tfpcbpggsz.Includes.functions import *
from tfpcbpggsz.amp_masses import *
Kspipi = PyD0ToKspipi2018()
Kspipi.init()

from tfpcbpggsz.amp_up import *
Kspipi_up = PyD0ToKSpipi2018()
Kspipi_up.init()

import mplhep

import time

"""Default configuration options for `matplotlib`"""
from matplotlib import rc, rcParams
# rc('text', usetex=True)
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
    "histtype"    : 'errorbar',
    "linestyle"   : "None",
    "color"       : "black",
    # "markersize" : 2,
    # "capsize" : 1.5,    
}
mplhep.style.use("LHCb2")
kwargs_data = {
    "histtype"  : 'errorbar',
    "linestyle" : "None",
    "color"     : "black"
}

parser = argparse.ArgumentParser(
                    prog='02_Gen_and_Fit_Toys.py',
                    description='Fit generated events',
                    epilog='Blabla at bottom')
parser.add_argument('--NFreeCoeff')      # option that takes a value
parser.add_argument('-n', '--numbernormalisation')      # option that takes a value
parser.add_argument('--date')      # option that takes a value
args = parser.parse_args()
N_Free_coeff = int(args.NFreeCoeff) # "Legendre_2_2"
numbernormalisation  = args.numbernormalisation # "Legendre_2_2"
date  = args.date # "Legendre_2_2"


# Efficiency_shape = "Flat"
Efficiency_shape_gen = "Legendre_5_5"
Efficiency_shape_fit = Efficiency_shape_gen
if (Efficiency_shape_gen not in EFFICIENCY_SHAPES):
    print("Unknown efficiency ----- exit")
    exit()
    pass
if (Efficiency_shape_fit not in EFFICIENCY_SHAPES):
    print("Unknown efficiency ----- exit")
    exit()
    pass

print(f"Generation efficiency shape is {Efficiency_shape_gen}")
print(f"Fit efficiency shape is {Efficiency_shape_fit}")
print(f"  with {N_Free_coeff} free coefficients in the fit")
# 

########## CONDOR
with open("job_id.txt") as f:
    job_id = int(np.loadtxt(f))
    pass
plot_dir=f"/shared/scratch/rj23972/safety_net/tfpcbpggsz/canorman_Efficiency/{date}/03_Gen_and_Fit_Toys_FullEff_{N_Free_coeff}FreeCoeff_{numbernormalisation}"
# plot_dir=f"./2025_02_07/toy_results_{Efficiency_shape}"
# condor_id=2
# while os.path.isdir(f'{plot_dir}/{condor_id}'):
#     condor_id+=1
#     pass
# plot_dir = f'{plot_dir}/{condor_id}'
#########

# ####### NOT CONDOR
# plot_dir=f"./2025_02_20/toy_results_{Efficiency_shape_gen}_{Efficiency_shape_fit}_FullEff"
# job_id=1
# while os.path.isdir(f'{plot_dir}/study_{job_id}'):
#     job_id+=1
#     pass
# ######

plot_dir = f'{plot_dir}/study_{job_id}'
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(f"{plot_dir}/preFit/",exist_ok=True)
os.makedirs(f"{plot_dir}/generation/",exist_ok=True)
if (job_id < 10):
    shutil.copyfile('03_Gen_and_Fit_Toys_FullEff.py', f'{plot_dir}/03_Gen_and_Fit_Toys_FullEff.py')
    pass
pathname_output = plot_dir


time1 = time.time()
#Generating the B2DK signal 
pcgen = pcbpggsz_generator()

fixed_variables = dict(**VARDICT["SDATA"])
B2DK_mass_variables  = list(fixed_variables["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"]["mass"].values())
B2Dpi_mass_variables = list(fixed_variables["CB2DPI_D2KSPIPI_DD"]["Dpi_Kspipi"]["mass"].values())
B2Dpi_misID_mass_variables = list(fixed_variables["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"]["mass"].values())


####### LOAD THE RESULT OF THE EFFICIENCY FIT
with open(f"means_results_efficiency_shape.json") as f:
    means = json.load(f)
    pass

efficiency_variables = []
free_efficiency_coeffs_gen = ["c10","c20","c01", "c02", "c11", "c21", "c12", "c22", "c30", "c40", "c50", "c04", "c22", "c24", "c32", "c34", "c42", "c44"]
if (N_Free_coeff == 0):
    free_efficiency_coeffs_gen = []
    pass    
else:
    free_efficiency_coeffs_gen = free_efficiency_coeffs_gen[:N_Free_coeff+1]
    pass

for eff_var in DICT_NAME_COEFF[Efficiency_shape_gen]:
    if (f"{eff_var}" in free_efficiency_coeffs_gen):
        efficiency_variables.append(means[f"{eff_var}_DK_Kspipi_DD"])
    else:
        efficiency_variables.append(fixed_variables["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"]["Bplus"][eff_var])
    pass


B2DK_Bplus_efficiency_variables    = efficiency_variables
B2DK_Bminus_efficiency_variables   = efficiency_variables
B2Dpi_Bminus_efficiency_variables  = efficiency_variables
B2Dpi_Bplus_efficiency_variables   = efficiency_variables


print("B2DK_Bplus_efficiency_variables    :   ", B2DK_Bplus_efficiency_variables   )
print("B2DK_Bminus_efficiency_variables   :   ", B2DK_Bminus_efficiency_variables  )
print("B2Dpi_Bminus_efficiency_variables  :   ", B2Dpi_Bminus_efficiency_variables )
print("B2Dpi_Bplus_efficiency_variables   :   ", B2Dpi_Bplus_efficiency_variables  )

####################### some tools for plotting: 2D grid of invariant Dalitz masses
BDT_cut = 0.4
# min_mass = 5200
min_mass = 5080
max_mass = 5800
Bmass_vec = np.arange(min_mass, max_mass, 1)
tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)
####
min_dalitz = 0.4
max_dalitz = 3.0
nbins=100
m_Kspip_range = [min_dalitz, max_dalitz]
num_mass_values = 1000.
Dalitz_mass_vec = np.arange(min_dalitz, max_dalitz, (max_dalitz-min_dalitz)/num_mass_values )
tf_Dalitz_mass_vec = tf.cast(Dalitz_mass_vec, tf.float64)
####
BR_B2DK  = 3.63e-4
BR_B2Dpi = 4.68e-3
str_BDT_cut = "(BDT_output > "+str(BDT_cut)+")"
#### scaling for comparison histogram vs pdfs
mass_scaling   = (max_mass - min_mass) / float(nbins)
dalitz_scaling = (m_Kspip_range[1]-m_Kspip_range[0]) / float(nbins)
######## 2D grid
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
        ampD0bar[row][col] = np.negative(tmp_amps[1])
        pass
    pass




###
B_mass_range = [5080, 5800]
srd_range = [-1,1]

############## START THE FIT
components = {
    "CB2DK_D2KSPIPI_DD": [
        ["DK_Kspipi", "Cruijff+Gaussian"  , Efficiency_shape_fit],
        # ["Dpi_Kspipi_misID", "SumCBShape" , "Legendre_2_2"]
    ],
    "CB2DPI_D2KSPIPI_DD": [
        ["Dpi_Kspipi", "Cruijff+Gaussian" , Efficiency_shape_fit]
    ],
}
components_tex = {
    "DK_Kspipi" : r"$B^{\pm} \rightarrow D K^{\pm}$",
    "Dpi_Kspipi": r"$B^{\pm} \rightarrow D \pi^{\pm}$",
    "Dpi_Kspipi_misID": r"$B^{\pm} \rightarrow D \pi^{\pm}_{\pi\rightarrow K}$"
}

list_channels = ["CB2DK_D2KSPIPI_DD", "CB2DPI_D2KSPIPI_DD"]

############
yields = { # bper B sign
    "DK_Kspipi_DD"        : 6266, # 1000, # 10,  #  
    "Dpi_Kspipi_DD"       : 89941,# 1000, # 10,  #  
    "Dpi_Kspipi_DD_misID" : 10  , #  1500, # 
}

# GAMMA    = 90
# RB_DK    = 1
# DELTA_DK = 90
generation_parameters = {
    "CB2DK_D2KSPIPI_DD": {
        "DK_Kspipi": [
            yields["DK_Kspipi_DD"]    ,
            "Cruijff+Gaussian"        ,
            "B2Dh_D2Kspipi"           ,
            B2DK_mass_variables       ,
            [GAMMA, RB_DK, DELTAB_DK] ,
            B2DK_Bplus_efficiency_variables ,
            B2DK_Bminus_efficiency_variables
        ],
        # "Dpi_Kspipi_misID": [
        #     yields["Dpi_Kspipi_DD_misID"]    ,
        #     "SumCBShape"              ,
        #     "B2Dh_D2Kspipi"           ,
        #     B2Dpi_misID_mass_variables       ,
        #     [GAMMA, RB_DPI, DELTAB_DPI] ,
        # ]
    },
    "CB2DPI_D2KSPIPI_DD": {
        "Dpi_Kspipi": [
            yields["Dpi_Kspipi_DD"]  ,
            "Cruijff+Gaussian"                   ,
            "B2Dh_D2Kspipi"                ,
            B2Dpi_mass_variables     ,
            [GAMMA, RB_DPI, DELTAB_DPI]    ,
            B2Dpi_Bplus_efficiency_variables ,
            B2Dpi_Bminus_efficiency_variables
        ],
    },
}

ret_Bp_DK, ret_Bp_DK_mass = {}, {}
ret_Bm_DK, ret_Bm_DK_mass = {}, {}
p1_p_DK,p2_p_DK,p3_p_DK = {}, {}, {}
p1_m_DK,p2_m_DK,p3_m_DK = {}, {}, {}
m12_p_DK = {}
m13_p_DK = {}
m12_m_DK = {}
m13_m_DK = {}
srd_p_DK = {}
srd_m_DK = {}


############################ TRANSFORM THE TOYS INTO "DATA-LIKE" NTUPLES
basic_list_var = ["Bu_ID", "zp_p", "zm_pp", "m_Kspip", "m_Kspim"]
for particle in ["KS","pim","pip"]:
    for mom in ["PE", "PX", "PY", "PZ"]:
        basic_list_var += [f"{particle}_{mom}"]
        pass
    pass

variable_to_fit = {
    "CB2DK_D2KSPIPI_DD" : "Bu_constD0KSPV_M",
    "CB2DPI_D2KSPIPI_DD": "Bu_constD0KSPV_swapBachToPi_M",
}


for channel in list_channels:
    print("Start generating events for channel ", channel)
    ret_Bp_DK[channel], ret_Bp_DK_mass[channel] = {}, {}
    ret_Bm_DK[channel], ret_Bm_DK_mass[channel] = {}, {}
    p1_p_DK[channel],p2_p_DK[channel],p3_p_DK[channel] = {}, {}, {}
    p1_m_DK[channel],p2_m_DK[channel],p3_m_DK[channel] = {}, {}, {}
    m12_p_DK[channel] = {}
    m13_p_DK[channel] = {}
    m12_m_DK[channel] = {}
    m13_m_DK[channel] = {}
    srd_p_DK[channel] = {}
    srd_m_DK[channel] = {}
    for comp in components[channel]: # generation_parameters[channel].keys():
        gen_comp = comp[0]
        print("     component: ", gen_comp)
        print("                  ",generation_parameters[channel][gen_comp][0], " Bplus")
        ret_Bp_DK[channel][gen_comp], ret_Bp_DK_mass[channel][gen_comp] = pcgen.generate(
            generation_parameters[channel][gen_comp][0],
            type="b2dh",
            gamma=generation_parameters[channel][gen_comp][4][0],
            rb=generation_parameters[channel][gen_comp][4][1],
            dB=generation_parameters[channel][gen_comp][4][2],
            charge=1,
            generate_B_mass = True,
            B_mass_range    = B_mass_range,
            mass_shape_name = generation_parameters[channel][gen_comp][1],
            mass_variables  = generation_parameters[channel][gen_comp][3],
            efficiency_function  = Efficiency_shape_gen,
            efficiency_variables = generation_parameters[channel][gen_comp][5],
        )
        print("                  ",generation_parameters[channel][gen_comp][0], " Bminus")
        ret_Bm_DK[channel][gen_comp], ret_Bm_DK_mass[channel][gen_comp] = pcgen.generate(
            generation_parameters[channel][gen_comp][0],
            type="b2dh",
            gamma=generation_parameters[channel][gen_comp][4][0],
            rb=generation_parameters[channel][gen_comp][4][1],
            dB=generation_parameters[channel][gen_comp][4][2],
            charge=-1,
            generate_B_mass = True,
            B_mass_range    = B_mass_range,
            mass_shape_name = generation_parameters[channel][gen_comp][1],
            mass_variables  = generation_parameters[channel][gen_comp][3],
            efficiency_function  = Efficiency_shape_gen,
            efficiency_variables = generation_parameters[channel][gen_comp][6],
        )
        p1_p_DK[channel][gen_comp],p2_p_DK[channel][gen_comp],p3_p_DK[channel][gen_comp] = ret_Bp_DK[channel][gen_comp] # Ks, pi-, pi+
        p1_m_DK[channel][gen_comp],p2_m_DK[channel][gen_comp],p3_m_DK[channel][gen_comp] = ret_Bm_DK[channel][gen_comp] # Ks, pi-, pi+
        m12_p_DK[channel][gen_comp] = get_mass(p1_p_DK[channel][gen_comp],p2_p_DK[channel][gen_comp])
        m13_p_DK[channel][gen_comp] = get_mass(p1_p_DK[channel][gen_comp],p3_p_DK[channel][gen_comp])
        m12_m_DK[channel][gen_comp] = get_mass(p1_m_DK[channel][gen_comp],p2_m_DK[channel][gen_comp])
        m13_m_DK[channel][gen_comp] = get_mass(p1_m_DK[channel][gen_comp],p3_m_DK[channel][gen_comp])        
        srd_p_DK[channel][gen_comp] = phsp_to_srd(m12_p_DK[channel][gen_comp],m13_p_DK[channel][gen_comp])
        srd_m_DK[channel][gen_comp] = phsp_to_srd(m12_m_DK[channel][gen_comp],m13_m_DK[channel][gen_comp])
        pass
    pass


toy_data = {}
for channel in list_channels:
    toy_data[channel] = {
        "Bu_ID" : np.concatenate((
            [   np.ones(m13_p_DK[channel][gen_comp].shape) for gen_comp in generation_parameters[channel].keys() ] +
            [-1*np.ones(m13_m_DK[channel][gen_comp].shape) for gen_comp in generation_parameters[channel].keys() ]
        )),
        "zp_p"  : np.concatenate((
            [   srd_p_DK[channel][gen_comp][0] for gen_comp in generation_parameters[channel].keys() ] +
            [   srd_m_DK[channel][gen_comp][0] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "zm_pp" : np.concatenate((
            [   srd_p_DK[channel][gen_comp][1] for gen_comp in generation_parameters[channel].keys() ] +
            [   srd_m_DK[channel][gen_comp][1] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "m_Kspip" : np.concatenate((
            [   m13_p_DK[channel][gen_comp] for gen_comp in generation_parameters[channel].keys() ] +
            [   m13_m_DK[channel][gen_comp] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "m_Kspim" : np.concatenate((
            [   m12_p_DK[channel][gen_comp] for gen_comp in generation_parameters[channel].keys() ] +
            [   m12_m_DK[channel][gen_comp] for gen_comp in generation_parameters[channel].keys() ]
        )),
        variable_to_fit[channel]: np.concatenate((
            [   ret_Bp_DK_mass[channel][gen_comp] for gen_comp in generation_parameters[channel].keys() ] +
            [   ret_Bm_DK_mass[channel][gen_comp] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "KS_PE"    : np.concatenate((
            [   p1_p_DK[channel][gen_comp][:,0] for gen_comp in generation_parameters[channel].keys() ] +
            [   p1_m_DK[channel][gen_comp][:,0] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "KS_PX"    : np.concatenate((
            [   p1_p_DK[channel][gen_comp][:,1] for gen_comp in generation_parameters[channel].keys() ] +
            [   p1_m_DK[channel][gen_comp][:,1] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "KS_PY"    : np.concatenate((
            [   p1_p_DK[channel][gen_comp][:,2] for gen_comp in generation_parameters[channel].keys() ] +
            [   p1_m_DK[channel][gen_comp][:,2] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "KS_PZ"    : np.concatenate((
            [   p1_p_DK[channel][gen_comp][:,3] for gen_comp in generation_parameters[channel].keys() ] +
            [   p1_m_DK[channel][gen_comp][:,3] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pim_PE"   : np.concatenate((
            [   p2_p_DK[channel][gen_comp][:,0] for gen_comp in generation_parameters[channel].keys() ] +
            [   p2_m_DK[channel][gen_comp][:,0] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pim_PX"   : np.concatenate((
            [   p2_p_DK[channel][gen_comp][:,1] for gen_comp in generation_parameters[channel].keys() ] +
            [   p2_m_DK[channel][gen_comp][:,1] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pim_PY"   : np.concatenate((
            [   p2_p_DK[channel][gen_comp][:,2] for gen_comp in generation_parameters[channel].keys() ] +
            [   p2_m_DK[channel][gen_comp][:,2] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pim_PZ"   : np.concatenate((
            [   p2_p_DK[channel][gen_comp][:,3] for gen_comp in generation_parameters[channel].keys() ] +
            [   p2_m_DK[channel][gen_comp][:,3] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pip_PE"   : np.concatenate((
            [   p3_p_DK[channel][gen_comp][:,0] for gen_comp in generation_parameters[channel].keys() ] +
            [   p3_m_DK[channel][gen_comp][:,0] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pip_PX"   : np.concatenate((
            [   p3_p_DK[channel][gen_comp][:,1] for gen_comp in generation_parameters[channel].keys() ] +
            [   p3_m_DK[channel][gen_comp][:,1] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pip_PY"   : np.concatenate((
            [   p3_p_DK[channel][gen_comp][:,2] for gen_comp in generation_parameters[channel].keys() ] +
            [   p3_m_DK[channel][gen_comp][:,2] for gen_comp in generation_parameters[channel].keys() ]
        )),
        "pip_PZ"   : np.concatenate((
            [   p3_p_DK[channel][gen_comp][:,3] for gen_comp in generation_parameters[channel].keys() ] +
            [   p3_m_DK[channel][gen_comp][:,3] for gen_comp in generation_parameters[channel].keys() ]
        )),
    }    
    pass



uproot_data = {}
print(" =============    STORE TOY AT")
print(f"{pathname_output}/outfile_{job_id}.root")
outfile = up.recreate(f"{pathname_output}/outfile_{job_id}.root")
for channel in list_channels:
    uproot_data[channel] = {}
    # for key in toy_data[channel].keys():
    #     print(len(toy_data[channel][key]))
    uproot_data[channel]  = pd.DataFrame.from_dict(toy_data[channel])
    outfile[channel]  = uproot_data[channel]
    pass



print("Some gen level plots")
for channel in list_channels:
    mplhep.histplot(
        np.histogram(
            uproot_data[channel].query("Bu_ID>0")[variable_to_fit[channel]],
            bins=nbins,
            range=B_mass_range
        ),
        label="Bplus"
    )
    mplhep.histplot(
        np.histogram(
            uproot_data[channel].query("Bu_ID<0")[variable_to_fit[channel]],
            bins=nbins,
            range=B_mass_range
        ),
    label="Bminus"
    )
    plt.title(f"generation {channel}")
    plt.legend()
    plt.xlabel(f"invariant mass")
    plt.tight_layout()
    if (job_id < 10):
        plt.savefig(f"{plot_dir}/generation/{channel}_invariant_mass.png")
        pass
    plt.close("all")
    ###### Bplus srd
    mplhep.hist2dplot(
        np.histogram2d(
            uproot_data[channel].query("Bu_ID>0")["zp_p" ],
            uproot_data[channel].query("Bu_ID>0")["zm_pp"],
            bins=[nbins,nbins],
            range= [srd_range, srd_range])
    )
    plt.title(f"generation {channel}")
    plt.xlabel(zp_p_tex)
    plt.ylabel(zm_pp_tex)
    plt.tight_layout()
    if (job_id < 10):
        plt.savefig(f"{plot_dir}/generation/{channel}_Bplus_srd_generation.png")
        pass
    plt.close("all")
    ###### Bminus srd    
    mplhep.hist2dplot(
        np.histogram2d(
            uproot_data[channel].query("Bu_ID<0")["zp_p" ],
            uproot_data[channel].query("Bu_ID<0")["zm_pp"],
            bins=[nbins,nbins],
            range= [srd_range, srd_range])
    )
    plt.title(f"generation {channel}")
    plt.xlabel(zp_p_tex)
    plt.ylabel(zm_pp_tex)
    plt.tight_layout()
    if (job_id < 10):
        plt.savefig(f"{plot_dir}/generation/{channel}_Bminus_srd_generation.png")
        pass
    plt.close("all")
    ###### Bplus dalitz    
    mplhep.hist2dplot(
        np.histogram2d(
            uproot_data[channel].query("Bu_ID>0")["m_Kspip" ],
            uproot_data[channel].query("Bu_ID>0")["m_Kspim"],
            bins=[nbins,nbins],
            range= [m_Kspip_range,m_Kspip_range])
    )
    plt.title(f"generation {channel}")
    plt.xlabel("Ks pi+")
    plt.ylabel("Ks pi-")
    plt.tight_layout()
    if (job_id < 10):
        plt.savefig(f"{plot_dir}/generation/{channel}_Bplus_dalitz_generation.png")
        pass
    plt.close("all")
    ###### Bminus dalitz    
    mplhep.hist2dplot(
        np.histogram2d(
            uproot_data[channel].query("Bu_ID<0")["m_Kspip" ],
            uproot_data[channel].query("Bu_ID<0")["m_Kspim"],
            bins=[nbins,nbins],
            range= [m_Kspip_range,m_Kspip_range])
    )
    plt.title(f"generation {channel}")
    plt.xlabel("Ks pi+")
    plt.ylabel("Ks pi-")
    plt.tight_layout()
    if (job_id < 10):
        plt.savefig(f"{plot_dir}/generation/{channel}_Bminus_dalitz_generation.png")
        pass
    plt.close("all")
    pass

############# NORMALISATION

#PHSP
yield_normalisation = int(numbernormalisation)
phsp = PhaseSpaceGenerator().generate
phsp_p, phsp_m = phsp(yield_normalisation), phsp(yield_normalisation)

phsp_p1, phsp_p2, phsp_p3 = phsp_p
phsp_m1, phsp_m2, phsp_m3 = phsp_m

phsp_m12_p = get_mass(phsp_p1,phsp_p2)
phsp_m13_p = get_mass(phsp_p1,phsp_p3)
phsp_m12_m = get_mass(phsp_m1,phsp_m2)
phsp_m13_m = get_mass(phsp_m1,phsp_m3)

phsp_srd_p = phsp_to_srd(phsp_m12_p,phsp_m13_p)
phsp_srd_m = phsp_to_srd(phsp_m12_m,phsp_m13_m)

amp_phsp_p, ampbar_phsp_p = pcgen.amp(phsp_p), pcgen.ampbar(phsp_p)
amp_phsp_m, ampbar_phsp_m = pcgen.amp(phsp_m), pcgen.ampbar(phsp_m)

toy_phsp_DK = {
    "ampD0" : {
        "Bplus"  : amp_phsp_p,
        "Bminus" : amp_phsp_m,
    },
    "ampD0bar" : {
        "Bplus"  : ampbar_phsp_p,
        "Bminus" : ampbar_phsp_m,
    },
    "zp_p" : {
        "Bplus"  : phsp_srd_p[0],
        "Bminus" : phsp_srd_m[0],
    },
    "zm_pp" : {
        "Bplus"  : phsp_srd_p[1],
        "Bminus" : phsp_srd_m[1],
    },
}

toy_phsp_Dpi = {
    "ampD0" : {
        "Bplus"  : amp_phsp_p,
        "Bminus" : amp_phsp_m,
    },
    "ampD0bar" : {
        "Bplus"  : ampbar_phsp_p,
        "Bminus" : ampbar_phsp_m,
    },
    "zp_p" : {
        "Bplus"  : phsp_srd_p[0],
        "Bminus" : phsp_srd_m[0],
    },
    "zm_pp" : {
        "Bplus"  : phsp_srd_p[1],
        "Bminus" : phsp_srd_m[1],
    },
}

    
# mplhep.hist2dplot(
#     np.histogram2d(
#         toy_phsp_DK["zp_p" ]["Bplus"],
#         toy_phsp_DK["zm_pp"]["Bplus"],
#         bins=[nbins,nbins],
#         range= [srd_range, srd_range])
# )
# plt.title(r"$B^{+} \rightarrow [K_S\pi^+\pi^-]_D K^{+}$")
# plt.xlabel(zp_p_tex)
# plt.ylabel(zm_pp_tex)
# plt.tight_layout()
# plt.savefig(f"{plot_dir}/Bplus_PHSP_srd_generation.png")
# plt.close("all")

# mplhep.hist2dplot(
#     np.histogram2d(
#         toy_phsp_DK["zp_p" ]["Bminus"],
#         toy_phsp_DK["zm_pp"]["Bminus"],
#         bins=[nbins,nbins],
#         range= [srd_range, srd_range])
# )
# plt.title(r"$B^{-} \rightarrow [K_S\pi^+\pi^-]_D K^{-}$")
# plt.xlabel(zp_p_tex)
# plt.ylabel(zm_pp_tex)
# plt.tight_layout()
# plt.savefig(f"{plot_dir}/Bminus_PHSP_srd_generation.png")
# plt.close("all")


####### 
input_variables = {}
for channel in list_channels:
    input_variables[channel]       = {} ## loop over channel
    for comp in components[channel]: ## loop over components
        input_variables[channel][comp[0]] = VARDICT["SDATA"][channel][comp[0]]
        print(comp[0])
        pass
    pass

input_variables["SHARED_THROUGH_CHANNELS"] = VARDICT["SDATA"]["SHARED_THROUGH_CHANNELS"]



ntuples = {}
for channel in list_channels:
    ntuples[channel]  = Ntuple(
        "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow",channel,"YRUN2", "MagAll"
    )
    index_channel = list(input_variables.keys()).index(channel)
    ntuples[channel].initialise_fit(components[channel], index_channel)
    list_var = [ntuples[channel].variable_to_fit] + basic_list_var
    ntuples[channel].store_events(
        f"{pathname_output}/outfile_{job_id}.root:{channel}",
        list_var,
        None,
        Kspipi_up
    )
    pass

###########
######## define starting values of the variables in the fit
inputs = {
    "gamma" : deg_to_rad(GAMMA), "rb" : RB_DK, "dB" : deg_to_rad(DELTAB_DK),
    "rb_dpi": RB_DPI, "dB_dpi": deg_to_rad(DELTAB_DPI)
}
inputs["xplus"], inputs["yplus"], inputs["xminus"], inputs["yminus"], inputs["xxi"], inputs["yxi"] = get_xy_xi(
    (inputs["gamma"], inputs["rb"], inputs["dB"], inputs["rb_dpi"], inputs["dB_dpi"])
)

start_values = {
    "yield_Bplus_DK"  :                          yields["DK_Kspipi_DD"],
    "yield_Bminus_DK" :                          yields["DK_Kspipi_DD"],
    "signal_mean_DK":                            varDict['signal_mean']+50,
    "signal_width_DK":                           varDict['sigma_dk_DD']+50,
    "yield_Bplus_Dpi"  :                          yields["Dpi_Kspipi_DD"],
    "yield_Bminus_Dpi" :                          yields["Dpi_Kspipi_DD"],
    "signal_mean_Dpi":                            varDict['signal_mean']+50,
    "signal_width_Dpi":                           varDict['sigma_dk_DD']+50,
    # "yield_Bplus_Dpi_misID"  :                      400.,
    # "yield_Bminus_Dpi_misID" :                      400.,
    "xplus"                                        : inputs["xplus" ], #  0.50,
    "yplus"                                        : inputs["yplus" ], # -0.00,
    "xminus"                                       : inputs["xminus"], #  0.50,
    "yminus"                                       : inputs["yminus"], # -0.00,
    "xxi"                                          : inputs["xxi"   ], #  0.50,
    "yxi"                                          : inputs["yxi"   ], # -0.00,
    # "c01_DK_Kspipi_DD_Bplus"                       : 0., # -0.00,
}

####### define limits 
limit_values = {
    "yield_Bplus_DK"  :                            [0, 2*yields["DK_Kspipi_DD"]],
    "yield_Bminus_DK" :                            [0, 2*yields["DK_Kspipi_DD"]],
    "signal_mean_DK":                              [5080,5800],
    "signal_width_DK":                             [1,100],
    "yield_Bplus_Dpi"  :                            [0, 200000.],
    "yield_Bminus_Dpi" :                            [0, 200000.],
    "signal_mean_Dpi":                              [5080,5800],
    "signal_width_Dpi":                             [1,100],
    # "yield_Bplus_Dpi_misID"  :                            [0, 10000.],
    # "yield_Bminus_Dpi_misID" :                            [0, 10000.],
    "xplus"                                        : [-1., 1.],
    "yplus"                                        : [-1., 1.],
    "xminus"                                       : [-1., 1.],
    "yminus"                                       : [-1., 1.],
    "xxi"                                          : [-1., 1.],
    "yxi"                                          : [-1., 1.],
    # "c01_DK_Kspipi_DD_Bplus"                       : [-10.,10.],
}

###### define which free fit parameters is applied to which "real" variables
dict_shared_parameters = {
    "yield_Bplus_DK"  :  [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "yield_Bplus"]
    ],
    "yield_Bminus_DK" :  [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "yield_Bminus"]
    ],
    "signal_mean_DK": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_m0"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "gauss_mean"],
    ],
    "signal_width_DK": [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_sigmaL"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "cruij_sigmaR"  ],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "mass", "gauss_sigma"],
    ],
    "yield_Bplus_Dpi"  :  [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bplus"]
    ],
    "yield_Bminus_Dpi" :  [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "yield_Bminus"]
    ],
    "signal_mean_Dpi": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_m0"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "gauss_mean"],
    ],
    "signal_width_Dpi": [
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_sigmaL"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "cruij_sigmaR"  ],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "mass", "gauss_sigma"],
    ],
    # "yield_Bplus_Dpi_misID"  :  [
    #     ["CB2DK_D2KSPIPI_DD", "Dpi_Kspipi_misID", "mass", "yield_Bplus"]
    # ],
    # "yield_Bminus_Dpi_misID" :  [
    #     ["CB2DK_D2KSPIPI_DD", "Dpi_Kspipi_misID", "mass", "yield_Bminus"]
    # ],
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
    "xxi": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "xxi"],
    ],
    "yxi": [
        ["SHARED_THROUGH_CHANNELS", "parameters", "mass", "yxi"],
    ],
    # "c01_DK_Kspipi_DD_Bplus":  [
    #     ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "Bplus", "c01"],
    #     ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "Bplus", "c01"],
    # ],
}

free_efficiency_coeffs_fit = ["c10","c20","c01", "c02", "c11", "c21", "c12", "c22", "c30", "c40", "c50", "c04", "c22", "c24", "c32", "c34", "c42", "c44"]
if (N_Free_coeff == 0):
    free_efficiency_coeffs_fit = []
    pass    
else:
    free_efficiency_coeffs_fit = free_efficiency_coeffs_fit[:N_Free_coeff+1]
    pass

# 
for coeff in free_efficiency_coeffs_fit:
    start_values[f"{coeff}_DK_Kspipi_DD_Bplus"] =  0. # -0.00,
    limit_values[f"{coeff}_DK_Kspipi_DD_Bplus"] =  [-10.,10.]
    dict_shared_parameters[f"{coeff}_DK_Kspipi_DD_Bplus"] =   [
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "Bplus", f"{coeff}"],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "Bplus", f"{coeff}"],
        ["CB2DK_D2KSPIPI_DD", "DK_Kspipi", "Bminus", f"{coeff}"],
        ["CB2DPI_D2KSPIPI_DD", "Dpi_Kspipi", "Bminus", f"{coeff}"],
    ]
    pass
    


####### OF COURSE THIS IS WRONG
### WE'LL UPDATE THIS ONCE WE HAVE FINAL NUMERS
ratio_Dpi_misID_to_Dpi = 0. # float(yields["Dpi_Kspipi_DD_misID"]) / float(yields["Dpi_Kspipi_DD"])

dict_constrained_parameters = [
    ## constrain misID DPI from goodID in DK
    # [ ["CB2DK_D2KSPIPI_DD", "Dpi_Kspipi_misID" , "mass", "yield_Bplus"], ["CB2DPI_D2KSPIPI_DD" , "Dpi_Kspipi" , "mass", "yield_Bplus",  ratio_Dpi_misID_to_Dpi] ],
    # [ ["CB2DK_D2KSPIPI_DD", "Dpi_Kspipi_misID" , "mass", "yield_Bminus"], ["CB2DPI_D2KSPIPI_DD" , "Dpi_Kspipi" , "mass", "yield_Bminus",  ratio_Dpi_misID_to_Dpi] ],
]
dict_gaussian_constraints   = []



NLL = NLLComputation(
    start_values,
    limit_values,
    dict_shared_parameters,
    dict_constrained_parameters,
    dict_gaussian_constraints,
    list_channels,
    input_variables,
    ntuples
)

parameters_to_fit = NLL.parameters_to_fit

preFit_list_variables = ntuples["CB2DK_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=list(start_values.values()),
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)
preFit_list_variables = ntuples["CB2DPI_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=list(start_values.values()),
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)


ntuples["CB2DK_D2KSPIPI_DD"].define_dalitz_pdfs(
    toy_phsp_DK["ampD0"], toy_phsp_DK["ampD0bar"], toy_phsp_DK["zp_p"], toy_phsp_DK["zm_pp"]
)
ntuples["CB2DPI_D2KSPIPI_DD"].define_dalitz_pdfs(
    toy_phsp_Dpi["ampD0"], toy_phsp_Dpi["ampD0bar"], toy_phsp_Dpi["zp_p"], toy_phsp_Dpi["zm_pp"]
)


# ### get the efficiency KDE for plotting purposes
# from scipy import stats
# positions = np.vstack([Dalitz_Kspip_mat.ravel(), Dalitz_Kspim_mat.ravel()])
# values = np.vstack([phsp_m13_p[:100000], phsp_m12_p[:100000]])
# kernel = stats.gaussian_kde(values)
# print("This will take a bit of time depending on the number of normalisation events.")
# print(" Less than a minute for 100k events.")
# Z = np.reshape(kernel(positions).T, Dalitz_Kspip_mat.shape)

# fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#           extent=[0.4, 3., 0.4, 3.0])
# ax.plot(phsp_m13_p[:100000], phsp_m12_p[:100000], 'k.', markersize=2)
# ax.set_xlim([0.4, 3.])
# ax.set_ylim([0.4, 3.])
# plt.savefig(f"{plot_dir}/efficiency.png")
# plt.close("all")



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
    # tmp_data = {
    #     "Bplus" : ntuple.Bplus_data ,
    #     "Bminus": ntuple.Bminus_data,
    # }
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
            ## if comp is DK or DPI, the pdf is a function of the amplitudes
            print(comp_pdf.component)
            tmp_pdf_values = comp_pdf.pdf(ampD0, ampD0bar, zp_p, zm_pp)
            print(np.max(tmp_pdf_values))
            #### multiply each pdf by its yield
            # index_yields  = INDEX_YIELDS[Bsign]
            # comp_yield    = variables[ntuple.i_c][i][2][index_yields]
            tmp_pdf_values = tmp_pdf_values # *Z/np.mean(Z)
            # print(np.mean(Z))
            tmp_pdf_values = tmp_pdf_values*dalitz_scaling*dalitz_scaling # *comp_yield
            pdfs_values[Bsign][comp_pdf.component]     = tmp_pdf_values
            pdfs_values[Bsign]["total_pdf"]           += tmp_pdf_values
            pass # loop comps
        pass # loop signs
    return pdfs_values

dalitz_pdfs_values = {}
for channel in list_channels:
    #### compute the pdfs
    dalitz_pdfs_values[channel] = draw_dalitz_pdf(
        ntuples[channel],
        ampD0   ,
        ampD0bar,
        zp_p    ,
        zm_pp   ,
        variables=preFit_list_variables
    )
    pass


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
            plt.title(f"{channel} {Bsign}")
            cbar = plt.colorbar(cs)
            # cs.cmap.set_over('red')
            # cs.cmap.set_under('white')
            # cs.changed()
            plt.tight_layout()
            if (job_id < 10):
                plt.savefig(f"{plot_dir}/preFit/{channel}_{Bsign}_{comp}.png")
                pass
            plt.close("all")
            pass
        pass
    pass


def draw_projection_pdf(ntuple, pdfs_values, variables = None):
    """
    This function integrates separately over the two dimensions
    to get both projections of the Dalitz pdfs.
    """
    if (variables == None):
        variables = ntuple.list_variables
        pass
    else:
        ntuple.initialise_fixed_pdfs(variables)
        pass
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
            index_yields  = INDEX_YIELDS[Bsign]
            comp_yield    = variables[ntuple.i_c][i][2][index_yields]
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
                )) # *float(nbins)
                #### Kspim
                Kspim_projection[Bsign][comp_pdf.component][i_Kspi] = float(tf.reduce_mean(
                    tmp_numpy_pdfs[i_Kspi]
                )) # *float(nbins)
                # Kspim_projection[Bsign]["total_pdf"][i_Kspi] += float(np.mean(
                #     tmp_numpy_pdfs[i_Kspi]
                # )) # *float(nbins)
                # if (Kspip_projection[Bsign][comp_pdf.component][i_Kspip] > 0):
                #     print(Kspip_projection[Bsign][comp_pdf.component][i_Kspip])
                pass # loop axes
            Kspip_projection[Bsign][comp_pdf.component] = comp_yield*norm_distribution(
                Dalitz_mass_vec,
                Kspip_projection[Bsign][comp_pdf.component]
            ) # *float(nbins)
            Kspip_projection[Bsign]["total_pdf"] += Kspip_projection[Bsign][comp_pdf.component]
            Kspim_projection[Bsign][comp_pdf.component] = comp_yield*norm_distribution(
                Dalitz_mass_vec,
                Kspim_projection[Bsign][comp_pdf.component]
            ) # *float(nbins)
            Kspim_projection[Bsign]["total_pdf"] += Kspim_projection[Bsign][comp_pdf.component]
            pass # loop comps
        pass # loop signs
    return Kspip_projection, Kspim_projection

def get_pull_projections(data, Kspip_proj = None, Kspim_proj = None):
    histo_pulls = {}
    proj  = {
        "Kspip" : Kspip_proj,
        "Kspim" : Kspim_proj,
    }
    for Kspi in ["Kspip", "Kspim"]:
        h_data_to_fit = np.histogram(tmp_data[f"m_{Kspi}"], bins=nbins, range=m_Kspip_range)
        bin_edges     = h_data_to_fit[1]
        bin_widths    = bin_edges[1:] - bin_edges[:-1]
        bin_centers   = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        # fitted_yield = result_MC[resonance].params[yield_MC[resonance]]['value']
        binned_integrals = []
        # dx = Dalitz_mass_vec[1] - Dalitz_mass_vec[0]
        for limits in zip(bin_edges[:-1], bin_edges[1:]):
            tmp_model = np.where((Dalitz_mass_vec>limits[0]) & (Dalitz_mass_vec<limits[1]),
                                 proj[Kspi],
                                 0)
            # print("tmp_model[123]:", tmp_model[123])
            # print("limits:", limits)
            # print("Dalitz_mass_vec[123]:", Dalitz_mass_vec[123])
            # print("len(Dalitz_mass_vec):", len(Dalitz_mass_vec))
            # print("trapz(tmp_model, Dalitz_mass_vec):", )
            if (trapz(tmp_model, Dalitz_mass_vec) == np.inf):
                binned_integrals.append(0)
            else:
                binned_integrals.append(trapz(tmp_model, Dalitz_mass_vec))
            # binned_integrals.append(trapz(tmp_model, dx=dx))
            print(" ")
            pass
        binned_integrals = np.array(binned_integrals)
        diffs = np.array( binned_integrals - h_data_to_fit[0] )
        all_pulls = np.array( diffs / np.sqrt(h_data_to_fit[0]) )
        # print(all_pulls)
        all_pulls = np.where(np.isnan(all_pulls) | np.isinf(all_pulls), 0, all_pulls)
        histo_pulls[Kspi] = (all_pulls, h_data_to_fit[1])
        pass
    return histo_pulls

########## get the projections
Kspi_projection = {}
B_data          = {}
pulls           = {}
for channel in list_channels:
    Kspi_projection[channel] = draw_projection_pdf(
        ntuples[channel],
        dalitz_pdfs_values[channel]
    )
    B_data[channel] = {
        "Bplus" : ntuples[channel].Bplus_data ,
        "Bminus": ntuples[channel].Bminus_data,
    }
    pulls[channel] = {}
    for Bsign in BSIGNS.keys():
        tmp_data = B_data[channel][Bsign]
        pulls[channel][Bsign] = get_pull_projections(
            tmp_data,
            Kspip_proj = Kspi_projection[channel][0][Bsign]["total_pdf"],
            Kspim_proj = Kspi_projection[channel][1][Bsign]["total_pdf"]
        )
        pass
    pass

def plot_projections(_data, _Kspi_projection, _pulls, _list_variables, fit_step="preFit/"):
    if (fit_step not in ["preFit/", ""]):
        print("Fit step is wrong---- abort")
        return
    for channel in list_channels:
        ntuple_plot = ntuples[channel]
        #####
        for Bsign in BSIGNS.keys():
            tmp_data = _data[channel][Bsign]
            ################### Kspip
            fig, ax = plt.subplots(2,gridspec_kw={'height_ratios': [6, 1]}, figsize=(11,11))
            plt.suptitle(f"{channel} {Bsign}")
            ax[0].plot(
                Dalitz_mass_vec,
                dalitz_scaling*_Kspi_projection[channel][0][Bsign]["total_pdf"],
                label="total_pdf"
            )
            for i in range(len(ntuple_plot.dalitz_pdfs[Bsign])):
                comp_pdf = ntuple_plot.dalitz_pdfs[Bsign][i]
                index_yields  = INDEX_YIELDS[Bsign]
                if (_list_variables[ntuple_plot.i_c][i][2][index_yields] == 0): continue
                ax[0].fill_between(
                    Dalitz_mass_vec,
                    dalitz_scaling*_Kspi_projection[channel][0][Bsign][comp_pdf.component],
                    alpha=0.5,
                    label=comp_pdf.component
                )
                pass
            mplhep.histplot(
                np.histogram(tmp_data["m_Kspip"], bins=nbins, range=m_Kspip_range),
                label=ntuple_plot.channel.tex,
                ax=ax[0],
                **kwargs_data
                # , density=True
            )
            ax[0].legend(title=Bsign)
            ax[0].set_xlabel("")
            ax[0].set_xticks(ax[0].get_xticks(), "")
            ax[0].set_xlim([0.35,3])
            ax[0].set_ylabel(f"Events / ({round(dalitz_scaling,3)} MeV$^2$)")
            mplhep.histplot(_pulls[channel][Bsign]["Kspip"],
                            label="Pulls Kspip",
                            ax=ax[1],
                            histtype="fill"
                            )
            ax[1].set_xlabel("Kspip")
            ax[1].set_xlim([0.35,3])
            ax[1].set_ylabel("Pulls")
            ax[1].set_ylim([-5,5])
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            if (job_id < 10):
                plt.savefig(f"{plot_dir}/{fit_step}{channel}_{Bsign}_total_pdf_Kspip_projection.png")
                pass
            plt.close("all")
            ################### Kspim
            fig, ax = plt.subplots(2,gridspec_kw={'height_ratios': [6, 1]}, figsize=(11,11))
            plt.suptitle(f"{channel} {Bsign}")
            ax[0].plot(
                Dalitz_mass_vec,
                dalitz_scaling*_Kspi_projection[channel][1][Bsign]["total_pdf"],
                label="total_pdf"
            )
            for i in range(len(ntuple_plot.dalitz_pdfs[Bsign])):
                comp_pdf = ntuple_plot.dalitz_pdfs[Bsign][i]
                index_yields  = INDEX_YIELDS[Bsign]
                if (_list_variables[ntuple_plot.i_c][i][2][index_yields] == 0): continue
                ax[0].fill_between(
                    Dalitz_mass_vec,
                    dalitz_scaling*_Kspi_projection[channel][1][Bsign][comp_pdf.component],
                    alpha=0.5,
                    label=comp_pdf.component
                )
                pass
            mplhep.histplot(
                np.histogram(tmp_data["m_Kspim"], bins=nbins, range=m_Kspip_range),
                label=ntuple_plot.channel.tex,
                ax=ax[0],
                **kwargs_data
                # , density=True
            )
            ax[0].legend(title=Bsign)
            ax[0].set_xlabel("")
            ax[0].set_xticks(ax[0].get_xticks(), "")
            ax[0].set_xlim([0.35,3])
            ax[0].set_ylabel(f"Events / ({round(dalitz_scaling,3)} MeV$^2$)")
            mplhep.histplot(_pulls[channel][Bsign]["Kspim"],
                            label="Pulls Kspim",
                            ax=ax[1],
                            histtype="fill"
                            )
            ax[1].set_xlabel("Kspim")
            ax[1].set_xlim([0.35,3])
            ax[1].set_ylabel("Pulls")
            ax[1].set_ylim([-5,5])
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            if (job_id < 10):
                plt.savefig(f"{plot_dir}/{fit_step}{channel}_{Bsign}_total_pdf_Kspim_projection.png")
                pass
            plt.close("all")
            pass
        pass
    return





####### and now the mass pdfs
mass_pdfs_values = {}
for channel in list_channels:
    mass_pdfs_values[channel] = ntuples[channel].draw_mass_pdfs(
        tf_Bmass_vec,
        preFit_list_variables
    )
    for Bsign in BSIGNS.keys():
        tmp_data = B_data[channel][Bsign]
        mplhep.histplot(
            np.histogram(tmp_data[ntuples[channel].variable_to_fit],
                         bins=nbins,
                         range=B_mass_range),
            label=ntuples[channel].channel.tex+" toy data"
        )
        plt.plot(
            Bmass_vec,
            mass_scaling*mass_pdfs_values[channel][Bsign]["total_mass_pdf"],
            label="Total"
        )
        for comp in ntuples[channel].components:
            plt.plot(
                Bmass_vec,
                mass_pdfs_values[channel][Bsign][comp[0]]*mass_scaling,linestyle="--",
                label=components_tex[comp[0]]+"\n"+comp[1]
            )
            pass
        plt.xlabel("Constrained $DK$ mass")
        plt.ylabel(f"Events / ({round(mass_scaling)} MeV)")
        plt.legend()
        plt.tight_layout()
        if (job_id < 10):
            plt.savefig(f"{plot_dir}/preFit/{channel}_{Bsign}_mass_distribution.png")
            pass
        plt.close("all")
        pass
    pass


#######
@tf.function
def nll(x):
    return NLL.get_total_nll(x) # , tensor_to_fit)


dict_norm_ampD0    = {
    "CB2DK_D2KSPIPI_DD" :  toy_phsp_DK["ampD0"],
    "CB2DPI_D2KSPIPI_DD" :  toy_phsp_Dpi["ampD0"],
}
dict_norm_ampD0bar = {
    "CB2DK_D2KSPIPI_DD" :  toy_phsp_DK["ampD0bar"],
    "CB2DPI_D2KSPIPI_DD" :  toy_phsp_Dpi["ampD0bar"],
}
dict_norm_zp_p = {
    "CB2DK_D2KSPIPI_DD" :  toy_phsp_DK["zp_p"],
    "CB2DPI_D2KSPIPI_DD" :  toy_phsp_Dpi["zp_p"],
}
dict_norm_zm_pp = {
    "CB2DK_D2KSPIPI_DD" :  toy_phsp_DK["zm_pp"],
    "CB2DPI_D2KSPIPI_DD" :  toy_phsp_Dpi["zm_pp"],
}


x = tf.cast(list(start_values.values()),tf.float64)
print("start computing")
print("test nll(x) : ", nll(x))

test_values = dict(**start_values)
test_values["xplus"]  = test_values["xplus"]*1000.
test_values["xminus"] = test_values["xminus"]*1000.
test_values["yplus"]  = test_values["yplus"]*1000.
test_values["yminus"] = test_values["yminus"]*1000.
x = tf.cast(list(test_values.values()),tf.float64)
print("start computing")
print("test nll(x) : ", nll(x))

import iminuit
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

chi2_function =  chi2_xy_to_physics_param(xplus      = means["xplus" ], 
                                          yplus      = means["yplus" ], 
                                          yminus     = means["yminus"], 
                                          xminus     = means["xminus"], 
                                          xxi        = means["xxi"   ], 
                                          yxi        = means["yxi"   ],
                                          pd_cov     = pd_cov)

# chi2_function(x_phys)
physics_param = ["gamma", "rb", "dB", "rb_dpi", "dB_dpi"]
x_phys = [1,1,1,1,1]
m_phys = iminuit.Minuit(chi2_function, x_phys, name=physics_param)
m_phys.limits = list([ [0, np.pi],
                       [0, 1.],
                       [0, 2*np.pi],
                       [0, 1.],
                       [-np.pi, np.pi],
                      ])
mg_phys = m_phys.migrad()

# result_phys = opt.minimize(chi2_function, x_phys)
print(mg_phys)
means_phys  = mg_phys.values
errors_phys = mg_phys.errors
hesse_phys  = mg_phys.hesse()
cov_phys    = hesse_phys.covariance
corr_phys   = cov_phys.correlation()
for par in ["gamma", "dB", "dB_dpi"]:
    means_phys[par]  = rad_to_deg(means_phys[par])
    errors_phys[par] = rad_to_deg(errors_phys[par])
    pass

print("Means   ", means_phys)

# print("Means   ", means)
# print("Errors  ", errors)
means_results = dict(zip(parameters_to_fit,means))
errors_results = dict(zip(parameters_to_fit,errors))
means_results.update(dict(zip(physics_param, means_phys)))
errors_results.update(dict(zip(physics_param, errors_phys)))

print(json.dumps(means_results,indent=4))
print(json.dumps(errors_results,indent=4))


    
    

if ((mg_phys.valid==True) and (mg.valid==True)):
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
if (job_id < 10):
    plt.savefig(f"{plot_dir}/covariance.png")
    pass
plt.close("all")


postFit_list_variables = ntuples["CB2DK_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=means,
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)
postFit_list_variables = ntuples["CB2DPI_D2KSPIPI_DD"].get_list_variables(
    NLL.fixed_variables,
    params=means,
    shared_parameters=NLL.shared_parameters,
    constrained_parameters=NLL.constrained_parameters
)

################ PLOTS

#### compute the pdfs
postFit_dalitz_pdfs_values = {}
for channel in list_channels:
    # print(1)
    postFit_dalitz_pdfs_values[channel] = draw_dalitz_pdf(
        ntuples[channel],
        ampD0   ,
        ampD0bar,
        zp_p    ,
        zm_pp   ,
        variables=postFit_list_variables
    )
    # print(2)
    ### plotting these pdfs
    for Bsign in postFit_dalitz_pdfs_values[channel].keys():
        # print(3)
        for comp in postFit_dalitz_pdfs_values[channel][Bsign].keys():
            # print(4)
            # postFit_dalitz_pdfs_values[channel][Bsign][comp] *= Z
            try:
                cs = plt.contourf(
                    Dalitz_Kspip_mat,
                    Dalitz_Kspim_mat,
                    postFit_dalitz_pdfs_values[channel][Bsign][comp],
                    levels=100) # ,
                # norm="log")
            except TypeError:
                continue
            # , levels=[10, 30, 50], colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
            plt.xlabel("Kspip")
            plt.ylabel("Kspim")
            plt.title(f"{channel} {Bsign}")
            cbar = plt.colorbar(cs)
            # cs.cmap.set_over('red')
            # cs.cmap.set_under('white')
            # cs.changed()
            plt.tight_layout()
            if (job_id < 10):
                plt.savefig(f"{plot_dir}/{channel}_{Bsign}_{comp}.png")
                pass
            plt.close("all")
            pass
        pass
    pass



########## get the projections
postFit_Kspi_projection = {}
postFit_pulls           = {}
for channel in list_channels:
    postFit_Kspi_projection[channel] = draw_projection_pdf(
        ntuples[channel],
        postFit_dalitz_pdfs_values[channel]
    )
    postFit_pulls[channel] = {}
    for Bsign in BSIGNS.keys():
        tmp_data = B_data[channel][Bsign]
        postFit_pulls[channel][Bsign] = get_pull_projections(
            tmp_data,
            Kspip_proj = postFit_Kspi_projection[channel][0][Bsign]["total_pdf"],
            Kspim_proj = postFit_Kspi_projection[channel][1][Bsign]["total_pdf"]
        )
        pass
    pass

plot_projections(B_data, postFit_Kspi_projection, postFit_pulls, postFit_list_variables, fit_step="")


####### and now the mass pdfs
postFit_mass_pdfs_values = {}
for channel in list_channels:
    postFit_mass_pdfs_values[channel] = ntuples[channel].draw_mass_pdfs(
        tf_Bmass_vec,
        postFit_list_variables
    )
    for Bsign in BSIGNS.keys():
        tmp_data = B_data[channel][Bsign]
        mplhep.histplot(
            np.histogram(tmp_data[ntuples[channel].variable_to_fit],
                         bins=nbins,
                         range=B_mass_range),
            label=ntuples[channel].channel.tex
        )
        plt.plot(
            Bmass_vec,
            mass_scaling*postFit_mass_pdfs_values[channel][Bsign]["total_mass_pdf"],
            label="Total"
        )
        for comp in ntuples[channel].components:
            plt.plot(
                Bmass_vec,
                postFit_mass_pdfs_values[channel][Bsign][comp[0]]*mass_scaling,linestyle="--",
                label=components_tex[comp[0]]+"\n"+comp[1]
            )
            pass
        plt.xlabel(ntuples[channel].variable_to_fit)
        plt.ylabel(f"Events / ({round(mass_scaling)} MeV)")
        plt.title(f"{channel} {Bsign}")
        plt.tight_layout()
        if (job_id < 10):
            plt.savefig(f"{plot_dir}/{channel}_{Bsign}_mass_distribution.png")
            pass
        plt.close("all")
        pass
    pass


if ("scratch" in plot_dir):
    os.remove(f"{pathname_output}/outfile_{job_id}.root")
    pass



# tmp_data = B_data["CB2DK_D2KSPIPI_DD"]["Bplus"]
# plt.scatter(
#     tmp_data["m_Kspim"],
#     tmp_data["m_Kspip"],
#     c=np.conj(tmp_data["AmpD0"])*tmp_data["AmpD0"]
# )
# plt.xlabel("Kspim")
# plt.ylabel("Kspip")
# plt.savefig("Bplus_ampD0.png")
