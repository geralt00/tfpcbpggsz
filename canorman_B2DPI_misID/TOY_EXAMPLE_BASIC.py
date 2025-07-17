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
from tfpcbpggsz.dalitz_pdfs import DICT_EFFICIENCY_FUNCTIONS
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

numbernormalisation  = 1000000 # args.numbernormalisation # "Legendre_2_2"

# 
plot_dir=f"2025_07_16/TOY_EXAMPLE_BASIC"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(f"{plot_dir}/preFit/",exist_ok=True)
os.makedirs(f"{plot_dir}/generation/",exist_ok=True)
shutil.copyfile('TOY_EXAMPLE_BASIC.py', f'{plot_dir}/TOY_EXAMPLE_BASIC.py')
pathname_output = plot_dir
print(" plot_dir       : ", plot_dir)
print(" pathname_output: ", pathname_output)


# Create the generator object
pcgen = pcbpggsz_generator()

# copy the big VARDICT dictionary
fixed_variables = dict(**VARDICT["SDATA"])

# get the mass shape parameters for the generator
B2DK_mass_variables  = list(fixed_variables["CB2DK_D2KSPIPI_DD"]["DK_Kspipi"]["mass"].values())
B2Dpi_mass_variables = list(fixed_variables["CB2DPI_D2KSPIPI_DD"]["Dpi_Kspipi"]["mass"].values())
B2Dpi_misID_mass_variables = list(fixed_variables["CB2DK_D2KSPIPI_DD"]["Dpi_Kspipi_misID"]["mass"].values())


print(B2DK_mass_variables)

####### LOAD THE RESULT OF THE EFFICIENCY FIT

B2DK_Bplus_efficiency_variables    = [1.]
B2DK_Bminus_efficiency_variables   = [1.]


##### B2Dpi is generated without efficiency effects
B2Dpi_Bminus_efficiency_variables  = [1.] 
B2Dpi_Bplus_efficiency_variables   = [1.]

##### B2Dpi misID

B2Dpi_misID_Bminus_efficiency_variables  = [1.]
B2Dpi_misID_Bplus_efficiency_variables   = [1.]

#################

####################### some tools for plotting
# mass
min_mass = 5239
max_mass = 5309
Bmass_vec = np.arange(min_mass, max_mass, 1)
tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)
B_mass_range = [min_mass, max_mass]
#### dalitz
min_dalitz = 0.4
max_dalitz = 3.0
nbins=100
m_Kspip_range = [min_dalitz, max_dalitz]
num_mass_values = 1000.
Dalitz_mass_vec = np.arange(min_dalitz, max_dalitz, (max_dalitz-min_dalitz)/num_mass_values )
tf_Dalitz_mass_vec = tf.cast(Dalitz_mass_vec, tf.float64)
#### scaling for comparison histogram vs pdfs
mass_scaling   = (max_mass - min_mass) / float(nbins)
dalitz_scaling = (m_Kspip_range[1]-m_Kspip_range[0]) / float(nbins)
######## 2D grid for plotting 
Dalitz_Kspip_mat, Dalitz_Kspim_mat = np.meshgrid(Dalitz_mass_vec,Dalitz_mass_vec)
## SRD variables
RD_var = func_var_rotated(Dalitz_Kspip_mat, Dalitz_Kspim_mat, QMI_zpmax_Kspi, QMI_zpmin_Kspi, QMI_zmmax_Kspi, QMI_zmmin_Kspi)
SRD_var = func_var_rotated_stretched(RD_var)
zp_p  = SRD_var[0]
zm_pp = SRD_var[1]
srd_range = [-1,1]
# compute the amplitudes of this 2D mesh
ampD0    = np.zeros(zp_p.shape , dtype=complex)
ampD0bar = np.zeros(zm_pp.shape, dtype=complex)
for row in range(zp_p.shape[0]):
    # if (row%100==0): print("Processed ", row)
    for col in range(zp_p.shape[1]):
        tmp_amps = Kspipi.get_amp(
            zp_p[row][col] ,
            zm_pp[row][col]
        )
        ampD0[row][col]    = tmp_amps[0]
        ampD0bar[row][col] = np.negative(tmp_amps[1])
        pass
    pass



############## DEFINE THE COMPONENTS IN THE TOY
components = {
    "CB2DK_D2KSPIPI_DD": [ # the channel
        ["DK_Kspipi", "Cruijff+Gaussian"  , "Flat"], # name of the component, mass shape, and efficiency function
    ],
    "CB2DPI_D2KSPIPI_DD": [
        ["Dpi_Kspipi", "Cruijff+Gaussian" , "Flat"]
    ],
}
components_tex = { # for convenience
    "DK_Kspipi" : r"$B^{\pm} \rightarrow D K^{\pm}$",
    "Dpi_Kspipi": r"$B^{\pm} \rightarrow D \pi^{\pm}$",
    "Dpi_Kspipi_misID": r"$B^{\pm} \rightarrow D \pi^{\pm}_{\pi\rightarrow K}$"
}

# list of the channels that will be included in these toys
list_channels = ["CB2DK_D2KSPIPI_DD", "CB2DPI_D2KSPIPI_DD"]

# name of the mass variables for each channel
variable_to_fit = {
    "CB2DK_D2KSPIPI_DD" : "Bu_constD0KSPV_M",
    "CB2DPI_D2KSPIPI_DD": "Bu_constD0KSPV_swapBachToPi_M",
}

############
yields = { # per B sign
    "DK_Kspipi_DD"        : 6266 , # 1000, # 10,  #  
    "Dpi_Kspipi_DD"       : 89941, # 1000, # 10,  #  
}

# generation parameters
generation_parameters = {
    "CB2DK_D2KSPIPI_DD": {
        "DK_Kspipi": [
            yields["DK_Kspipi_DD"]    ,       # yield
            "Cruijff+Gaussian"        ,       # mass shape
            "B2Dh_D2Kspipi"           ,       # useless
            B2DK_mass_variables       ,       # mass variables
            [GAMMA, RB_DK, DELTAB_DK] ,       # inputs
            B2DK_Bplus_efficiency_variables , # Dalitz efficiencies Bplus
            B2DK_Bminus_efficiency_variables, # Dalitz efficiencies Bminus
            "Flat"                            # name of the efficiency function
        ],
    },
    "CB2DPI_D2KSPIPI_DD": {
        "Dpi_Kspipi": [
            yields["Dpi_Kspipi_DD"]  ,
            "Cruijff+Gaussian"                   ,
            "B2Dh_D2Kspipi"                ,
            B2Dpi_mass_variables     ,
            [GAMMA, RB_DPI, DELTAB_DPI]    ,
            B2Dpi_Bplus_efficiency_variables ,
            B2Dpi_Bminus_efficiency_variables,
            "Flat"
        ],
    },
}

# dictionaries that contain the generated events
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

# start the generation per channel
for channel in list_channels:
    print("Start generating events for channel ", channel)
    # initialise dictionaries
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
    for comp in components[channel]: # loop over the components in each channel
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
            efficiency_function  = generation_parameters[channel][gen_comp][7],
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
            efficiency_function  = generation_parameters[channel][gen_comp][7],
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

# some plots
for channel in list_channels:
    for comp in components[channel]: # loop over the components in each channel
        gen_comp = comp[0]
        plt.scatter(m13_p_DK[channel][gen_comp], m12_p_DK[channel][gen_comp], label=rf"$B^+$, {gen_comp}",
                    marker="o",
                    s=1,
                    color="black")
        plt.xlabel(r"$m(K_S\pi^+)^2$ [GeV/$c^2$]")
        plt.ylabel(r"$m(K_S\pi^-)^2$ [GeV/$c^2$]")
        plt.savefig(f"{plot_dir}/generation/{channel}_Bplus_{gen_comp}.png")
        plt.close("all")
        plt.scatter(m13_m_DK[channel][gen_comp], m12_m_DK[channel][gen_comp], label=rf"$B^-$, {gen_comp}",
                    marker="o",
                    s=1,
                    color="black")
        plt.xlabel(r"$m(K_S\pi^+)^2$ [GeV/$c^2$]")
        plt.ylabel(r"$m(K_S\pi^-)^2$ [GeV/$c^2$]")
        plt.savefig(f"{plot_dir}/generation/{channel}_Bminus_{gen_comp}.png")
        plt.close("all")
        pass
    pass

# organise the generated toys the same way the real data is organised before feeding it to the fitter
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


# save the toy in the same format as the real data
uproot_data = {}
print(" =============    STORE TOY AT")
print(f"{pathname_output}/outfile.root")
outfile = up.recreate(f"{pathname_output}/outfile.root")
for channel in list_channels:
    uproot_data[channel] = {}
    uproot_data[channel]  = pd.DataFrame.from_dict(toy_data[channel])
    outfile[channel]  = uproot_data[channel]
    pass



############# GENERATE NORMALISATION
# this is done per channel (two components in the same channel will share the same normalisation sample)

######## CB2DK
#PHSP
yield_normalisation = int(numbernormalisation)

normalisation_parameters = {
    "CB2DK_D2KSPIPI_DD": [
        yield_normalisation,
        B2DK_Bplus_efficiency_variables , # efficiency parameters Bplus
        B2DK_Bminus_efficiency_variables, # efficiency parameters Bminus
        "Flat",                   # name of the efficiency function
    ],
    "CB2DPI_D2KSPIPI_DD": [
        yield_normalisation,
        B2Dpi_Bplus_efficiency_variables ,
        B2Dpi_Bminus_efficiency_variables,
        "Flat" # "Flat" implies no efficiency 
    ],
}

phsp_Bp, phsp_Bp_mass = {}, {}
phsp_Bm, phsp_Bm_mass = {}, {}
phsp_p1_p,phsp_p2_p,phsp_p3_p = {}, {}, {}
phsp_p1_m,phsp_p2_m,phsp_p3_m = {}, {}, {}
phsp_m12_p = {}
phsp_m13_p = {}
phsp_m12_m = {}
phsp_m13_m = {}
phsp_srd_p = {}
phsp_srd_m = {}
amp_phsp_p, ampbar_phsp_p = {}, {}
amp_phsp_m, ampbar_phsp_m = {}, {}
toy_phsp = {}
for channel in list_channels:
    print("Start generating normlisation events for channel ", channel)
    phsp_Bp[channel] = pcgen.generate(
        yield_normalisation,
        type="phsp",
        efficiency_function  = normalisation_parameters[channel][3],
        efficiency_variables = normalisation_parameters[channel][1],
    )
    phsp_Bm[channel] = pcgen.generate(
        yield_normalisation,
        type="phsp",
        efficiency_function  = normalisation_parameters[channel][3],
        efficiency_variables = normalisation_parameters[channel][2],
    )
    # isolate the four momenta of each particle (KS, pip, pim)
    phsp_p1_p[channel],phsp_p2_p[channel],phsp_p3_p[channel] = phsp_Bp[channel] # Ks, pi-, pi+
    phsp_p1_m[channel],phsp_p2_m[channel],phsp_p3_m[channel] = phsp_Bm[channel] # Ks, pi-, pi+
    # compute the invariant masses
    phsp_m12_p[channel] = get_mass(phsp_p1_p[channel],phsp_p2_p[channel])
    phsp_m13_p[channel] = get_mass(phsp_p1_p[channel],phsp_p3_p[channel])
    phsp_m12_m[channel] = get_mass(phsp_p1_m[channel],phsp_p2_m[channel])
    phsp_m13_m[channel] = get_mass(phsp_p1_m[channel],phsp_p3_m[channel])
    # srd variables
    phsp_srd_p[channel] = phsp_to_srd(phsp_m12_p[channel],phsp_m13_p[channel])
    phsp_srd_m[channel] = phsp_to_srd(phsp_m12_m[channel],phsp_m13_m[channel])
    amp_phsp_p[channel], ampbar_phsp_p[channel] = pcgen.amp(phsp_Bp[channel]), pcgen.ampbar(phsp_Bp[channel])
    amp_phsp_m[channel], ampbar_phsp_m[channel] = pcgen.amp(phsp_Bm[channel]), pcgen.ampbar(phsp_Bm[channel])

    # organise the normalisation samples the same way the real MC samples are
    toy_phsp[channel] = {
        "ampD0" : {
            "Bplus"  : amp_phsp_p[channel],
            "Bminus" : amp_phsp_m[channel],
        },
        "ampD0bar" : {
            "Bplus"  : ampbar_phsp_p[channel],
            "Bminus" : ampbar_phsp_m[channel],
        },
        "zp_p" : {
            "Bplus"  : phsp_srd_p[channel][0],
            "Bminus" : phsp_srd_m[channel][0],
        },
        "zm_pp" : {
            "Bplus"  : phsp_srd_p[channel][1],
            "Bminus" : phsp_srd_m[channel][1],
        },
    }
    pass


toy_phsp_DK  = toy_phsp["CB2DK_D2KSPIPI_DD"]
toy_phsp_Dpi = toy_phsp["CB2DPI_D2KSPIPI_DD"]


####### START THE FITTING SECTION

# From the big VARDICT directory, reads only the channels and components that we want to include
# This is actually quite important because it determines the order in which the channels and components
#     appear in the big table.
input_variables = {}
for channel in list_channels:
    input_variables[channel]       = {} ## loop over channel
    for comp in components[channel]: ## loop over components
        input_variables[channel][comp[0]] = VARDICT["SDATA"][channel][comp[0]]
        # print(comp[0])
        pass
    pass

# This one HAS to be there
input_variables["SHARED_THROUGH_CHANNELS"] = VARDICT["SDATA"]["SHARED_THROUGH_CHANNELS"]


# define input values of the variables in the fit
inputs = {
    "gamma" : deg_to_rad(GAMMA), "rb" : RB_DK, "dB" : deg_to_rad(DELTAB_DK),
    "rb_dpi": RB_DPI, "dB_dpi": deg_to_rad(DELTAB_DPI)
}
inputs["xplus"], inputs["yplus"], inputs["xminus"], inputs["yminus"], inputs["xxi"], inputs["yxi"] = get_xy_xi(
    (inputs["gamma"], inputs["rb"], inputs["dB"], inputs["rb_dpi"], inputs["dB_dpi"])
)


############ initialise ntuples for fitting
# variables 
basic_list_var = ["Bu_ID", "zp_p", "zm_pp", "m_Kspip", "m_Kspim"]
for particle in ["KS","pim","pip"]:
    for mom in ["PE", "PX", "PY", "PZ"]:
        basic_list_var += [f"{particle}_{mom}"]
        pass
    pass

# ntuples objects
ntuples = {}
for channel in list_channels:
    ntuples[channel]  = Ntuple(
        "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow",channel,"YRUN2", "MagAll"
    )
    # again, an important caveat: we see here the variable "index_channel" that is given to the
    # initialise_fit method. This index channel is read from the input_variables dictionary.
    # This ensures that, when changing "input_variables" from a dictionary to a table,
    # the index used in the table for each channel does indeed match from the original dictionary.
    index_channel = list(input_variables.keys()).index(channel)
    ntuples[channel].initialise_fit(components[channel], index_channel)
    list_var = [ntuples[channel].variable_to_fit] + basic_list_var
    # this is the method that reads the data that is gonna be fitted, here we read the toy that we previously generated
    ntuples[channel].store_events(
        f"{pathname_output}/outfile.root:{channel}", # f"/shared/scratch/rj23972/safety_net/tfpcbpggsz/canorman_B2DPI_misID/2025_06_16/01_No_Efficiency_noMisID/study_-2/outfile_-2.root:{channel}", # f"/software/rj23972/safety_net/tfpcbpggsz/canorman_Efficency/2025_05_28/TEST_FITTER/REFERENCE_FITTER/outfile_{job_id}.root:{channel}", 
        list_var,
        None,
        Kspipi_up
    )
    pass


# starting values of the fitted parameters
start_values = {
    "yield_Bplus_DK"  :                          yields["DK_Kspipi_DD"]/1000,
    "yield_Bminus_DK" :                          yields["DK_Kspipi_DD"]/1000,
    "signal_mean_DK":                            varDict['signal_mean']+50,
    "signal_width_DK":                           varDict['sigma_dk_DD']+50,
    "yield_Bplus_Dpi"  :                          yields["Dpi_Kspipi_DD"],
    "yield_Bminus_Dpi" :                          yields["Dpi_Kspipi_DD"],
    "signal_mean_Dpi":                            varDict['signal_mean']+50,
    "signal_width_Dpi":                           varDict['sigma_dk_DD']+50,
    # "yield_Bplus_Dpi_misID"  :                      400.,
    # "yield_Bminus_Dpi_misID" :                      400.,
    "xplus"                                        :   0.50, # inputs["xplus" ], #
    "yplus"                                        :  -0.00, # inputs["yplus" ], #
    "xminus"                                       :   0.50, # inputs["xminus"], #
    "yminus"                                       :  -0.00, # inputs["yminus"], #
    "xxi"                                          :   0.50, # inputs["xxi"   ], #
    "yxi"                                          :  -0.00, # inputs["yxi"   ], #
    # "c01_DK_Kspipi_DD_Bplus"                       : 0., # -0.00,
}

# define limits 
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

# define which free fit parameters is applied to which "real" variables
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

free_efficiency_coeffs_fit = []

# multiplicative constraints, for example yield_1 = yield_2 * eff_1/eff_2, with yield_2 a free parameters of the fit
dict_constrained_parameters = [
    ## constrain misID DPI from goodID in DK
    # [ ["CB2DK_D2KSPIPI_DD", "Dpi_Kspipi_misID" , "mass", "yield_Bplus"], ["CB2DPI_D2KSPIPI_DD" , "Dpi_Kspipi" , "mass", "yield_Bplus",  ratio_Dpi_misID_to_Dpi] ],
    # [ ["CB2DK_D2KSPIPI_DD", "Dpi_Kspipi_misID" , "mass", "yield_Bminus"], ["CB2DPI_D2KSPIPI_DD" , "Dpi_Kspipi" , "mass", "yield_Bminus",  ratio_Dpi_misID_to_Dpi] ],
]
dict_gaussian_constraints   = []


# the object that combines all of the above ingredients (especially all the "ntuple object"
# for each channel
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

# name of the parameters to fit, this is extracted from the "start_values" dictionary
parameters_to_fit = NLL.parameters_to_fit

# extract the multi-D table (not dictionary !) for plotting purposes
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

# defines the normalisation sample for each channel
ntuples["CB2DK_D2KSPIPI_DD"].define_dalitz_pdfs(
    toy_phsp_DK["ampD0"], toy_phsp_DK["ampD0bar"], toy_phsp_DK["zp_p"], toy_phsp_DK["zm_pp"]
)
ntuples["CB2DPI_D2KSPIPI_DD"].define_dalitz_pdfs(
    toy_phsp_Dpi["ampD0"], toy_phsp_Dpi["ampD0bar"], toy_phsp_Dpi["zp_p"], toy_phsp_Dpi["zm_pp"]
)


# for plotting only: because we normalise the events with non-flat events, we need
# to multiply the PDFs entering the likelihood by the MC efficiency.
normalisation_params = {
    "CB2DK_D2KSPIPI_DD": {
        "Function": "Flat",  # name of the function # "Flat"
        "Parameters": {
            "Bplus" :  B2DK_Bplus_efficiency_variables , # parameters
            "Bminus":  B2DK_Bminus_efficiency_variables, # parameters
        }
    },
    "CB2DPI_D2KSPIPI_DD": {
        "Function": "Flat",
        "Parameters": {
            "Bplus" :  B2Dpi_Bplus_efficiency_variables ,
            "Bminus":  B2Dpi_Bminus_efficiency_variables,
        }
    },
}

def draw_dalitz_pdf(ntuple, ampD0, ampD0bar, zp_p, zm_pp, variables=None, normalisation_params=None):
    """
    this function is only for plotting purposes. It computes the pdfs 
    for each component and the sum.
    The "variables" argument is optional if the ntuples object already has a ntuple.list_variables object.
    Otherwise, it requires the complete multi-D table as input.
    ampD0, ampD0bar, zp_p, zm_pp are 2D grids containing the points where we want to compute 
    the PDFs.

    It returns "pdfs_values", a dictionary that contain the PDFs computed at each point 
    of the provided grids, one PDF for each component in this ntuple, and the total PDF
    called "total_pdf".
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
        if (normalisation_params==None):
            normalisation_eff = 1.
            pass
        else:
            _norm_fun = DICT_EFFICIENCY_FUNCTIONS[normalisation_params["Function"]]
            normalisation_eff = _norm_fun(ampD0, ampD0bar, zp_p, zm_pp, BSIGNS[Bsign],
                                          variables=normalisation_params["Parameters"][Bsign])
            pass
        for i in range(len(ntuple.dalitz_pdfs[Bsign])): # loop over the components of the ntuple
            comp_pdf = ntuple.dalitz_pdfs[Bsign][i]
            tmp_pdf_values = comp_pdf.pdf(ampD0, ampD0bar, zp_p, zm_pp)
            # multiply each pdf by the efficiency
            tmp_pdf_values = tmp_pdf_values * normalisation_eff
            # multiply each pdf by a scaling factor (twice because 2D)
            tmp_pdf_values = tmp_pdf_values*dalitz_scaling*dalitz_scaling
            pdfs_values[Bsign][comp_pdf.component]     = tmp_pdf_values
            # before adding the components together, we multiply each one by its total yield
            index_yields  = INDEX_YIELDS[Bsign]
            comp_yield    = variables[ntuple.i_c][i][4][index_yields]
            pdfs_values[Bsign]["total_pdf"]           += tmp_pdf_values * comp_yield
            pass # loop comps
        pass # loop signs
    return pdfs_values

# store the 2D pdf for each channel
dalitz_pdfs_values = {}
for channel in list_channels:
    #### compute the pdfs
    dalitz_pdfs_values[channel] = draw_dalitz_pdf(
        ntuples[channel],
        ampD0   ,
        ampD0bar,
        zp_p    ,
        zm_pp   ,
        variables=preFit_list_variables,
        normalisation_params=normalisation_params[channel]
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
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/preFit/{channel}_{Bsign}_{comp}.png")
            plt.close("all")
            pass
        pass
    pass


def draw_projection_pdf(ntuple, pdfs_values, variables = None):
    """
    This function integrates separately over the two dimensions
    to get both projections of the Dalitz pdfs.

    Returns Kspip_projection and Kspim_projection, 
    containing the projections in the Kspip and Kspim directions
    for sign of the B, and each component, as well as the total PDF
    under "total_pdf"
    """
    variables = ntuple.list_variables
    Kspip_projection = {}
    Kspim_projection = {}
    for Bsign in BSIGNS.keys():
        # initialise the dictionaries
        Kspip_projection[Bsign] = {}
        Kspim_projection[Bsign] = {}
        Kspip_projection[Bsign]["total_pdf"] = np.zeros(
            pdfs_values[Bsign]["total_pdf"].shape[0]
        ).astype(np.float64)
        Kspim_projection[Bsign]["total_pdf"] = np.zeros(
            pdfs_values[Bsign]["total_pdf"].shape[0]
        ).astype(np.float64)
        for i in range(len(ntuple.dalitz_pdfs[Bsign])): # loop over the components
            index_yields  = INDEX_YIELDS[Bsign]
            ### this 4 is the index of the "mass" entry in the "space" direction
            comp_pdf   = ntuple.dalitz_pdfs[Bsign][i]
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
                # loop over each point in the direction we want to project to,
                # and compute the projection by "MC-integrating" in the other direction
                # Kspip
                Kspip_projection[Bsign][comp_pdf.component][i_Kspi] = float(tf.reduce_mean(
                    tmp_numpy_pdfs.transpose()[i_Kspi]
                ))
                # Kspim
                Kspim_projection[Bsign][comp_pdf.component][i_Kspi] = float(tf.reduce_mean(
                    tmp_numpy_pdfs[i_Kspi]
                ))
                pass # loop axes
            # normalise both projections to the correct yields
            comp_yield = variables[ntuple.i_c][i][4][index_yields]
            Kspip_projection[Bsign][comp_pdf.component] = comp_yield*norm_distribution(
                Dalitz_mass_vec,
                Kspip_projection[Bsign][comp_pdf.component]
            )
            Kspip_projection[Bsign]["total_pdf"] += Kspip_projection[Bsign][comp_pdf.component]
            Kspim_projection[Bsign][comp_pdf.component] = comp_yield*norm_distribution(
                Dalitz_mass_vec,
                Kspim_projection[Bsign][comp_pdf.component]
            )
            Kspim_projection[Bsign]["total_pdf"] += Kspim_projection[Bsign][comp_pdf.component]
            pass # loop comps
        pass # loop signs
    return Kspip_projection, Kspim_projection

def get_pull_projections(data, Kspip_proj = None, Kspim_proj = None):
    """
    Compute the pulls for each projection.
    """
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
            # print(" ")
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
    """
    Simply plot the projections and the pulls
    """
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
                if (_list_variables[ntuple_plot.i_c][i][4][index_yields] == 0): continue
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
            plt.savefig(f"{plot_dir}/{fit_step}{channel}_{Bsign}_total_pdf_Kspip_projection.png")
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
                if (_list_variables[ntuple_plot.i_c][i][4][index_yields] == 0): continue
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
            plt.savefig(f"{plot_dir}/{fit_step}{channel}_{Bsign}_total_pdf_Kspim_projection.png")
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
        plt.savefig(f"{plot_dir}/preFit/{channel}_{Bsign}_mass_distribution.png")
        plt.close("all")
        pass
    pass


# the actual nll function to feed to migrad
# @tf.function
def nll(x):
    return NLL.get_total_nll(x) # , tensor_to_fit)

# some tests to confirm wa cen compute what we want
# it also builds the tensorflow graph
x = tf.cast(list(start_values.values()),tf.float64)
print("start computing")
print("test nll(x) : ", nll(x))

# minimise !
import iminuit
m = iminuit.Minuit(nll, x, name=parameters_to_fit)
m.limits = list(limit_values.values())
mg = m.migrad()


# the results
print(mg)
means  = mg.values
errors = mg.errors
hesse  = mg.hesse()
print(hesse)
cov    = hesse.covariance
corr   = cov.correlation()
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

# minimisation to find get gamma, deltaB and rB
chi2_function =  chi2_xy_to_physics_param(xplus      = means["xplus" ], 
                                          yplus      = means["yplus" ], 
                                          yminus     = means["yminus"], 
                                          xminus     = means["xminus"], 
                                          xxi        = means["xxi"   ], 
                                          yxi        = means["yxi"   ],
                                          pd_cov     = pd_cov)
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

# results
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

means_results = dict(zip(parameters_to_fit,means))
errors_results = dict(zip(parameters_to_fit,errors))
means_results.update(dict(zip(physics_param, means_phys)))
errors_results.update(dict(zip(physics_param, errors_phys)))

print(json.dumps(means_results,indent=4))
print(json.dumps(errors_results,indent=4))    

# save the results if both fit converged
if ((mg_phys.valid==True) and (mg.valid==True)):
    with open(f"{plot_dir}/means_results.json", "w") as f:
        json.dump(means_results, f, indent=4)
        pass

    with open(f"{plot_dir}/errors_results.json", "w") as f:
        json.dump(errors_results, f, indent=4)
        pass

    pass

# plot covariance
import seaborn as sns
fig = plt.figure(figsize=(30, 24))  
sns.heatmap(pd_cov, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig(f"{plot_dir}/covariance.png")
plt.close("all")

# plot correlation
import seaborn as sns
fig = plt.figure(figsize=(30, 24))  
sns.heatmap(pd_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig(f"{plot_dir}/correlation.png")
plt.close("all")

# get the table of variables post Fit
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
    # get the 2D PDFs
    postFit_dalitz_pdfs_values[channel] = draw_dalitz_pdf(
        ntuples[channel],
        ampD0   ,
        ampD0bar,
        zp_p    ,
        zm_pp   ,
        variables=postFit_list_variables,
        normalisation_params=normalisation_params[channel]
    )
    ### plotting these pdfs
    for Bsign in postFit_dalitz_pdfs_values[channel].keys():
        for comp in postFit_dalitz_pdfs_values[channel][Bsign].keys():
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
            plt.savefig(f"{plot_dir}/{channel}_{Bsign}_{comp}.png")
            plt.close("all")
            pass
        pass
    pass



# get the projections
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

# plot the projections
plot_projections(B_data, postFit_Kspi_projection, postFit_pulls, postFit_list_variables, fit_step="")


# and the mass pdf
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
        plt.savefig(f"{plot_dir}/{channel}_{Bsign}_mass_distribution.png")
        plt.close("all")
        pass
    pass


# useful: it plots separately each generated component with its fitted PDF
for channel in list_channels:
    for i in range(len(components[channel])): # generation_parameters[channel].keys():
        comp = components[channel][i]
        gen_comp = comp[0]
        mplhep.histplot(
            np.histogram(m13_p_DK[channel][gen_comp], bins=nbins, range=m_Kspip_range),
            label=ntuples[channel].channel.tex,
            **kwargs_data
        )
        comp_pdf = ntuples[channel].dalitz_pdfs["Bplus"][i]
        index_yields  = INDEX_YIELDS["Bplus"]
        if (postFit_list_variables[ntuples[channel].i_c][i][4][index_yields] == 0): continue
        plt.fill_between(
            Dalitz_mass_vec,
            dalitz_scaling*postFit_Kspi_projection[channel][0]["Bplus"][comp_pdf.component],
            alpha=0.5,
            label=comp_pdf.component
        )
        plt.xlabel(r"$m(K_S\pi^+)^2$ [GeV/$c^2$]")
        plt.ylabel(r"$m(K_S\pi^-)^2$ [GeV/$c^2$]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{channel}_Bplus_{gen_comp}.png")
        plt.close("all")
        mplhep.histplot(
            np.histogram(m13_m_DK[channel][gen_comp], bins=nbins, range=m_Kspip_range),
            label=ntuples[channel].channel.tex,
            **kwargs_data
            # , density=True
        )
        comp_pdf = ntuples[channel].dalitz_pdfs["Bminus"][i]
        index_yields  = INDEX_YIELDS["Bminus"]
        if (postFit_list_variables[ntuples[channel].i_c][i][4][index_yields] == 0): continue
        plt.fill_between(
            Dalitz_mass_vec,
            dalitz_scaling*postFit_Kspi_projection[channel][0]["Bminus"][comp_pdf.component],
            alpha=0.5,
            label=comp_pdf.component
        )
        plt.xlabel(r"$m(K_S\pi^+)^2$ [GeV/$c^2$]")
        plt.ylabel(r"$m(K_S\pi^-)^2$ [GeV/$c^2$]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{channel}_Bminus_{gen_comp}.png")
        plt.close("all")
        pass
    pass



##################
