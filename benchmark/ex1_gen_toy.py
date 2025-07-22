
from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import p4_to_srd
from matplotlib import pyplot as plt
from tfpcbpggsz.amp.amplitude import Amplitude
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from plothist import plot_hist, make_hist 
from tfpcbpggsz.variable import VarsManager
import argparse 
import importlib.resources


argparser = argparse.ArgumentParser()
argparser.add_argument('--index', type=int, default=1)
args = argparser.parse_args()

index = args.index
import time

#Set the path for the data and plot, get the package path
import os
path = importlib.resources.files('tfpcbpggsz').joinpath('../benchmark')

data_path= os.path.join(path, 'data/')


os.makedirs(data_path,exist_ok=True)

time1 = time.time()
#Load the fitted results
order=6
fit_result = np.load(f'{path}/results/full_data_fit_order{order}.npz', allow_pickle=True)
coefficients = fit_result['fitted_params']

#Call the amplitude
Amplitude_D = Amplitude(model='evtgen')
Amplitude_D.init()
#Generating the B2DK signal 
pcgen = pcbpggsz_generator(amplitude=Amplitude_D)
pcgen.add_bias(correctionType="antiSym_legendre", order=order, coefficients=coefficients)

##CP odd 
np.random.seed(int(time.time()))

N_gen = np.load(f"{path}/results/poisson_yields.npz", allow_pickle=True)
#CP odd
cp_odd_tags=['kspi0', 'kseta_gamgam', 'ksetap_pipieta', 'kseta_3pi', 'ksetap_gamrho', 'ksomega', 'klpi0pi0']
ret_cp_odd = {}
srd_cp_odd = {}
for tag in cp_odd_tags:
    n_cp_odd = N_gen[tag][index]
    ret_cp_odd[tag] = pcgen.generate(n_cp_odd, type="cp_odd")#8444
    srd_cp_odd[tag] = p4_to_srd(ret_cp_odd[tag])

#CP even
cp_even_tags=['kk', 'pipi', 'pipipi0', 'kspi0pi0', 'klpi0']
ret_cp_even = {}
srd_cp_even = {}
for tag in cp_even_tags:
    n_cp_even = N_gen[tag][index]
    ret_cp_even[tag] = pcgen.generate(n_cp_even, type="cp_even")#14646
    srd_cp_even[tag] = p4_to_srd(ret_cp_even[tag])

#Double Kspipi
cp_mixed_tags=['full', 'misspi', 'misspi0']
ret_sig = {}
ret_tag = {}
srd_sig = {}
srd_tag = {}
for tag in cp_mixed_tags:
    n_sig = N_gen[tag][index]
    ret_sig[tag], ret_tag[tag] = pcgen.generate(n_sig*2, type="cp_mixed")#10923
    srd_sig[tag] = p4_to_srd(ret_sig[tag])
    srd_tag[tag] = p4_to_srd(ret_tag[tag])

#B2DK
B_rec = ['b2dk_LL', 'b2dk_DD']
ret_Bp = {}
ret_Bm = {}
srd_Bp = {}
srd_Bm = {}
for tag in B_rec:
    n_b2dk = N_gen[tag][index]
    ret_Bp[tag] = pcgen.generate(int(n_b2dk/2), type="b2dh", gamma=68.7, rb=0.0904, dB=118.3, charge='p', decay=tag, apply_eff=False)
    ret_Bm[tag] = pcgen.generate(int(n_b2dk/2), type="b2dh", gamma=68.7, rb=0.0904, dB=118.3, charge='m', decay=tag, apply_eff=False)
    srd_Bp[tag] = p4_to_srd(ret_Bp[tag])
    srd_Bm[tag] = p4_to_srd(ret_Bm[tag])


print("Signal generated")
time2 = time.time()

#cp_odd
amp_cp_odd, ampbar_cp_odd = {}, {}
for tag in cp_odd_tags:
    amp_cp_odd[tag] = pcgen.amp(ret_cp_odd[tag])
    ampbar_cp_odd[tag] = pcgen.ampbar(ret_cp_odd[tag])
#cp_even
amp_cp_even, ampbar_cp_even = {}, {}
for tag in cp_even_tags:
    amp_cp_even[tag] = pcgen.amp(ret_cp_even[tag])
    ampbar_cp_even[tag] = pcgen.ampbar(ret_cp_even[tag])
#Double Kspipi
amp_sig, ampbar_sig = {}, {}
amp_tag, ampbar_tag = {}, {}
for tag in cp_mixed_tags:
    amp_sig[tag] = pcgen.amp(ret_sig[tag])
    ampbar_sig[tag] = pcgen.ampbar(ret_sig[tag])
    amp_tag[tag] = pcgen.amp(ret_tag[tag])
    ampbar_tag[tag] = pcgen.ampbar(ret_tag[tag])
#B2DK
amp_Bp, ampbar_Bp = {}, {}
amp_Bm, ampbar_Bm = {}, {}
for tag in B_rec:
    amp_Bp[tag] = pcgen.amp(ret_Bp[tag])
    ampbar_Bp[tag] = pcgen.ampbar(ret_Bp[tag])
    amp_Bm[tag] = pcgen.amp(ret_Bm[tag])
    ampbar_Bm[tag] = pcgen.ampbar(ret_Bm[tag])

#PHSP
phsp = PhaseSpaceGenerator().generate
phsp_cp_odd, phsp_cp_even = {}, {}
srd_phsp_cp_odd, srd_phsp_cp_even = {}, {}
amp_phsp_cp_odd, ampbar_phsp_cp_odd = {}, {}
amp_phsp_cp_even, ampbar_phsp_cp_even = {}, {}
amp_phsp_sig, ampbar_phsp_sig = {}, {}
amp_phsp_tag, ampbar_phsp_tag = {}, {}
amp_phsp_b2dk_p, ampbar_phsp_b2dk_p = {}, {}
amp_phsp_b2dk_m, ampbar_phsp_b2dk_m = {}, {}
for tag in cp_odd_tags:
    phsp_cp_odd[tag] = phsp(100000)
    srd_phsp_cp_odd[tag] = p4_to_srd(phsp_cp_odd[tag])
    amp_phsp_cp_odd[tag] = pcgen.amp(phsp_cp_odd[tag])
    ampbar_phsp_cp_odd[tag] = pcgen.ampbar(phsp_cp_odd[tag])   
for tag in cp_even_tags:
    phsp_cp_even[tag] = phsp(100000)
    srd_phsp_cp_even[tag] = p4_to_srd(phsp_cp_even[tag])
    amp_phsp_cp_even[tag] = pcgen.amp(phsp_cp_even[tag])
    ampbar_phsp_cp_even[tag] = pcgen.ampbar(phsp_cp_even[tag])
phsp_sig, phsp_tag = {}, {}
srd_phsp_sig, srd_phsp_tag = {}, {}
for tag in cp_mixed_tags:
    phsp_sig[tag] = phsp(100000)
    phsp_tag[tag] = phsp(100000)
    srd_phsp_sig[tag] = p4_to_srd(phsp_sig[tag])
    srd_phsp_tag[tag] = p4_to_srd(phsp_tag[tag])
    amp_phsp_sig[tag] = pcgen.amp(phsp_sig[tag])
    ampbar_phsp_sig[tag] = pcgen.ampbar(phsp_sig[tag])
    amp_phsp_tag[tag] = pcgen.amp(phsp_tag[tag])
    ampbar_phsp_tag[tag] = pcgen.ampbar(phsp_tag[tag])
phsp_b2dk_p, phsp_b2dk_m = {}, {}
srd_phsp_b2dk_p, srd_phsp_b2dk_m = {}, {}
for tag in B_rec:
    phsp_b2dk_p[tag] = phsp(500000)
    phsp_b2dk_m[tag] = phsp(500000)
    srd_phsp_b2dk_p[tag] = p4_to_srd(phsp_b2dk_p[tag])
    srd_phsp_b2dk_m[tag] = p4_to_srd(phsp_b2dk_m[tag])
    amp_phsp_b2dk_p[tag] = pcgen.amp(phsp_b2dk_p[tag])
    ampbar_phsp_b2dk_p[tag] = pcgen.ampbar(phsp_b2dk_p[tag])
    amp_phsp_b2dk_m[tag] = pcgen.amp(phsp_b2dk_m[tag])
    ampbar_phsp_b2dk_m[tag] = pcgen.ampbar(phsp_b2dk_m[tag])

time3 = time.time()


data_LHCb = {'data_Bp': ret_Bp, 'data_Bm': ret_Bm, 'phsp_Bp': phsp_b2dk_p, 'phsp_Bm': phsp_b2dk_m}
data_BES = {'data_cp_odd': ret_cp_odd, 'data_cp_even': ret_cp_even, 'data_sig': ret_sig, 'data_tag': ret_tag, 'phsp_cp_odd': phsp_cp_odd, 'phsp_cp_even': phsp_cp_even, 'phsp_sig': phsp_sig, 'phsp_tag': phsp_tag}
np.save(f"{data_path}LHCb_data_{index}.npy", data_LHCb)
np.save(f"{data_path}BES_data_{index}.npy", data_BES)
time3 = time.time()
print("Data saved")
