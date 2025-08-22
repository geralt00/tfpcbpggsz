
from email import parser
from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.amp.amplitude import Amplitude
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from tfpcbpggsz.generator.utils import save_as_root
from tfpcbpggsz.bes.config_loader import ConfigLoader
import argparse 
import importlib.resources


argparser = argparse.ArgumentParser()
argparser.add_argument('--index', type=int, default=1)
argparser.add_argument("--config", type=str, default="config.yml")
argparser.add_argument("--order", type=int, default=1)
argparser.add_argument('--fit_result', type=str, default=None)
args = argparser.parse_args()

index = args.index
fit_result_file = args.fit_result
import time

#Set the path for the data and plot, get the package path
import os
path = importlib.resources.files('tfpcbpggsz').joinpath('../benchmark')

data_path= os.path.join(path, 'data/')


os.makedirs(data_path,exist_ok=True)
os.makedirs(f'{data_path}/cpodd',exist_ok=True)
os.makedirs(f'{data_path}/cpeven',exist_ok=True)
os.makedirs(f'{data_path}/dks',exist_ok=True)

time1 = time.time()
#Load the fitted results
order=6
config = ConfigLoader(args.config)
config.get_order()

means = {}
errors = {}
for key in config.idx.keys():
    means[key] = config.get_sig_num(key)
    errors[key] = config.mass_fit_errors['sig_range_nsig']

#generate the yields with gaussian distributions
yields = {}
np.random.seed(int(time.time()%10000))
for key in config.idx.keys():
    yields[key] = int(np.random.normal(loc=means[key], scale=errors[key]))

#save the yields to a file
with open("gaussian_yields.npz", "wb") as f:
    np.savez(f, **yields)
#project path
fit_result = {}
import json
with open(fit_result_file, 'r') as f:
    fit_result = json.load(f)
coefficients = list(fit_result['Means'].values())

#Call the amplitude
Amplitude_D = Amplitude(model=config.amp.model_name)
Amplitude_D.init()
#Generating the B2DK signal 
pcgen = pcbpggsz_generator(amplitude=Amplitude_D)
pcgen.add_bias(correctionType="antiSym_legendre", order=order, coefficients=coefficients)

##CP odd 
np.random.seed(int(time.time()))

#CP odd
cp_odd_tags=['kspi0', 'kseta_gamgam', 'ksetap_pipieta', 'kseta_3pi', 'ksetap_gamrho', 'ksomega', 'klpi0pi0']
ret_cp_odd = {}

for tag in cp_odd_tags:
    n_cp_odd = yields[tag]
    ret_cp_odd[tag] = pcgen.generate(n_cp_odd, type="cp_odd")#8444
    save_as_root(ret_cp_odd[tag], f"{data_path}/cpodd/{tag}_{index}.root")

#CP even
cp_even_tags=['kk', 'pipi', 'pipipi0', 'kspi0pi0', 'klpi0']
ret_cp_even = {}
for tag in cp_even_tags:
    n_cp_even = yields[tag]
    ret_cp_even[tag] = pcgen.generate(n_cp_even, type="cp_even")#14646
    save_as_root(ret_cp_even[tag], f"{data_path}/cpeven/{tag}_{index}.root")

#Double Kspipi
cp_mixed_tags=['full', 'misspi', 'misspi0']
ret_sig = {}
ret_tag = {}

for tag in cp_mixed_tags:
    n_sig = yields[tag]
    ret_sig[tag], ret_tag[tag] = pcgen.generate(n_sig*2, type="cp_mixed")#10923
    ret=(ret_sig[tag], ret_tag[tag])
    save_as_root(ret, f"{data_path}/dks/{tag}_{index}.root")
print("Signal generated")
time2 = time.time()
#PHSP
phsp = PhaseSpaceGenerator().generate
phsp_cp_odd, phsp_cp_even = {}, {}
for tag in cp_odd_tags:
    phsp_cp_odd[tag] = phsp(yields[tag]*100)
    save_as_root(phsp_cp_odd[tag], f"{data_path}/cpodd/phsp_{tag}_{index}.root")
for tag in cp_even_tags:
    phsp_cp_even[tag] = phsp(yields[tag]*100)
    save_as_root(phsp_cp_even[tag], f"{data_path}/cpeven/phsp_{tag}_{index}.root")
phsp_sig, phsp_tag = {}, {}
srd_phsp_sig, srd_phsp_tag = {}, {}
for tag in cp_mixed_tags:
    phsp_sig[tag] = phsp(yields[tag]*100)
    phsp_tag[tag] = phsp(yields[tag]*100)
    save_as_root((phsp_sig[tag], phsp_tag[tag]), f"{data_path}/dks/phsp_{tag}_{index}.root")
time3 = time.time()

