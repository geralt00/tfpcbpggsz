
from tfpcbpggsz.tensorflow_wrapper import tf
import numpy as np
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, get_mass_bes
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_test import *
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from plothist import plot_hist, make_hist 
from tfpcbpggsz.variable import VarsManager
from tfpcbpggsz.bes.config_loader import ConfigLoader
import os
import time


time1 = time.time()
#Generating the B2DK signal 


#CP odd Kspipi
config = ConfigLoader("config_big.yml")
config.get_all_data()
config.vm

print("Signal generated")

time2 = time.time()

#PHSP
#phsp = PhaseSpaceGenerator().generate
phsp = PhaseSpaceGenerator().generate


time3 = time.time()
print("D decay amplitudes generated")
from tfpcbpggsz.bes.model import BaseModel

Model = BaseModel(config)

Model.pc.correctionType = "antiSym_legendre"
Model.pc.order = 1
Model.pc.PhaseCorrection()
Model.pc.DEBUG = False


time4 = time.time()
print("NLL function defined")
from iminuit import Minuit

var_args = {}
var_names = Model.vm.trainable_vars
x0 = []
for i in var_names:
    x0.append(tf.ones_like(Model.vm.get(i)).shape)
    var_args[i] = Model.vm.get(i)

m = Minuit(
    Model.fun,
    #var_args,
    np.array(x0),
    name=var_names,
)
mg = m.migrad()
print(mg)   

from tfpcbpggsz.bes.plotter import Plotter

Plotter = Plotter(Model)
Plotter.plot_cato('cp_odd')
Plotter.plot_cato('cp_even')
Plotter.plot_cato('dks')




time5 = time.time()

plot_dir="/software/pc24403/tfpcbpggsz/benchmark/plots/"
os.makedirs(plot_dir,exist_ok=True)
#plot the phase correction in phase space
plot_phsp = phsp(100000)
p1_noeff,p2_noeff,p3_noeff = plot_phsp
m12_noeff = get_mass(p1_noeff,p2_noeff)
m13_noeff = get_mass(p1_noeff,p3_noeff)
srd_noeff = phsp_to_srd(m12_noeff,m13_noeff)
pc_coeff=np.array(m.values)
phase_correction_noeff = Model.pc.eval_corr(srd_noeff)

plt.clf()
plt.scatter(srd_noeff[0],srd_noeff[1],c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+f"data_PhaseCorrection_srd_order{Model.pc.order}.png")
plt.clf()
plt.scatter(m12_noeff,m13_noeff,c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+f"data_PhaseCorrection_mass_order{Model.pc.order}.png")

#Save all the generated data
time6 = time.time()
np.savez(f"full_data_fit_order{Model.pc.order}.npz", 
         fitted_params=m.values, fitted_params_error= mg.errors,
         )
#import pickle
#with open('fitted_model.pkl', 'wb') as f:
#    pickle.dumps(Model, f)

print("All data saved")
print("Save data: ",time6-time1)