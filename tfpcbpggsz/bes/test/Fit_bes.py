
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
from tfpcbpggsz.bes.plotter import Plotter
import os
import time


time1 = time.time()
#Generating the B2DK signal 


#CP odd Kspipi
config = ConfigLoader("config.yml")
config.get_all_data()
config.vm

'''
s12_sig, s13_sig = [], []
for tag in config.idx:
    s12_sig_i, s13_sig_i = config.get_data_mass(tag=tag)
    s12_sig.append(s12_sig_i)
    s13_sig.append(s13_sig_i)

s12_sig = np.concatenate(s12_sig)
s13_sig = np.concatenate(s13_sig)



from plothist import plot_hist, make_hist, make_2d_hist, plot_2d_hist


h_2d_sig = make_2d_hist([s12_sig,s13_sig],bins=[100,100],range=[[0.3,3.2],[0.3,3.2]])


fig1, ax1, ax_colorbar1 = plot_2d_hist(h_2d_sig, colorbar_kwargs={"label": "Entries"})
fig1.savefig(plot_dir+"data_cp_odd_sig_2d_hist.png")

'''
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
Model.pc.order = 4
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

Plotter = Plotter(Model)
Plotter.plot_cato('cp_odd')
#Plotter.plot_cato('cp_even')



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
plt.savefig(plot_dir+"data_PhaseCorrection_srd.png")
plt.clf()
plt.scatter(m12_noeff,m13_noeff,c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+"data_PhaseCorrection_mass.png")

print("Total time taken: ",time5-time1)