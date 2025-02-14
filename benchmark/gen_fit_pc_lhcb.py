
from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_test import *
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from plothist import plot_hist, make_hist 

import time


time1 = time.time()
#Generating the B2DK signal 
pcgen = pcbpggsz_generator()
pcgen.add_bias()
#B2DK
ret_Bp = pcgen.generate(6267, type="b2dh", gamma=68.7, rb=0.0904, dB=118.3, charge=1)
ret_Bm = pcgen.generate(6267, type="b2dh", gamma=68.7, rb=0.0904, dB=118.3, charge=-1)

p1_p,p2_p,p3_p = ret_Bp
p1_m,p2_m,p3_m = ret_Bm

m12_p = get_mass(p1_p,p2_p)
m13_p = get_mass(p1_p,p3_p)
m12_m = get_mass(p1_m,p2_m)
m13_m = get_mass(p1_m,p3_m)

srd_p = phsp_to_srd(m12_p,m13_p)
srd_m = phsp_to_srd(m12_m,m13_m)

plot_dir="/software/pc24403/tfpcbpggsz/benchmark/plots/"
os.makedirs(plot_dir,exist_ok=True)
from plothist import plot_hist, make_hist, make_2d_hist, plot_2d_hist

h_2d_Bp = make_2d_hist([srd_p[0],srd_p[1]],bins=[100,100])
h_2d_Bm = make_2d_hist([srd_m[0],srd_m[1]],bins=[100,100])

fig1, ax1, ax_colorbar1 = plot_2d_hist(h_2d_Bp, colorbar_kwargs={"label": "Entries"})
fig1.savefig(plot_dir+"B2DK_Bp_2d_hist.png")

fig2, ax2, ax_colorbar2 = plot_2d_hist(h_2d_Bm, colorbar_kwargs={"label": "Entries"})
fig2.savefig(plot_dir+"B2DK_Bm_2d_hist.png")

print("B2DK signal generated")
time2 = time.time()

#B2dk
amp_p, ampbar_p = pcgen.amp(ret_Bp), pcgen.ampbar(ret_Bp)
amp_m, ampbar_m = pcgen.amp(ret_Bm), pcgen.ampbar(ret_Bm)
#PHSP
phsp = PhaseSpaceGenerator().generate
phsp_p, phsp_m = phsp(1000000), phsp(1000000)
p1_phsp_p,p2_phsp_p,p3_phsp_p = phsp_p
p1_phsp_m,p2_phsp_m,p3_phsp_m = phsp_m
m12_phsp_p = get_mass(p1_phsp_p,p2_phsp_p)
m13_phsp_p = get_mass(p1_phsp_p,p3_phsp_p)
m12_phsp_m = get_mass(p1_phsp_m,p2_phsp_m)
m13_phsp_m = get_mass(p1_phsp_m,p3_phsp_m)
srd_phsp_p = phsp_to_srd(m12_phsp_p,m13_phsp_p)
srd_phsp_m = phsp_to_srd(m12_phsp_m,m13_phsp_m)
#srd_phsp_m = [srd_phsp_m[:,0],srd_phsp_m[:,1]]
#srd_phsp_p = [srd_phsp_p[:,0],srd_phsp_p[:,1]]

amp_phsp_p, ampbar_phsp_p = pcgen.amp(phsp_p), pcgen.ampbar(phsp_p)
amp_phsp_m, ampbar_phsp_m = pcgen.amp(phsp_m), pcgen.ampbar(phsp_m)
time3 = time.time()
print("B2DK amplitudes generated")

import tfpcbpggsz.core as core

ampMC={'b2dk_p':amp_phsp_p,'b2dk_m':amp_phsp_m}
ampbarMC={'b2dk_p':ampbar_phsp_p,'b2dk_m':ampbar_phsp_m}

Norm_p = core.Normalisation(ampMC, ampbarMC, 'b2dk_p')
Norm_m = core.Normalisation(ampMC, ampbarMC, 'b2dk_m')

Norm_p.initialise()
Norm_m.initialise()

srd_phsp_p_tag = (tf.gather(srd_phsp_p[0], Norm_p.tagged_i), tf.gather(srd_phsp_p[1], Norm_p.tagged_i))

from tfpcbpggsz.phasecorrection import PhaseCorrection
pc = PhaseCorrection()
pc.correctionType = "antiSym_legendre"
pc.order = 4
pc.PhaseCorrection()
pc.DEBUG = False



@tf.function
def NLL_LHCb(x):

    params = x
    pc.set_coefficients(params[4:])
    phase_correction_p = pc.eval_corr(srd_p)
    phase_correction_m = pc.eval_corr(srd_m)
    Norm_p.setParams(params)
    Norm_m.setParams(params)
    phase_correction_MC_p = pc.eval_corr(srd_phsp_p)
    phase_correction_MC_m = pc.eval_corr(srd_phsp_m)
    Norm_p.add_pc(phase_correction_MC_p)
    Norm_m.add_pc(phase_correction_MC_m)
    Norm_p.Update_crossTerms()
    Norm_m.Update_crossTerms()

    prob_p = core.prob_totalAmplitudeSquared_XY(1, amp_p, ampbar_p, params, phase_correction_p)
    prob_m = core.prob_totalAmplitudeSquared_XY(-1, amp_m, ampbar_m, params, phase_correction_m)
    norm_p = Norm_p.Integrated_4p_sig(1)
    norm_m = Norm_m.Integrated_4p_sig(-1)

    nll_p = tf.reduce_sum(-2*tf.math.log(prob_p/norm_p))
    nll_m = tf.reduce_sum(-2*tf.math.log(prob_m/norm_m))

    return nll_p + nll_m

@tf.function
def NLL(x):
    return   NLL_LHCb(x)

time4 = time.time()
print("NLL function defined")
from iminuit import Minuit

x_phase_correction = np.zeros((pc.nTerms_),dtype=np.float64)
#x = np.array([0., 0., 0., 0., *x_phase_correction],dtype=np.float64)

m = Minuit(NLL_LHCb, x_phase_correction)
mg = m.migrad()
print(mg)   

time5 = time.time()
#plot the phase correction in phase space
plot_phsp = phsp(100000)
p1_noeff,p2_noeff,p3_noeff = plot_phsp
m12_noeff = get_mass(p1_noeff,p2_noeff)
m13_noeff = get_mass(p1_noeff,p3_noeff)
srd_noeff = phsp_to_srd(m12_noeff,m13_noeff)
#pc_coeff=np.array(m.values[4:])
pc_coeff=np.array(m.values)
pc.set_coefficients(pc_coeff)
phase_correction_noeff = pc.eval_corr(srd_noeff)

plt.clf()
plt.scatter(srd_noeff[0],srd_noeff[1],c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+"PhaseCorrection_srd.png")
plt.clf()
plt.scatter(m12_noeff,m13_noeff,c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+"PhaseCorrection_mass.png")
print("Total time taken: ",time5-time1)