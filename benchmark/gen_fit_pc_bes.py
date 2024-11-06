
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
#pcgen.add_bias()
#CP odd 
ret_cp_odd = pcgen.generate(8444, type="cp_odd")
p1_cp_odd,p2_cp_odd,p3_cp_odd = ret_cp_odd
m12_cp_odd = get_mass(p1_cp_odd,p2_cp_odd)
m13_cp_odd = get_mass(p1_cp_odd,p3_cp_odd)
srd_cp_odd = phsp_to_srd(m12_cp_odd,m13_cp_odd)

#CP even
ret_cp_even = pcgen.generate(14646, type="cp_even")
p1_cp_even,p2_cp_even,p3_cp_even = ret_cp_even
m12_cp_even = get_mass(p1_cp_even,p2_cp_even)
m13_cp_even = get_mass(p1_cp_even,p3_cp_even)
srd_cp_even = phsp_to_srd(m12_cp_even,m13_cp_even)

#Double Kspipi
ret_sig, ret_tag = pcgen.generate(10923, type="cp_mixed")
p1_sig,p2_sig,p3_sig = ret_sig
p1_tag,p2_tag,p3_tag = ret_tag

#Flavour tagging 143623
m12_sig = get_mass(p1_sig,p2_sig)
m13_sig = get_mass(p1_sig,p3_sig)
m12_tag = get_mass(p1_tag,p2_tag)
m13_tag = get_mass(p1_tag,p3_tag)

srd_sig = phsp_to_srd(m12_sig,m13_sig)
srd_tag = phsp_to_srd(m12_tag,m13_tag)


plot_dir="/software/pc24403/tfpcbpggsz/benchmark/plots/"
os.makedirs(plot_dir,exist_ok=True)
from plothist import plot_hist, make_hist, make_2d_hist, plot_2d_hist


h_2d_sig = make_2d_hist([srd_sig[0],srd_sig[1]],bins=[100,100])
h_2d_tag = make_2d_hist([srd_tag[0],srd_tag[1]],bins=[100,100])
h_2d_cp_odd = make_2d_hist([srd_cp_odd[0],srd_cp_odd[1]],bins=[100,100])
h_2d_cp_even = make_2d_hist([srd_cp_even[0],srd_cp_even[1]],bins=[100,100])


fig1, ax1, ax_colorbar1 = plot_2d_hist(h_2d_cp_odd, colorbar_kwargs={"label": "Entries"})
fig1.savefig(plot_dir+"cp_odd_2d_hist.png")

fig2, ax2, ax_colorbar2 = plot_2d_hist(h_2d_cp_even, colorbar_kwargs={"label": "Entries"})
fig2.savefig(plot_dir+"cp_even_2d_hist.png")

fig3, ax3, ax_colorbar3 = plot_2d_hist(h_2d_sig, colorbar_kwargs={"label": "Entries"})
fig3.savefig(plot_dir+"cp_mixed_sig_2d_hist.png")

fig4, ax4, ax_colorbar4 = plot_2d_hist(h_2d_tag, colorbar_kwargs={"label": "Entries"})
fig4.savefig(plot_dir+"cp_mixed_tag_2d_hist.png")


print("Signal generated")
time2 = time.time()

#cp_odd
amp_cp_odd, ampbar_cp_odd = pcgen.amp(ret_cp_odd), pcgen.ampbar(ret_cp_odd)
#cp_even
amp_cp_even, ampbar_cp_even = pcgen.amp(ret_cp_even), pcgen.ampbar(ret_cp_even)

#Double Kspipi
amp_sig, ampbar_sig = pcgen.amp(ret_sig), pcgen.ampbar(ret_sig)
amp_tag, ampbar_tag = pcgen.amp(ret_tag), pcgen.ampbar(ret_tag)


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
time3 = time.time()

amp_phsp_p, ampbar_phsp_p = pcgen.amp(phsp_p), pcgen.ampbar(phsp_p)
amp_phsp_m, ampbar_phsp_m = pcgen.amp(phsp_m), pcgen.ampbar(phsp_m)
time3 = time.time()
print("D decay amplitudes generated")

import tfpcbpggsz.core as core

ampMC={'charm_p':amp_phsp_p,'charm_m':amp_phsp_m,'charm_sig':amp_phsp_p,'charm_tag':amp_phsp_m}
ampbarMC={'charm_p':ampbar_phsp_p,'charm_m':ampbar_phsp_m,'charm_sig':ampbar_phsp_p,'charm_tag':ampbar_phsp_m}



Norm_p = core.Normalisation(ampMC, ampbarMC, 'charm_sig')
Norm_cp_odd = core.Normalisation(ampMC, ampbarMC,'charm_p')
Norm_cp_even = core.Normalisation(ampMC, ampbarMC,'charm_p')


Norm_p.initialise()
Norm_cp_odd.initialise()
Norm_cp_even.initialise()

from tfpcbpggsz.phasecorrection import PhaseCorrection
pc = PhaseCorrection()
pc.correctionType = "antiSym_legendre"
pc.order = 4
pc.PhaseCorrection()
pc.DEBUG = False


@tf.function
def NLL_kspipi(x):

    params = x
    pc.set_coefficients(params)

    phase_correction_sig = pc.eval_corr(srd_sig)
    phase_correction_tag = pc.eval_corr(srd_tag)
    Norm_p.setParams(params)
    phase_correction_MC_sig = pc.eval_corr(srd_phsp_p)
    phase_correction_MC_tag = pc.eval_corr(srd_phsp_m)

    Norm_p.add_pc(phase_correction_MC_sig, pc_tag=phase_correction_MC_tag)
    Norm_p.Update_crossTerms()

    prob = core.prob_totalAmplitudeSquared_CP_mix(amp_sig, ampbar_sig,amp_tag, ampbar_tag, phase_correction_sig, phase_correction_tag)
    norm = Norm_p._crossTerms_complex

    nll = tf.reduce_sum(-2*tf.math.log(prob/norm))

    return nll


@tf.function
def NLL_CP_odd(x, Dsign):
    params = x
    pc.set_coefficients(params)
    phase_correction = pc.eval_corr(srd_cp_odd)
    Norm_cp_odd.setParams(params)
    phase_correction_MC = pc.eval_corr(srd_phsp_p)
    Norm_cp_odd.add_pc(phase_correction_MC)
    Norm_cp_odd.Update_crossTerms()

    prob = core.prob_totalAmplitudeSquared_CP_tag(Dsign, amp_cp_odd, ampbar_cp_odd, phase_correction)
    norm = Norm_cp_odd.Integrated_CP_tag(Dsign)

    nll = tf.reduce_sum(-2*tf.math.log(prob/norm))

    return nll

@tf.function
def NLL_CP_even(x, Dsign):
    params = x
    pc.set_coefficients(params)
    phase_correction = pc.eval_corr(srd_cp_even)
    Norm_cp_even.setParams(params)
    phase_correction_MC = pc.eval_corr(srd_phsp_p)
    Norm_cp_even.add_pc(phase_correction_MC)
    Norm_cp_even.Update_crossTerms()

    prob = core.prob_totalAmplitudeSquared_CP_tag(Dsign, amp_cp_even, ampbar_cp_even, phase_correction)
    norm = Norm_cp_even.Integrated_CP_tag(Dsign)

    nll = tf.reduce_sum(-2*tf.math.log(prob/norm))

    return nll


@tf.function
def NLL(x):
    return   NLL_kspipi(x) + NLL_CP_even(x, 1) + NLL_CP_odd(x, -1)


time4 = time.time()
print("NLL function defined")
from iminuit import Minuit

x_phase_correction = np.zeros((pc.nTerms_),dtype=np.float64)

m = Minuit(NLL, x_phase_correction)
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