
from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, get_mass_bes
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_test import *
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from plothist import plot_hist, make_hist 

import time


time1 = time.time()
#Generating the B2DK signal 


#Double Kspipi
data_kspipi = up.open("/software/pc24403/tfpcbpggsz/benchmark/data/qcmc/qcmc.root:nt")
branches = ['p4_Ks','p4_pim','p4_pip','p4_Ks2','p4_pim2','p4_pip2']
data_arr_old = data_kspipi.arrays(branches, cut="(mBC1>1.86) & (mBC1<1.87)")
data_arr_old = data_arr_old[:5800]
data_arr = {}
data_arr['p4_Ks'] = np.array([data_arr_old['p4_Ks'][:,3],data_arr_old['p4_Ks'][:,0],data_arr_old['p4_Ks'][:,1],data_arr_old['p4_Ks'][:,2]]).T
data_arr['p4_pim'] = np.array([data_arr_old['p4_pim'][:,3],data_arr_old['p4_pim'][:,0],data_arr_old['p4_pim'][:,1],data_arr_old['p4_pim'][:,2]]).T
data_arr['p4_pip'] = np.array([data_arr_old['p4_pip'][:,3],data_arr_old['p4_pip'][:,0],data_arr_old['p4_pip'][:,1],data_arr_old['p4_pip'][:,2]]).T
data_arr['p4_Ks2'] = np.array([data_arr_old['p4_Ks2'][:,3],data_arr_old['p4_Ks2'][:,0],data_arr_old['p4_Ks2'][:,1],data_arr_old['p4_Ks2'][:,2]]).T
data_arr['p4_pim2'] = np.array([data_arr_old['p4_pim2'][:,3],data_arr_old['p4_pim2'][:,0],data_arr_old['p4_pim2'][:,1],data_arr_old['p4_pim2'][:,2]]).T
data_arr['p4_pip2'] = np.array([data_arr_old['p4_pip2'][:,3],data_arr_old['p4_pip2'][:,0],data_arr_old['p4_pip2'][:,1],data_arr_old['p4_pip2'][:,2]]).T


ret_sig, ret_tag = [data_arr['p4_Ks'],data_arr['p4_pip'],data_arr['p4_pim']],[data_arr['p4_Ks2'],data_arr['p4_pip2'],data_arr['p4_pim2']]
p1_sig,p2_sig,p3_sig = ret_sig
p1_tag,p2_tag,p3_tag = ret_tag

#p1_sig = np.array([p1_sig[:,3],p1_sig[:,0],p1_sig[:,1],p1_sig[:,2]]).T
#p2_sig = np.array([p2_sig[:,3],p2_sig[:,0],p2_sig[:,1],p2_sig[:,2]]).T
#p3_sig = np.array([p3_sig[:,3],p3_sig[:,0],p3_sig[:,1],p3_sig[:,2]]).T
#p1_tag = np.array([p1_tag[:,3],p1_tag[:,0],p1_tag[:,1],p1_tag[:,2]]).T
#p2_tag = np.array([p2_tag[:,3],p2_tag[:,0],p2_tag[:,1],p2_tag[:,2]]).T
#p3_tag = np.array([p3_tag[:,3],p3_tag[:,0],p3_tag[:,1],p3_tag[:,2]]).T


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


fig3, ax3, ax_colorbar3 = plot_2d_hist(h_2d_sig, colorbar_kwargs={"label": "Entries"})
fig3.savefig(plot_dir+"data_cp_mixed_sig_2d_hist.png")

fig4, ax4, ax_colorbar4 = plot_2d_hist(h_2d_tag, colorbar_kwargs={"label": "Entries"})
fig4.savefig(plot_dir+"data_cp_mixed_tag_2d_hist.png")


print("Signal generated")
time2 = time.time()

pcgen = pcbpggsz_generator()

#Double Kspipi
amp_sig, ampbar_sig = pcgen.amp(ret_sig), pcgen.ampbar(ret_sig)
amp_tag, ampbar_tag = pcgen.amp(ret_tag), pcgen.ampbar(ret_tag)


#PHSP
#phsp = PhaseSpaceGenerator().generate
phsp = PhaseSpaceGenerator().generate

phsp_kspipi = up.open("/software/pc24403/tfpcbpggsz/benchmark/data/phsp/phsp.root:nt")
branches = ['p4_Ks','p4_pim','p4_pip','p4_Ks2','p4_pim2','p4_pip2']
phsp_arr_old = phsp_kspipi.arrays(branches, cut="(mBC1>1.86) & (mBC1<1.87)")

phsp_arr = {}
phsp_arr['p4_Ks'] = np.array([phsp_arr_old['p4_Ks'][:,3],phsp_arr_old['p4_Ks'][:,0],phsp_arr_old['p4_Ks'][:,1],phsp_arr_old['p4_Ks'][:,2]]).T
phsp_arr['p4_pim'] = np.array([phsp_arr_old['p4_pim'][:,3],phsp_arr_old['p4_pim'][:,0],phsp_arr_old['p4_pim'][:,1],phsp_arr_old['p4_pim'][:,2]]).T
phsp_arr['p4_pip'] = np.array([phsp_arr_old['p4_pip'][:,3],phsp_arr_old['p4_pip'][:,0],phsp_arr_old['p4_pip'][:,1],phsp_arr_old['p4_pip'][:,2]]).T
phsp_arr['p4_Ks2'] = np.array([phsp_arr_old['p4_Ks2'][:,3],phsp_arr_old['p4_Ks2'][:,0],phsp_arr_old['p4_Ks2'][:,1],phsp_arr_old['p4_Ks2'][:,2]]).T
phsp_arr['p4_pim2'] = np.array([phsp_arr_old['p4_pim2'][:,3],phsp_arr_old['p4_pim2'][:,0],phsp_arr_old['p4_pim2'][:,1],phsp_arr_old['p4_pim2'][:,2]]).T
phsp_arr['p4_pip2'] = np.array([phsp_arr_old['p4_pip2'][:,3],phsp_arr_old['p4_pip2'][:,0],phsp_arr_old['p4_pip2'][:,1],phsp_arr_old['p4_pip2'][:,2]]).T

phsp_p, phsp_m = [phsp_arr['p4_Ks'],phsp_arr['p4_pip'],phsp_arr['p4_pim']],[phsp_arr['p4_Ks2'],phsp_arr['p4_pip2'],phsp_arr['p4_pim2']]
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

ampMC={'charm_sig':amp_phsp_p,'charm_tag':amp_phsp_m}
ampbarMC={'charm_sig':ampbar_phsp_p,'charm_tag':ampbar_phsp_m}



Norm_cpmix = core.Normalisation(ampMC, ampbarMC, 'charm_sig')
Norm_cpmix.initialise()

from tfpcbpggsz.phasecorrection import PhaseCorrection
pc = PhaseCorrection()
pc.correctionType = "antiSym_legendre"
pc.order = 3
pc.PhaseCorrection()
pc.DEBUG = False


@tf.function
def NLL_kspipi(x):

    params = x
    pc.set_coefficients(params)

    phase_correction_sig = pc.eval_corr(srd_sig)
    phase_correction_tag = pc.eval_corr(srd_tag)
    Norm_cpmix.setParams(params)
    phase_correction_MC_sig = pc.eval_corr(srd_phsp_p)
    phase_correction_MC_tag = pc.eval_corr(srd_phsp_m)
    Norm_cpmix.add_pc(phase_correction_MC_sig, pc_tag=phase_correction_MC_tag)
    Norm_cpmix.Update_crossTerms()

    prob = core.prob_totalAmplitudeSquared_CP_mix(amp_sig, ampbar_sig,amp_tag, ampbar_tag, phase_correction_sig, phase_correction_tag)
    norm = Norm_cpmix._crossTerms_complex

    nll = tf.reduce_sum(-2*tf.math.log(prob/norm))

    return nll


@tf.function
def NLL(x):
    return   NLL_kspipi(x)

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
pc_coeff=np.array(m.values)
pc.set_coefficients(pc_coeff)
phase_correction_noeff = pc.eval_corr(srd_noeff)

plt.clf()
plt.scatter(srd_noeff[0],srd_noeff[1],c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+"data_PhaseCorrection_srd.png")
plt.clf()
plt.scatter(m12_noeff,m13_noeff,c=phase_correction_noeff)
plt.colorbar()
plt.savefig(plot_dir+"data_PhaseCorrection_mass.png")
print("Total time taken: ",time5-time1)
