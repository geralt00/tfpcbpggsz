
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

h_2d_Bp = make_2d_hist([srd_p[:,0],srd_p[:,1]],bins=[100,100])
h_2d_Bm = make_2d_hist([srd_m[:,0],srd_m[:,1]],bins=[100,100])

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
amp_phsp_p, ampbar_phsp_p = pcgen.amp(phsp_p), pcgen.ampbar(phsp_p)
amp_phsp_m, ampbar_phsp_m = pcgen.amp(phsp_m), pcgen.ampbar(phsp_m)
time3 = time.time()
print("B2DK amplitudes generated")

import tfpcbpggsz.core as core

ampMC={'b2dk_p':amp_phsp_p,'b2dk_m':amp_phsp_m}
ampbarMC={'b2dk_p':ampbar_phsp_p,'b2dk_m':ampbar_phsp_m}

Norm_p = core.Normalisation(ampMC, ampbarMC, 'b2dk_p')
Norm_m = core.Normalisation(ampMC, ampbarMC, 'b2dk_m')
Norm_p._DEBUG = True
Norm_m._DEBUG = True
Norm_p.initialise()
Norm_m.initialise()



@tf.function
def NLL(x):

    params = x
    Norm_p.setParams(params)
    Norm_m.setParams(params)

    prob_p = core.prob_totalAmplitudeSquared_XY(1, amp_p, ampbar_p, params)
    prob_m = core.prob_totalAmplitudeSquared_XY(-1, amp_m, ampbar_m, params)
    norm_p = Norm_p.Integrated_4p_sig(1)
    norm_m = Norm_m.Integrated_4p_sig(-1)

    nll_p = tf.reduce_sum(-2*tf.math.log(prob_p/norm_p))
    nll_m = tf.reduce_sum(-2*tf.math.log(prob_m/norm_m))

    return nll_p + nll_m

time4 = time.time()
print("NLL function defined")
from iminuit import Minuit

x = [0., 0., 0., 0.]

m = Minuit(NLL, x)
mg = m.migrad()
print(mg)   





