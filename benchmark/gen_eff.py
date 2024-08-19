
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
from plothist import plot_hist, make_hist, make_2d_hist, plot_2d_hist

pcgen = pcbpggsz_generator()
pcgen.apply_eff = True
pcgen.add_eff(charge='p', decay='b2dk_DD')
ret = pcgen.generate(1000000, type='phsp')
p1, p2, p3 = ret
m12 = get_mass(p1, p2)
m13 = get_mass(p1, p3)
srd = phsp_to_srd(m12, m13)


#h_2d_phsp = make_2d_hist([srd[0], srd[1]], bins=[100, 100])
h_2d_phsp_norm = make_2d_hist([m12, m13], bins=[100, 100])

#fig, ax, ax_colorbar = plot_2d_hist(h_2d_phsp, colorbar_kwargs={"label": "Entries"})
#fig.savefig("phsp_2d_hist.png")


fig2, ax2, ax_colorbar2 = plot_2d_hist(h_2d_phsp_norm, colorbar_kwargs={"label": "Entries"})
fig2.savefig("phsp_2d_hist_norm.png")
