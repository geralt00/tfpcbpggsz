
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

pcgen = pcbpggsz_generator()
ret_sig, ret_tag = pcgen.generate(10000, type="cp_odd")

p1,p2,p3 = ret_sig


m12 = get_mass(p1,p2)
m13 = get_mass(p1,p3)

srd = phsp_to_srd(m12,m13)


plot_dir="/software/pc24403/tfpcbpggsz/tfpcbpggsz/generator/benchmark/plots/"
plt.hist(m12.numpy(),bins=50)
plt.show()
plt.savefig(f"{plot_dir}m12.png")
plt.clf()
plt.hist(m13.numpy(),bins=50)
plt.show()
plt.savefig(f"{plot_dir}m13.png")
plt.clf()
plt.hist2d(m12.numpy(),m13.numpy(),bins=50)
plt.show()
plt.savefig(f"{plot_dir}m12m13.png")

#plt.hist2d(srd[:,0],srd[:,1],bins=50)
#plt.show()
#plt.savefig(f"{plot_dir}srd.png")