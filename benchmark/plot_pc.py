import tensorflow as tf
import numpy as np
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from tfpcbpggsz.tensorflow_wrapper import *
from matplotlib import pyplot as plt
from tfpcbpggsz.core import eff_fun
from tfpcbpggsz.phasecorrection import *
from tfpcbpggsz.generator.gen_pcbpggsz import *

pc = PhaseCorrection()
pc.PhaseCorrection()
pc.DEBUG = True
gen = PhaseSpaceGenerator()

ret = gen.generate(100000)

p1, p2, p3 = ret

s12, s13, s23 = get_mass(p1, p2), get_mass(p1, p3), get_mass(p2, p3)

srd = phsp_to_srd(s12, s13)
pc.correctionType="antiSym_legendre"
pc.order = 4
pc.PhaseCorrection()
fitted_coefficients = np.array([-0.609190, -1.826173, 0.765715, 0.695342, -0.936132, 0.208572])
pc.set_coefficients(fitted_coefficients)
value = pc.eval_corr(srd)
plt.scatter(s12, s13, c=value)
plt.colorbar()
plt.savefig("plot_pc.png")
plt.clf()
fitted_coefficients = np.array([-0.609190, -1.826173, 0.765715, 0.695342, -0.936132, 0.208572])
pc.set_coefficients(fitted_coefficients)
value = pc.eval_corr(srd)
plt.scatter(s12, s13, c=value)
plt.colorbar()
plt.savefig("plot_pc2.png")

