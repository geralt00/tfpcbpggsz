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
pc.order = 6
pc.PhaseCorrection()
fitted_coefficients = np.array([0.054954, -1.064545, 0.262564, -2.424909, -3.430244, -2.056977, 0.249967, 1.159846, -0.656559, -0.722374, 0.134728, -0.143797])
pc.set_coefficients(fitted_coefficients)
value = pc.eval_corr_tf(srd)
plt.scatter(s12, s13, c=value)
plt.colorbar()
plt.savefig("plot_pc.png")
plt.clf()
fitted_coefficients = np.array([-0.200354, 100, 0.110016, -1.747495, -2.797758, -1.596963, 0.10734, 1.08292, -0.34064, -0.400786, 0.081452, 0.125092])
pc.set_coefficients(fitted_coefficients)
value = pc.eval_corr_tf(srd)
plt.scatter(s12, s13, c=value)
plt.colorbar()
plt.savefig("plot_pc2.png")

