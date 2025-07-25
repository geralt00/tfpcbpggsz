
from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
import uproot as up
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, get_mass_bes
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_up import *
from tfpcbpggsz.amp_up.amp import GetCiSi
from tfpcbpggsz.core import DeltadeltaD
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from plothist import plot_hist, make_hist 

import time



D0ToKSpipi2018 = PyD0ToKSpipi2018()
D0ToKSpipi2018.init()

Getcisi = GetCiSi(D0ToKSpipi2018,binning_file='/software/pc24403/ampgen_build/options_and_data/binningSchemes/KsPiPi_modOptimal.txt')

#data = Getcisi.read_binning(binning_file='/software/pc24403/ampgen_build/options_and_data/binningSchemes/KsPiPi_modOptimal.txt')
#print(data)
time1 = time.time()
#Generating the B2DK signal 
pcgen = pcbpggsz_generator()
#pcgen.add_bias()
#CP odd 
#ret_cp_odd = pcgen.generate(8444, type="cp_odd")
#p1_cp_odd,p2_cp_odd,p3_cp_odd = ret_cp_odd
#m12_cp_odd = get_mass(p1_cp_odd,p2_cp_odd)
#m13_cp_odd = get_mass(p1_cp_odd,p3_cp_odd)
#srd_cp_odd = phsp_to_srd(m12_cp_odd,m13_cp_odd)
#
##CP even
#ret_cp_even = pcgen.generate(14646, type="cp_even")
#p1_cp_even,p2_cp_even,p3_cp_even = ret_cp_even
#m12_cp_even = get_mass(p1_cp_even,p2_cp_even)
#m13_cp_even = get_mass(p1_cp_even,p3_cp_even)
#srd_cp_even = phsp_to_srd(m12_cp_even,m13_cp_even)

#Double Kspipi
ret_sig, ret_tag = pcgen.generate(923, type="cp_mixed")

p1_sig,p2_sig,p3_sig = ret_sig
p1_tag,p2_tag,p3_tag = ret_tag
#Flavour tagging 143623
m12_sig = get_mass(p1_sig,p2_sig)
m13_sig = get_mass(p1_sig,p3_sig)
m12_tag = get_mass(p1_tag,p2_tag)
m13_tag = get_mass(p1_tag,p3_tag)

#Double Kspipi
amp_sig, ampbar_sig = pcgen.amp(ret_sig), pcgen.ampbar(ret_sig)
amp_tag, ampbar_tag = pcgen.amp(ret_tag), pcgen.ampbar(ret_tag)
model_phase_sig = DeltadeltaD(amp_sig, ampbar_sig)
model_phase_tag = DeltadeltaD(amp_tag, ampbar_tag)

event_sig = {'s12':m12_sig, 's13':m13_sig, 'amp':amp_sig, 'ampbar':ampbar_sig, 'model_phase':model_phase_sig}
event_tag = {'s12':m12_tag, 's13':m13_tag, 'amp':amp_tag, 'ampbar':ampbar_tag, 'model_phase':model_phase_tag}


ci, si, bins = Getcisi.get_cisi(event_sig)

fig, ax = plt.subplots(figsize=(8, 8))


x = np.linspace(-1, 1, 100)
y = np.sqrt(1 - x**2)

plt.plot(x, y, 'g--')
plt.plot(x, -y, 'g--')
plt.axis('equal')

# Set axis labels and title
ax.set_xlabel('Ci')
ax.set_ylabel('Si')
ax.set_title('Ci vs Si')


positive_bin = bins[bins>0]
negative_bin = bins[bins<0]

positive_ci = np.array([ci[f'{bin}'] for bin in positive_bin])
positive_si = np.array([si[f'{bin}'] for bin in positive_bin])
negative_ci = np.array([ci[f'{bin}'] for bin in negative_bin])
negative_si = np.array([si[f'{bin}'] for bin in negative_bin])

pos_ci_keys = [int(bin) for bin in positive_bin]
neg_ci_keys = [int(bin) for bin in negative_bin]

scatter = ax.scatter(positive_ci, positive_si, label='Positive bins', c=pos_ci_keys, cmap='Set1', s=50, alpha=0.8)
ax.set_xlabel('ci')
ax.set_ylabel('si')
ax.set_title('Modified optimal binning scheme')

for index, (x, y) in zip(pos_ci_keys, zip(positive_ci, positive_si)):
    ax.annotate(index, (x, y), textcoords="offset points", xytext=(0,10), ha='center')


cbar = plt.colorbar(scatter, ax=ax,location='right')
cbar.set_label('Bin index')

plt.show()
fig.savefig('cisi.png')

