import numpy as np
import tensorflow as tf
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.generator.gen_pcbpggsz import pcbpggsz_generator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from matplotlib import pyplot as plt

import time


time1 = time.time()
def get_mass(p1,p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)



#gen = PhaseSpaceGenerator(m0,mi)
gen = PhaseSpaceGenerator()
p1, p2, p3 = gen.generate(1000)
pcgen = pcbpggsz_generator()

amp, ampbar = pcgen.amp((p1, p2, p3)), pcgen.ampbar((p1, p2, p3)) 

m12 = get_mass(p1,p2)
m13 = get_mass(p1,p3)
srd = phsp_to_srd(m12,m13)
plt.hist(m12.numpy(),bins=100, weights=np.abs(amp)**2)
plt.show()
plt.savefig("m12.png")
plt.clf()
#plt.hist2d(srd[:,0],srd[:,1],bins=100)
#plt.show()
#plt.savefig("srd.png")



plt.hist(srd[1], weights=np.abs(amp)**2)
plt.savefig('weighted.png')


time2 = time.time()
print("Gen + Plot: ",time2-time1)
#Save as a tensor
DalitzEvent = {}
DalitzEvent["P4_Ks"] = p1
DalitzEvent["P4_pip"] = p2
DalitzEvent["P3_pim"] = p3
DalitzEvent["mkp"] = m12
DalitzEvent["mkm"] = m13



    
#save momenta to cache
np.save("DalitzEvent.npy",DalitzEvent)

time_save_root = time.time()
time3 = time.time()
print("Save Time: ",time3-time2)

