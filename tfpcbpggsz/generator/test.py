import numpy as np
import tensorflow as tf
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from matplotlib import pyplot as plt

import time


time1 = time.time()
def get_mass(p1,p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)


m0 = 1.86483
m1 = 0.493677
m2 = 0.13957018
m3 = 0.13957018
mi=[m1,m2,m3]

gen = PhaseSpaceGenerator(m0,mi)

p1, p2, p3 = gen.generate(1000000)

m12 = get_mass(p1,p2)
m13 = get_mass(p1,p3)
srd = phsp_to_srd(m12,m13)
plt.hist(m12.numpy(),bins=100)
plt.show()
plt.savefig("m12.png")
plt.hist2d(srd[:,0],srd[:,1],bins=100)
plt.show()
plt.savefig("srd.png")

time2 = time.time()
print("Gen + Plot: ",time2-time1)
branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]

#create dictionary to store momenta based on branch names
DalitzEvent = {}
for name in branch_names:
    DalitzEvent[name] = []

#fill dictionary with momenta
for i in range(len(p1)):
    DalitzEvent["_1_K0S0_E"].append(p1[i][0])
    DalitzEvent["_1_K0S0_Px"].append(p1[i][1])
    DalitzEvent["_1_K0S0_Py"].append(p1[i][2])
    DalitzEvent["_1_K0S0_Pz"].append(p1[i][3])

    DalitzEvent["_2_pi#_E"].append(p2[i][0])
    DalitzEvent["_2_pi#_Px"].append(p2[i][1])
    DalitzEvent["_2_pi#_Py"].append(p2[i][2])
    DalitzEvent["_2_pi#_Pz"].append(p2[i][3])

    DalitzEvent["_3_pi~_E"].append(p3[i][0])
    DalitzEvent["_3_pi~_Px"].append(p3[i][1])
    DalitzEvent["_3_pi~_Py"].append(p3[i][2])
    DalitzEvent["_3_pi~_Pz"].append(p3[i][3])
    DalitzEvent["mkp"] = m12
    DalitzEvent["mkm"] = m13
    
#save momenta to cache
np.save("DalitzEvent.npy",DalitzEvent)

time_save_root = time.time()
time3 = time.time()
print("Save Time: ",time3-time2)

