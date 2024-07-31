import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import numpy as np
import tensorflow as tf
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest,BaseGenerator,ARGenerator
from tfpcbpggsz.amp import *

Kspipi = PyD0ToKSpipi2018()
Kspipi.init()

import time


time1 = time.time()
def get_mass(p1,p2):
    """
    Calculate the invariant mass of two particles.
    """
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)

def amp(data):

    p1,p2,p3 = data
    amp = []
    for i in range(len(p1)):
        amp.append(Kspipi.AMP(p1[i].numpy().tolist(), p2[i].numpy().tolist(), p3[i].numpy().tolist()))
    return tf.abs(amp)**2


def single_sampling(phsp, amp, N):
    """
    Perform accept-reject sampling for a single chunk.
    """
    from tfpcbpggsz.generator.data import data_mask
    data = phsp(N)
    weight = amp(data)
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * max(weight) * 1.1 < weight
    print("Efficiency",tf.reduce_sum(tf.cast(cut, tf.float32)) / N)
    data = data_mask(data, cut)
    return data

m0 = 1.86483
m1 = 0.493677
m2 = 0.13957018
m3 = 0.13957018
mi=[m1,m2,m3]

gen = PhaseSpaceGenerator(m0,mi).generate

p1, p2, p3 = single_sampling(gen,amp,100000)
#Apparently, the efficiency is not good enough and slow


m12 = get_mass(p1,p2)
m13 = get_mass(p1,p3)

srd = phsp_to_srd(m12,m13)


plt.hist(m13.numpy(),bins=100,range=(0.3,3.2))
plt.show()
plt.savefig("m13.png")
plt.clf()
plt.hist2d(srd[:,0],srd[:,1],bins=100)
plt.show()
plt.savefig("srd.png")

time2 = time.time()
print("Gen + Plot: ",time2-time1)


'''
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
'''
