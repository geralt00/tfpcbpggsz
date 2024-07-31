
from tfpcbpggsz.tensorflow_wrapper import *
import numpy as np
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from matplotlib import pyplot as plt
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_test import *

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

    #time_cal_amp_start = time.time()
    p1,p2,p3 = data
    amp_i = Kspipi.AMP(p1.numpy().tolist(), p2.numpy().tolist(), p3.numpy().tolist())    
    amp_i = tf.cast(amp_i, tf.complex128)
    #time_cal_amp_end = time.time()
    #print("Time to calculate amplitude: ",time_cal_amp_end-time_cal_amp_start)
    return tf.cast(tf.abs(amp_i)**2, tf.float64)


def multi_sampling(
    phsp,
    amp,
    N,
    max_N=200000,
    force=True,
    max_weight=None,
    importance_f=None,
    display=True,
):


    a = GenTest(max_N, display=display)
    all_data = []

    for i in a.generate(N):
        data, new_max_weight = single_sampling2(
            phsp, amp, i, max_weight, importance_f
        )
        if max_weight is None:
            max_weight = new_max_weight * 1.1
        if new_max_weight > max_weight and len(all_data) > 0:
            tmp = data_merge(*all_data)
            rnd = tf.random.uniform((data_shape(tmp),), dtype=max_weight.dtype)
            cut = (
                rnd * new_max_weight / max_weight < 1.0
            )  # .max_amplitude < 1.0
            max_weight = new_max_weight * 1.05
            tmp = data_mask(tmp, cut)
            all_data = [tmp]
            a.set_gen(data_shape(tmp))
        a.add_gen(data_shape(data))
        # print(a.eff, a.N_gen, max_weight)
        all_data.append(data)

    ret = data_merge(*all_data)

    if force:
        cut = tf.range(data_shape(ret)) < N
        ret = data_mask(ret, cut)

    status = (a, max_weight)

    return ret, status

def single_sampling(phsp, amp, N):
    """
    Perform accept-reject sampling for a single chunk.
    """
    data = phsp(N)
    weight = amp(data)
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * max(weight) * 1.1 < weight
    data = data_mask(data, cut)
    return data

def single_sampling2(phsp, amp, N, max_weight=None, importance_f=None):

    data = phsp(N)
    weight = amp(data)
    weight = tf.cast(weight, tf.float64)
    #time_cal_importance_start = time.time()
    if importance_f is not None:
        print("Importance")
        weight = weight / importance_f(data)
    #time_cal_importance_end = time.time()
    #print("Time to calculate importance: ",time_cal_importance_end-time_cal_importance_start)

    new_max_weight = tf.reduce_max(weight)
    if max_weight is None or max_weight < new_max_weight:
        max_weight = new_max_weight * 1.01
    #time_cal_max_weight_end = time.time()
    #print("Time to calculate max weight: ",time_cal_max_weight_end-time_cal_importance_end)
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * max_weight < weight
    data = data_mask(data, cut)
    #time_mask_end = time.time()
    #print("Time to mask: ",time_mask_end-time_cal_importance_end)
    return data, max_weight




gen = PhaseSpaceGenerator().generate




N = 1000
max_N = 200000

ret, status = multi_sampling(
    gen,
    amp,
    N,
    force=True,
    max_N=max_N,
)

time_gen_end = time.time()
p1,p2,p3 = ret


m12 = get_mass(p1,p2)
m13 = get_mass(p1,p3)

srd = phsp_to_srd(m12,m13)

plt.hist(m12.numpy(),bins=50)
plt.show()
plt.savefig("m12.png")
plt.clf()
plt.hist(m13.numpy(),bins=50)
plt.show()
plt.savefig("m13.png")
plt.clf()
plt.hist2d(srd[:,0],srd[:,1],bins=50)
plt.show()
plt.savefig("srd.png")

time2 = time.time()
print("Gen: ",time_gen_end-time1)
print("Gen + Plot: ",time2-time1)

