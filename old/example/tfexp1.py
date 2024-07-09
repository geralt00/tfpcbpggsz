import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
tf.get_logger().setLevel('INFO')

import uproot as up
import numpy as np
import time
import iminuit
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

import matplotlib.pyplot as plt
import sys

time1 = time.time()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

sys.path.append('/software/pc24403/tfpcbpggsz/func')
sys.path.append('/software/pc24403/tfpcbpggsz/amp')
from amp import *


mc_path = '/shared/scratch/pc24403/amp'


amp_Data_dk_dd_p, ampbar_Data_dk_dd_p = getAmp('b2dk_DD', 'p')
amp_Data_dk_dd_m, ampbar_Data_dk_dd_m = getAmp('b2dk_DD', 'm')
amp_Data_dk_ll_p, ampbar_Data_dk_ll_p = getAmp('b2dk_LL', 'p')
amp_Data_dk_ll_m, ampbar_Data_dk_ll_m = getAmp('b2dk_LL', 'm')
amp_dk_dd_p = np.load(mc_path + '/Int_b2dk_DD_p_amp.npy')
ampbar_dk_dd_p = np.load(mc_path + '/Int_b2dk_DD_p_ampbar.npy')
amp_dk_dd_m = np.load(mc_path + '/Int_b2dk_DD_m_amp.npy')
ampbar_dk_dd_m = np.load(mc_path + '/Int_b2dk_DD_m_ampbar.npy')
amp_dk_ll_p = np.load(mc_path + '/Int_b2dk_LL_p_amp.npy')
ampbar_dk_ll_p = np.load(mc_path + '/Int_b2dk_LL_p_ampbar.npy')
amp_dk_ll_m = np.load(mc_path + '/Int_b2dk_LL_m_amp.npy')
ampbar_dk_ll_m = np.load(mc_path + '/Int_b2dk_LL_m_ampbar.npy')

#ampbar_Data_dk_dd_p, amp_Data_dk_dd_p = getAmp('b2dk_DD', 'p')
#ampbar_Data_dk_dd_m, amp_Data_dk_dd_m = getAmp('b2dk_DD', 'm')
#ampbar_Data_dk_ll_p, amp_Data_dk_ll_p = getAmp('b2dk_LL', 'p')
#ampbar_Data_dk_ll_m, amp_Data_dk_ll_m = getAmp('b2dk_LL', 'm')


#ampbar_dk_dd_p = np.load(mc_path + '/Int_b2dk_DD_p_amp.npy')
#amp_dk_dd_p = np.load(mc_path + '/Int_b2dk_DD_p_ampbar.npy')
#ampbar_dk_dd_m = np.load(mc_path + '/Int_b2dk_DD_m_amp.npy')
#amp_dk_dd_m = np.load(mc_path + '/Int_b2dk_DD_m_ampbar.npy')


#ampbar_dk_ll_p = np.load(mc_path + '/Int_b2dk_LL_p_amp.npy')
#amp_dk_ll_p = np.load(mc_path + '/Int_b2dk_LL_p_ampbar.npy')
#ampbar_dk_ll_m = np.load(mc_path + '/Int_b2dk_LL_m_amp.npy')
#amp_dk_ll_m = np.load(mc_path + '/Int_b2dk_LL_m_ampbar.npy')





def DeltadeltaD(A, Abar):
    var = tf.math.angle(A*np.conj(Abar))


    return var

def totalAmplitudeSquared_Integrated_crossTerm(A, Abar):
    '''
    This function calculates the total amplitude squared for the integrated decay, v0.1 only for MD fitted, no correction yet
    |A||Abar|cos(deltaD)
    |A||Abar|sin(deltaD)
    '''
    phase = DeltadeltaD(A, Abar)
    AAbar = tf.abs(A)*tf.abs(Abar)
    real_part = tf.math.reduce_mean(AAbar*tf.cos(phase))
    imag_part = tf.math.reduce_mean(AAbar*tf.sin(phase))

    return (real_part, imag_part)



def totalAmplitudeSquared_Integrated(Bsign=1, normA=1.1, normAbar=1.1, crossTerm=(0, 0), x=(0,0,0,0)):
    '''
    A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)

    A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)
    '''
    if Bsign == 1:
        xPlus = x[0]
        yPlus = x[1]
        rB2 = xPlus**2 + yPlus**2

        return (normA * rB2 + normAbar + 2*(xPlus *crossTerm[0] - yPlus * crossTerm[1]))
    
    else:
        xMinus = x[2]
        yMinus = x[3]
        rB2 = xMinus**2 + yMinus**2

        return (normA + normAbar  * rB2 + 2*(xMinus *crossTerm[0] + yMinus * crossTerm[1]))
    
def totalAmplitudeSquared_XY(Bsign=1, amp=[], ampbar=[], x=(0,0,0,0)):

    phase = DeltadeltaD(amp, ampbar)
    absA = tf.abs(amp)
    absAbar = tf.abs(ampbar)


    if Bsign == 1:
        xPlus = x[0]
        yPlus = x[1]
        rB2 = xPlus**2 + yPlus**2
        return (absA **2 * rB2 + absAbar**2 + 2 * (absA * absAbar) * (xPlus * tf.cos(phase) - yPlus * tf.sin(phase)))
    
    else:
        xMinus = x[2]
        yMinus = x[3]
        rB2 = xMinus**2 + yMinus**2

        return (absA**2  + absAbar **2 * rB2 + 2 * (absA * absAbar) * (xMinus * tf.cos(phase) + yMinus * tf.sin(phase)))

def nll_dk_ll(x):



    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_ll_p, ampbar_dk_ll_p)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_ll_m, ampbar_dk_ll_m)
    
    normA_p = tf.math.reduce_mean(tf.math.abs(amp_dk_ll_p)**2)
    normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dk_ll_p)**2)
    normA_m = tf.math.reduce_mean(tf.math.abs(amp_dk_ll_m)**2)
    normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dk_ll_m)**2)


    normalisation_Bplus = totalAmplitudeSquared_Integrated(1, normA_p, normAbar_p, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, normA_m, normAbar_m, normalisationCrossTerms_m, x)

    prob_p = totalAmplitudeSquared_XY(1, amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x)
    prob_m = totalAmplitudeSquared_XY(-1, amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x)



    ll_data_p = tf.math.log(prob_p/normalisation_Bplus)
    ll_data_m = tf.math.log(prob_m/normalisation_Bminus)


    return (tf.reduce_sum( -2* ll_data_p) + tf.reduce_sum( -2*ll_data_m))

def nll_dk_dd(x):


    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_dd_p, ampbar_dk_dd_p)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_dd_m, ampbar_dk_dd_m)
    normA_p = tf.math.reduce_mean(tf.math.abs(amp_dk_dd_p)**2)
    normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dk_dd_p)**2)
    normA_m = tf.math.reduce_mean(tf.math.abs(amp_dk_dd_m)**2)
    normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dk_dd_m)**2)


    normalisation_Bplus = totalAmplitudeSquared_Integrated(1, normA_p, normAbar_p, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, normA_m, normAbar_m, normalisationCrossTerms_m, x)

    prob_p = totalAmplitudeSquared_XY(1, amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x)
    prob_m = totalAmplitudeSquared_XY(-1, amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x)

    ll_data_p = tf.math.log(prob_p/normalisation_Bplus)
    ll_data_m = tf.math.log(prob_m/normalisation_Bminus)

    return (tf.reduce_sum(-2*ll_data_p) + tf.reduce_sum(-2*ll_data_m))

def nll_dk(x):
    print(nll_dk_ll(x)+nll_dk_dd(x))
    return nll_dk_dd(x)
    

with tf.device('GPU:1'):
    m = iminuit.Minuit(nll_dk, (0, 0, 0, 0.))
    mg = m.migrad()
    mg = m.scipy()
    print(mg)
time2 = time.time()
print(f'Fitting finished in {time2-time1} seconds')
