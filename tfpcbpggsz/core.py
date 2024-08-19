import tensorflow as tf
import time
import uproot as up
import numpy as np
import sys
import iminuit
from importlib.machinery import SourceFileLoader
#from tfpcbpggsz.masspdfs import *
#from tfpcbpggsz.phasecorrection import * #PhaseCorrection


#Common functions
_PI = tf.constant(np.pi, dtype=tf.float64)
def DeltadeltaD(A, Abar):
    """
    Function to calculate the phase difference between the amplitude and the conjugate amplitude
    """
    var_a = tf.math.angle(A*np.conj(Abar))+ _PI
    var_b = tf.where(var_a > _PI, var_a - 2*_PI, var_a)
    var = tf.where(var_b < -_PI, var_b + 2*_PI, var_b)

    return var

def name_convert(decay_str='b2dk_LL_p'):
    """
    Convert the decay string to the correct name for the data file
    """


    decay_str = decay_str.split('_')[0]+'_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    if decay_str.split('_')[0] == 'b2dpi':
        return 'DPi_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    elif decay_str.split('_')[0] == 'b2dk':
        return 'DK_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    

