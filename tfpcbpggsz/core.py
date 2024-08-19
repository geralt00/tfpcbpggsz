import tensorflow as tf
#import time
import uproot as up
import numpy as np
#import sys
#import iminuit
#from importlib.machinery import SourceFileLoader
#from tfpcbpggsz.masspdfs import *
#from tfpcbpggsz.phasecorrection import * #PhaseCorrection
_PI = tf.constant(np.pi, dtype=tf.float64)

class Module(object):
    pass




core = Module()


def name_convert(decay_str='b2dk_LL_p'):
    """
    Convert the decay string to the correct name for the data file
    """


    decay_str = decay_str.split('_')[0]+'_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    if decay_str.split('_')[0] == 'b2dpi':
        return 'DPi_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    elif decay_str.split('_')[0] == 'b2dk':
        return 'DK_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    

