import tensorflow as tf
import time
import uproot as up
import numpy as np
import sys
import iminuit
from importlib.machinery import SourceFileLoader
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.phasecorrection import * #PhaseCorrection


#Common functions
_PI = tf.constant(np.pi, dtype=tf.float64)
def DeltadeltaD(A, Abar):
    """
    Function to calculate the phase difference between the amplitude and the conjugate amplitude
    
    Args:
        A (Amplitude): the amplitude from sample
        Abar (Amplitude Bar): the conjugate amplitude from sample

    Returns:
        float64: phase difference between the amplitude and the conjugate amplitude
    """
    var_a = tf.math.angle(A*np.conj(Abar))+ _PI
    var_b = tf.where(var_a > _PI, var_a - 2*_PI, var_a)
    var = tf.where(var_b < -_PI, var_b + 2*_PI, var_b)

    return var

def name_convert(decay_str='b2dk_LL_p'):


    decay_str = decay_str.split('_')[0]+'_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    if decay_str.split('_')[0] == 'b2dpi':
        return 'DPi_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    elif decay_str.split('_')[0] == 'b2dk':
        return 'DK_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    
def clip_log(x, _epsilon=1e-6):
    """clip log to allowed large value"""
    x_cut = tf.where(x > _epsilon, x, tf.ones_like(x) * _epsilon)
    b_t = tf.math.log(x_cut)

    delta_x = x - _epsilon
    b_f = (
       np.log(_epsilon) + delta_x / _epsilon - (delta_x / _epsilon) ** 2 / 2.0
    )
    return tf.where(x > _epsilon, b_t, b_f)

def dalitz_transform(x_valid, y_valid):
    rotatedSymCoord = (y_valid + x_valid)/2  #z_+
    rotatedAntiSymCoord = (y_valid - x_valid)/2 #z_-

    m1_ = 2.23407421671132946
    c1_ = -3.1171885586526695
    m2_ = 0.8051636393861085
    c2_ = -9.54231895051727e-05

    stretchedSymCoord = m1_ * rotatedSymCoord + c1_
    stretchedAntiSymCoord = m2_ * rotatedAntiSymCoord + c2_
    antiSym_scale = 2.0
    antiSym_offset = 2.0
    stretchedAntiSymCoord_dp = (antiSym_scale * (stretchedAntiSymCoord)) / (antiSym_offset + stretchedSymCoord)
    return np.array([stretchedSymCoord, stretchedAntiSymCoord_dp])

def eff_fun(x, charge='p', decay='dk_LL'):
    """
    Function to calculate the efficiency value for the B2DK and B2Dpi decays
    
    """
    # in GeV !!
    res = 0
    zp_p = x[0]  # $z_{+}^{\prime}
    zm_pp = x[1]  # $z_{-}^{\prime\prime}
    Legendre_zp_1 = zp_p
    Legendre_zp_2 = (3 * np.power(zp_p, 2) - 1) / 2.
    Legendre_zp_3 = (5 * np.power(zp_p, 3) - 3 * zp_p) / 2.
    Legendre_zp_4 = (35 * np.power(zp_p, 4) - 30 * np.power(zp_p, 2) + 3) / 8.
    Legendre_zp_5 = (36 * np.power(zp_p, 5) - 70 * np.power(zp_p, 3) + 15 * zp_p) / 8.
    Legendre_zp_6 = (231 * np.power(zp_p, 6) - 315 * np.power(zp_p, 4) + 105 * np.power(zp_p, 2) - 5) / 16.
    Legendre_zm_2 = (3 * np.power(zm_pp, 2) - 1) / 2.
    Legendre_zm_4 = (35 * np.power(zm_pp, 4) - 30 * np.power(zm_pp, 2) + 3) / 8.
    Legendre_zm_6 = (231 * np.power(zm_pp, 6) - 315 * np.power(zm_pp, 4) + 105 * np.power(zm_pp, 2) - 5) / 16.
    params = {}
    offset = {}
    mean = {}
    params['bp_b2dk_LL'] = [-9.53264, 30.6442, -70.1242, -171.659, 20.997, 111.455, -25.7043, -208.538, -131.828, 5.30961, 38.3281, 52.9287, -45.1955, -92.5608, -53.7144, -6.41475]
    params['bm_b2dk_LL'] = [-11.6916, 36.6738, -75.5949, -177.263, 24.6625, 128.041, -29.1374, -222.949, -133.375, 5.42292, 50.5985, 63.6018, -44.7697, -94.6231, -65.164, -7.16556]
    params['bp_b2dpi_LL'] = [-11.726, 36.3723, -79.9267, -178.024, 26.746, 125.069, -31.2218, -233.282, -133.757, 6.55028, 48.6604, 59.0497, -43.2816, -99.2002, -65.8039, -6.45939]
    params['bm_b2dpi_LL'] = [-11.2904, 33.8601, -76.9433, -178.023, 25.2621, 118.649, -30.6079, -226.689, -135.131, 5.66233, 46.1495, 57.1009, -45.8845, -98.3241, -65.0337, -6.8956]
    params['bp_b2dk_DD'] = [-30.8217, 98.3102, -201.671, -450.003, 67.5668, 336.798, -77.9681, -580.061, -337.139, 17.1748, 135.283, 166.737, -108.95, -248.926, -160.071, -16.7914]
    params['bm_b2dk_DD'] = [-28.5793, 92.1537, -192.655, -445.199, 66.4849, 315.014, -76.3822, -547.47, -336.686, 17.0215, 130.277, 144.332, -112.366, -224.548, -150.207, -19.5028]
    params['bp_b2dpi_DD'] = [-25.1893, 83.6515, -191.42, -437.186, 59.6098, 286.787, -79.8558, -540.406, -331.55, 13.7906, 114.149, 135.039, -114.09, -219.42, -167.023, -22.0707 ]
    params['bm_b2dpi_DD'] = [-27.3281, 77.6543, -181.93, -451.338, 51.6441, 278.849, -66.5276, -522.833, -353.474, 11.865, 98.5612, 138.88, -121.508, -228.435, -139.317, -15.4476 ]
    offset['bp_b2dk_LL'] = 57.060094380400386
    offset['bm_b2dk_LL'] = 58.17587468590285
    offset['bp_b2dk_DD'] = 144.4818375757835
    offset['bm_b2dk_DD'] = 149.4970954503175
    offset['bp_b2dpi_LL'] = 58.02066608312892
    offset['bm_b2dpi_LL'] = 57.2679149533671
    offset['bp_b2dpi_DD'] = 143.85275330726384
    offset['bm_b2dpi_DD'] = 145.64578485313336
    mean['bp_b2dk_LL'] = 94.71806024672841
    mean['bm_b2dk_LL'] = 95.67192393339243
    mean['bp_b2dk_DD'] = 236.7559605125362
    mean['bm_b2dk_DD'] = 243.96305838750337
    mean['bp_b2dpi_LL'] = 95.5957545077163
    mean['bm_b2dpi_LL'] = 95.47377527333916
    mean['bp_b2dpi_DD'] = 238.0281142413688
    mean['bm_b2dpi_DD'] = 241.6462132232453

    decay = 'b'+charge+'_'+decay

    res = (
        params[decay][0]
        + params[decay][1] * Legendre_zp_1
        + params[decay][2] * Legendre_zp_2
        + params[decay][3] * Legendre_zm_2
        + params[decay][4] * Legendre_zp_3
        + params[decay][5] * Legendre_zp_1 * Legendre_zm_2
        + params[decay][6] * Legendre_zp_4
        + params[decay][7] * Legendre_zp_2 * Legendre_zm_2
        + params[decay][8] * Legendre_zm_4
        + params[decay][9] * Legendre_zp_5
        + params[decay][10] * Legendre_zm_2 * Legendre_zp_3
        + params[decay][11] * Legendre_zm_4 * Legendre_zp_1
        + params[decay][12] * Legendre_zm_6
        + params[decay][13] * Legendre_zm_4 * Legendre_zp_2
        + params[decay][14] * Legendre_zm_2 * Legendre_zp_4
        + params[decay][15] * Legendre_zp_6
    )

    return( res+offset[decay])/mean[decay]

def prob_totalAmplitudeSquared_XY(Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0,0), pc=None):
    """
    Function to calculate the amplitude squared for the B2DK and B2Dpi decays

    Args:
        Bsign (int, optional): the charge of B meson. Defaults to 1.
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        x (tuple, optional): the fitted params. Defaults to (0,0,0,0,0,0).
        pc (float, optional): the phase correction class. Defaults to None.

    Returns:
        float64: the amplitude squared
    """

    phase = DeltadeltaD(amp, ampbar)
    if pc is not None:
        phase = phase + pc

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)


    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)

        return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus * tf.cos(phase) - yPlus * tf.sin(phase)))
    
    elif Bsign == -1:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)

        return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus * tf.cos(phase) + yMinus * tf.sin(phase)))

def prob_totalAmplitudeSquared_CP_mix(amp_sig=[], ampbar_sig=[],amp_tag=[], ampbar_tag=[], pc_sig=None, pc_tag=None):
    """
    Function to calculate the amplitude squared for the D0->KsPiPi and D0bar->KsPiPi decays

    Args:
        amp_sig (Amplitude): the amplitude from signal sample
        ampbar_sig (Amplitude Bar): the conjugate amplitude from signal sample
        amp_tag (Amplitude): the amplitude from tag sample
        ampbar_tag (Amplitude Bar): the conjugate amplitude from tag sample
        pc_sig (float, optional): the phase correction for the signal. Defaults to None.
        pc_tag (float, optional): the phase correction for the tag. Defaults to None.

    Returns:
        float64: the amplitude squared
    """


    phase_sig = DeltadeltaD(amp_sig, ampbar_sig)
    phase_tag = DeltadeltaD(amp_tag, ampbar_tag)

    if pc_sig is not None:
        phase_sig = phase_sig + pc_sig
    if pc_tag is not None:
        phase_tag = phase_tag + pc_tag

    absA_sig = tf.cast(tf.abs(amp_sig), tf.float64)
    absAbar_sig = tf.cast(tf.abs(ampbar_sig), tf.float64)
    absA_tag = tf.cast(tf.abs(amp_tag), tf.float64)
    absAbar_tag = tf.cast(tf.abs(ampbar_tag), tf.float64)

    return (absA_sig*absAbar_tag)**2 + (absAbar_sig*absA_tag)**2 - 2*absA_sig*absAbar_tag*absAbar_sig*absA_tag*tf.math.cos(phase_sig-phase_tag)

def prob_totalAmplitudeSquared_CP_tag(CPsign=1, amp=[], ampbar=[], pc=None):
    """
    Function to calculate the amplitude squared for the CP tag decay

    Args:
        CPsign (int, optional): the CP sign for the decay. Defaults to +.
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        pc (float, optional): the phase correction class. Defaults to None.

    Returns:
        float64: the amplitude squared
    """



    DDsign = -1
    phase = DeltadeltaD(amp, ampbar)
    if pc is not None:
        phase = phase + pc

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)



    return (absA**2  + absAbar **2  + 2.0 * DDsign * CPsign * (absA * absAbar) * tf.cos(phase))

def prob_totalAmplitudeSquared_DPi_XY( Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0,0,0,0,0,0,0,0,0,0)):
    """
    Function to calculate the amplitude squared for the B2Dpi decay, the ratio between B2DK and B2Dpi is used, then two additional parameters are added

    Args:
        Bsign (int, optional): the charge of B meson. Defaults to 1.
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        x (tuple, optional): the fitted params. Defaults to (0,0,0,0,0,0,0,0,0,0,0,0,0,0).

    Returns:
        float64: the amplitude squared
    """

    phase = DeltadeltaD(amp, ampbar)
    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)

    xXi = tf.cast(x[4], tf.float64)
    yXi = tf.cast(x[5], tf.float64)

    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
        yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)
        rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)

        return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus_DPi * tf.cos(phase) - yPlus_DPi * tf.sin(phase)))
    
    elif Bsign == -1:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
        yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)
        rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)


        return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus_DPi * tf.cos(phase) + yMinus_DPi * tf.sin(phase)))


def prob_comb(amp=[], ampbar=[], normA=1.2, normAbar=1.2, fracDD=0.82, eff1=[], eff2=[]):
    """
    Function to calculate the amplitude squared for the combinatorial background in the B2DK and B2Dpi decays

    Args:
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        normA (float, optional): the normalisation of the amplitude squared for the decay. Defaults to 1.2.
        normAbar (float, optional): the normalisation of the amplitude squared for the decay. Defaults to 1.2.
        fracDD (float, optional): the fraction of the combinatorial background. Defaults to 0.82.
        eff1 (float, optional): the efficiency value for the B2DK decay. Defaults to [].
        eff2 (float, optional): the efficiency value for the B2Dpi decay. Defaults to [].

    Returns:
        float64: the amplitude squared
    """

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)
    normA = tf.cast(normA, tf.float64)
    normAbar = tf.cast(normAbar, tf.float64)
    frac1 = fracDD*0.5
    frac2 = 1.0 - fracDD
    prob1 = eff1*absA**2 /normA
    prob2 = eff2*absAbar**2 /normAbar 
    prob3 = tf.ones_like(absA)


    return (prob1 * frac1 + prob2 * frac1 + prob3 * frac2)
