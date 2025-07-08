import tensorflow as tf
import numpy as np
from tfpcbpggsz.Includes.common_constants import *
from tfpcbpggsz.ulti import get_xy_xi, deg_to_rad, rad_to_deg

def INFO(string):
    print(" INFO ====== ")
    print(string)
    print(" ")
    return

def ERROR(string):
    print(" ======= ERROR ====== ")
    print(" ======= ERROR ====== ")
    print(string)
    print(" ")
    return

def invariant_mass(h1_PE, h1_PX, h1_PY, h1_PZ, h2_PE, h2_PX, h2_PY, h2_PZ, sqrt = False):
    energy = np.power( (h1_PE+h2_PE) ,2)
    mom    = -np.power((h1_PX+h2_PX),2)-np.power((h1_PY+h2_PY),2)-np.power((h1_PZ+h2_PZ),2);
    res    = energy+mom;
    if (sqrt):
            res = np.sqrt(res);
            pass
    return res


def func_var_rotated(sp, sm, zpmax, zpmin, zmmax, zmmin):
    zp = sp + sm
    zm = sp - sm
    nump = 2*zp - zpmax - zpmin
    denomp = zpmax - zpmin
    zp_p = nump / denomp
    numm = 2*zm - zmmax - zmmin
    denomm = zmmax - zmmin
    zm_p = numm / denomm
    res = [zp_p, zm_p]
    return res

def func_var_rotated_stretched(var_rotated):
  zp_p = var_rotated[0]
  zm_p = var_rotated[1]
  zm_pp = 2*zm_p/(2+zp_p)
  res = [zp_p, zm_pp]
  return res



def Dalitz_upper_limit(skpip): # in GeV !!
    res = (0.394465 + 1.8821*skpip - 0.5* np.power(skpip,2) + 1.86484* np.sqrt(0.044744 - 0.485412*skpip + 1.13203*np.power(skpip,2) - 0.541203*np.power(skpip,3) + 0.0718881*np.power(skpip,4) ) ) / skpip
    return res


def Dalitz_lower_limit(skpip): # in GeV !!
    res =  (0.394465 + 1.8821*skpip - 0.5* np.power(skpip,2) - 1.86484* np.sqrt(0.044744 - 0.485412*skpip + 1.13203*np.power(skpip,2) - 0.541203*np.power(skpip,3) + 0.0718881*np.power(skpip,4) ) ) / skpip
    return res

# def in_Dalitz_plot(skpip, skpim): 
#   if ( (skpip < QMI_smin_Kspi ) or (skpip > QMI_smax_Kspi) ) :
#       return False
#   UP  = Dalitz_upper_limit(skpip)
#   LOW = Dalitz_lower_limit(skpip)
#   if (skpim > UP):
#     return False
#   if (skpim < LOW):
#     return False
#   return True

def in_Dalitz_plot(skpip, skpim):
    UP  = Dalitz_upper_limit(skpip)
    LOW = Dalitz_lower_limit(skpip)
    res = np.where(
        (LOW<skpim) & (skpim<UP) & (QMI_smin_Kspi<skpip) & (skpip<QMI_smax_Kspi),
        True,
        False
    )
    return res

def in_Dalitz_plot_SRD(zp_p, zm_pp):
    skpip, skpim = SRD_to_masses(zp_p, zm_pp, QMI_zpmax_Kspi, QMI_zpmin_Kspi, QMI_zmmax_Kspi, QMI_zmmin_Kspi)
    return in_Dalitz_plot(skpip, skpim)

def SRD_to_masses(zp_p, zm_pp, zpmax, zpmin, zmmax, zmmin):
    z_p   = 0.5*(zp_p*(zpmax - zpmin) + zpmax+zpmin)
    zm_p    = zm_pp*(2+zp_p)/2
    z_m   = 0.5*(zm_p*(zmmax - zmmin) + zmmax+zmmin)
    Kspip   = 0.5*(z_p+z_m)
    Kspim   = 0.5*(z_p-z_m)
    return Kspip, Kspim

def clip_log(x, _epsilon=1e-6):
    """clip log to allowed large value"""
    x_cut = tf.where(x > _epsilon, x, tf.ones_like(x) * _epsilon)
    b_t = tf.math.log(x_cut)

    delta_x = x - _epsilon
    b_f = (
       np.log(_epsilon) + delta_x / _epsilon - (delta_x / _epsilon) ** 2 / 2.0
    )
    return tf.where(x > _epsilon, b_t, b_f)


# @tf.function
def Legendre_2_2(ampD0, ampD0bar, zp_p, zm_pp, Bsign, variables=None, shared_variables=None):
    Legendre_zp_1 = zp_p
    Legendre_zp_2 = (3 * np.power(zp_p, 2) - 1) / 2.
    Legendre_zm_1 = zm_pp
    Legendre_zm_2 = (3 * np.power(zm_pp, 2) - 1) / 2.
    res = (
        variables[0] +
        Legendre_zp_1*variables[1] +
        Legendre_zp_2*variables[2] +
        Legendre_zm_1*variables[3] + 
        Legendre_zm_2*variables[4] + 
        Legendre_zp_1*Legendre_zm_1*variables[5] + 
        Legendre_zp_2*Legendre_zm_1*variables[6] + 
        Legendre_zp_1*Legendre_zm_2*variables[7] + 
        Legendre_zp_2*Legendre_zm_2*variables[8]
    )
    #### phase space
    res = tf.where(
        in_Dalitz_plot_SRD(zp_p, zm_pp) == True,
        res,
        0
    )
    ### doing this to avoid negative values
    res = tf.math.maximum(res, 0)
    # print("          ")
    # print(" IN LEGENDRE_2_2 res:")
    # print(" IN LEGENDRE_2_2 res:")
    # print(" IN LEGENDRE_2_2 res:")
    # print(res)
    # print(variables)
    # print(shared_variables)
    return res


# @tf.function
def Legendre_5_5(ampD0, ampD0bar, zp_p, zm_pp, Bsign, variables=None, shared_variables=None):
    Legendre_zp_1 = zp_p
    Legendre_zp_2 = (3 * np.power(zp_p, 2) - 1) / 2.
    Legendre_zp_3 = (5 * np.power(zp_p, 3) - 3 * zp_p) / 2.
    Legendre_zp_4 = (35 * np.power(zp_p, 4) - 30 * np.power(zp_p, 2) + 3) / 8.
    Legendre_zp_5 = (63 * np.power(zp_p, 5) - 70 * np.power(zp_p, 3) + 15 * zp_p) / 8.
    Legendre_zm_1 = zm_pp
    Legendre_zm_2 = (3 * np.power(zm_pp, 2) - 1) / 2.
    Legendre_zm_3 = (5 * np.power(zm_pp, 3) - 3 * zm_pp) / 2.
    Legendre_zm_4 = (35 * np.power(zm_pp, 4) - 30 * np.power(zm_pp, 2) + 3) / 8.
    Legendre_zm_5 = (63 * np.power(zm_pp, 5) - 70 * np.power(zm_pp, 3) + 15 * zm_pp) / 8.
    res = (variables[0] +
           Legendre_zp_1*variables[1] +
           Legendre_zp_2*variables[2] +
           Legendre_zm_1*variables[3] + 
           Legendre_zm_2*variables[4] + 
           Legendre_zp_1*Legendre_zm_1*variables[5] + 
           Legendre_zp_2*Legendre_zm_1*variables[6] + 
           Legendre_zp_1*Legendre_zm_2*variables[7] + 
           Legendre_zp_2*Legendre_zm_2*variables[8] +
           Legendre_zp_3*variables[9]  +
           Legendre_zp_4*variables[10] +
           Legendre_zp_5*variables[11] +
           Legendre_zm_3*variables[12] +
           Legendre_zm_4*variables[13] +
           Legendre_zm_5*variables[14] +
           Legendre_zp_2*Legendre_zm_4*variables[15] + 
           Legendre_zp_3*Legendre_zm_2*variables[16] + 
           Legendre_zp_3*Legendre_zm_4*variables[17] + 
           Legendre_zp_4*Legendre_zm_2*variables[18] + 
           Legendre_zp_4*Legendre_zm_4*variables[19] + 
           Legendre_zp_5*Legendre_zm_2*variables[20] + 
           Legendre_zp_5*Legendre_zm_4*variables[21])   
    #### phase space
    res = tf.where(
        in_Dalitz_plot_SRD(zp_p, zm_pp) == True,
        res,
        0
    )
    ### doing this to avoid negative values
    res = tf.math.maximum(res, 0)
    # print("          ")
    # print(" IN LEGENDRE_2_2 res:")
    # print(" IN LEGENDRE_2_2 res:")
    # print(" IN LEGENDRE_2_2 res:")
    # print(res)
    # print(variables)
    # print(shared_variables)
    return res


def Flat(ampD0, ampD0bar, zp_p, zm_pp, Bsign, variables=None, shared_variables=None):
    res = np.ones(zp_p.shape)
    res = np.where(
        in_Dalitz_plot_SRD(zp_p, zm_pp) == True,
        res,
        0
    )
    return tf.cast(res, tf.float64)



def chi2_xy_to_physics_param(xplus=None, xminus=None, yplus=None, yminus=None, xxi=None, yxi=None,
                             pd_cov=None):
    # assert(type(xplus) == list) # needs to contain value and error
    xy_params = np.array([xplus, yplus, xminus, yminus, xxi, yxi])
    par_names = ["xplus", "yplus", "xminus", "yminus", "xxi", "yxi"]
    covariance = np.zeros(len(par_names)*len(par_names)).reshape(len(par_names),len(par_names))
    for i in range(len(par_names)):
        param_i = par_names[i]
        for j in range(len(par_names)):
            param_j = par_names[j]
            covariance[i][j] = pd_cov[param_i][param_j]
            pass
        pass
    # print(covariance)
    inv_covariance = np.linalg.inv(covariance)
    def chi2(x):
        gamma, rb, dB, rb_dpi, dB_dpi = x
        pred_params = np.array(get_xy_xi(
            (gamma, rb, dB, rb_dpi, dB_dpi)
        ))
        # pred_params : m_xplus, m_yplus, m_xminus, m_yminus, m_xxi, m_yxi 
        res = 0
        diff = (pred_params - xy_params)
        res = np.matmul(inv_covariance, diff)
        res = np.matmul(diff, res)
        # for pred_par, meas_par in zip(pred_params, xy_params):
        #     res += np.power(/meas_par[1],2)
        return res
    return chi2
