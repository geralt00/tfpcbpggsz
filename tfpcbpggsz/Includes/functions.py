import tensorflow as tf
import numpy as np

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

def in_Dalitz_plot(skpip, skpim): 
  if ( (skpip < QMI_smin_Kspi ) or (skpip > QMI_smax_Kspi) ) :
      return False
  UP  = Dalitz_upper_limit(skpip)
  LOW = Dalitz_lower_limit(skpip)
  if (skpim > UP):
    return False
  if (skpim < LOW):
    return False
  return True


def clip_log(x, _epsilon=1e-6):
    """clip log to allowed large value"""
    x_cut = tf.where(x > _epsilon, x, tf.ones_like(x) * _epsilon)
    b_t = tf.math.log(x_cut)

    delta_x = x - _epsilon
    b_f = (
       np.log(_epsilon) + delta_x / _epsilon - (delta_x / _epsilon) ** 2 / 2.0
    )
    return tf.where(x > _epsilon, b_t, b_f)
