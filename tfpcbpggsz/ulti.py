import numpy as np


def get_mass(p1, p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)


def phsp_to_srd(x_valid, y_valid):
    """
    Convert the phase space coordinates to the Stretched Rotated Dalitz (SRD) coordinates
    """
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
    return np.array([stretchedSymCoord, stretchedAntiSymCoord_dp]).T

