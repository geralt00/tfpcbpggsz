import numpy as np


def get_mass(p1, p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)

def get_mass_bes(p1, p2):
    return ((p1[:,3]+p2[:,3])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,0]+p2[:,0])**2)


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
    return np.array([stretchedSymCoord, stretchedAntiSymCoord_dp])

def phsp_to_rd(x_valid, y_valid):
    """
    Convert the phase space coordinates to the Rotated Dalitz (RD) coordinates
    """
    rotatedSymCoord = (y_valid + x_valid)/2  #z_+
    rotatedAntiSymCoord = (y_valid - x_valid)/2 #z_-

    m1_ = 2.23407421671132946
    c1_ = -3.1171885586526695
    m2_ = 0.8051636393861085
    c2_ = -9.54231895051727e-05

    stretchedSymCoord = m1_ * rotatedSymCoord + c1_
    stretchedAntiSymCoord = m2_ * rotatedAntiSymCoord + c2_
    return np.array([stretchedSymCoord, stretchedAntiSymCoord])

def p4_to_srd(data):
    """
    Convert the momenta to the Stretched Rotated Dalitz (SRD) coordinates
    """
    p1, p2, p3 = data
    m12 = get_mass(p1, p2)
    m13 = get_mass(p1, p3)
    srd = phsp_to_srd(m12, m13)
    return srd

def p4_to_rd(data):
    """
    Convert the momenta to the Stretched Rotated Dalitz (SRD) coordinates
    """
    p1, p2, p3 = data
    m12 = get_mass(p1, p2)
    m13 = get_mass(p1, p3)
    rd = phsp_to_rd(m12, m13)
    return rd

def p4_to_phsp(data):
    """
    Convert the momenta to the PHSP coordinates
    """
    p1, p2, p3 = data
    m12 = get_mass(p1, p2)
    m13 = get_mass(p1, p3)
    coords = np.array([m12, m13])
    return coords

def deg_to_rad(deg):
    return deg*np.pi/180

def rad_to_deg(rad):
    return rad*180/np.pi

def get_xy_xi(physics_param):
    ''' 
    takes an input vector with [gamma, rB, deltaB] and returns x, y
    angles in RADIANS
    '''
    gamma  = physics_param[0]
    r_dk   = physics_param[1]
    d_dk   = physics_param[2]
    r_dpi  = physics_param[3]
    d_dpi  = physics_param[4]

    xm = r_dk * np.cos(d_dk - gamma)
    xp = r_dk * np.cos(d_dk + gamma)
    ym = r_dk * np.sin(d_dk - gamma)
    yp = r_dk * np.sin(d_dk + gamma)

    x_xi = (r_dpi/r_dk)*np.cos(d_dpi-d_dk)
    y_xi = (r_dpi/r_dk)*np.sin(d_dpi-d_dk)

    return [xp, yp, xm, ym, x_xi, y_xi]

def get_xy(physics_param):
    ''' 
    takes an input vector with [gamma, rB, deltaB] and returns x, y
    angles in RADIANS
    '''
    gamma  = physics_param[0]
    r_dk   = physics_param[1]
    d_dk   = physics_param[2]


    xm = r_dk * np.cos(d_dk - gamma)
    xp = r_dk * np.cos(d_dk + gamma)
    ym = r_dk * np.sin(d_dk - gamma)
    yp = r_dk * np.sin(d_dk + gamma)


    return [xp, yp, xm, ym] 