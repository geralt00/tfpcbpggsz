"""Utility functions for tfpcbpggsz package.

This module provides utility functions for physics calculations, coordinate transformations,
and data processing in the context of charm and beauty physics analyses. It includes:

- Invariant mass calculations from 4-momentum vectors
- Coordinate transformations between different Dalitz plot parameterizations
- Physics parameter conversions for CP violation studies
- Data masking and filtering utilities
- Angle conversion utilities

The module supports both standard and BES (Beijing Spectrometer) format 4-momentum vectors,
and provides transformations to various coordinate systems including phase space (PHSP),
Rotated Dalitz (RD), and Stretched Rotated Dalitz (SRD) coordinates.
"""

import numpy as np


def get_mass(p1, p2):
    """Calculates the invariant mass squared of a two-particle system.

    Args:
        p1 (np.ndarray): A numpy array of shape (N, 4) representing the 4-momentum (E, px, py, pz) of the first particle.
        p2 (np.ndarray): A numpy array of shape (N, 4) representing the 4-momentum (E, px, py, pz) of the second particle.
    Returns:
        np.ndarray: A numpy array of shape (N,) representing the invariant mass squared of the two-particle system.

    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("Input arrays must have the same number of rows (events).")
    if p1.shape[1] != 4 or p2.shape[1] != 4:
        raise ValueError("Input arrays must have shape (N, 4) for 4-momentum vectors.")
    mass_squared = (p1[:, 0] + p2[:, 0])**2 - (p1[:, 1] + p2[:, 1])**2 - (p1[:, 2] + p2[:, 2])**2 - (p1[:, 3] + p2[:, 3])**2

    return mass_squared

def get_mass_bes(p1, p2):
    """Calculates the invariant mass squared of a two-particle system in BES format.

    Args:
        p1 (np.ndarray): A numpy array of shape (N, 4) representing the 4-momentum (px, py, pz, E) of the first particle.
        p2 (np.ndarray): A numpy array of shape (N, 4) representing the 4-momentum (px, py, pz, E) of the second particle.
    
    Returns:
        np.ndarray: A numpy array of shape (N,) representing the invariant mass squared of the two-particle system.
    
    Raises:
        ValueError: If input arrays have different number of rows or don't have shape (N, 4).
    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("Input arrays must have the same number of rows (events).")
    if p1.shape[1] != 4 or p2.shape[1] != 4:
        raise ValueError("Input arrays must have shape (N, 4) for 4-momentum vectors.")
    mass_squared = (p1[:, 3] + p2[:, 3])**2 - (p1[:, 0] + p2[:, 0])**2 - (p1[:, 1] + p2[:, 1])**2 - (p1[:, 2] + p2[:, 2])**2

    return mass_squared

def read_minos_errors(m):
    """
    Read the minos errors from the Minuit object

    Args:
        m (Minuit): The Minuit object containing the fit results.

    Returns:
        dict: {'parameter_name_low': lower error, 'parameter_name_high': upper error}
    """
    errors = {}
    for key in m.merrors.keys():
        errors[f'{key}_low'] = m.merrors[key].lower
        errors[f'{key}_high'] = m.merrors[key].upper
    return errors

def phsp_to_srd(x_valid, y_valid):
    """Convert the phase space coordinates (s _-, s_+) to the Stretched Rotated Dalitz (SRD) coordinates.
    
    This function transforms phase space coordinates to SRD coordinates using rotation, 
    stretching, and additional scaling transformations.
    
    Args:
        x_valid (np.ndarray): The x-coordinate values (s_-) in phase space.
        y_valid (np.ndarray): The y-coordinate values (s_+) in phase space.
    
    Returns:
        np.ndarray: Array of shape (2, N) containing [z_+, z''_-].
    """
    rotatedSymCoord = (y_valid + x_valid)/2  
    rotatedAntiSymCoord = (y_valid - x_valid)/2 

    m1_ = 2.23289
    c1_ = -3.11554092
    m2_ = 0.40229469*2
    c2_ = 0

    stretchedSymCoord = m1_ * rotatedSymCoord + c1_
    stretchedAntiSymCoord = m2_ * rotatedAntiSymCoord + c2_
    antiSym_scale = 2.0
    antiSym_offset = 2.0
    stretchedAntiSymCoord_dp = (antiSym_scale * (stretchedAntiSymCoord)) / (antiSym_offset + stretchedSymCoord)
    return np.array([stretchedSymCoord, stretchedAntiSymCoord_dp])

def phsp_to_rd(x_valid, y_valid):
    """Convert the phase space coordinates to the Rotated Dalitz (RD) coordinates.
    
    This function transforms phase space coordinates to RD coordinates using rotation 
    and stretching transformations without the additional scaling applied in SRD.
    
    Args:
        x_valid (np.ndarray): The x-coordinate values (s _-) in phase space.
        y_valid (np.ndarray): The y-coordinate values (s_+) in phase space.
    
    Returns:
        np.ndarray: Array of shape (2, N) containing [z_+, z'_-].
    """
    rotatedSymCoord = (y_valid + x_valid)/2  #z_+
    rotatedAntiSymCoord = (y_valid - x_valid)/2 #z_-

    m1_ = 2.23289
    c1_ = -3.11554092
    m2_ = 0.40229469
    c2_ = 0


    stretchedSymCoord = m1_ * rotatedSymCoord + c1_
    stretchedAntiSymCoord = m2_ * rotatedAntiSymCoord + c2_
    return np.array([stretchedSymCoord, stretchedAntiSymCoord])

def p4_to_mag(data):
    """Calculate the momentum magnitude of 4-momentum vectors for three particles.
    
    This function computes the 3-momentum magnitude for each particle from their 4-momentum vectors.
    
    Args:
        data (tuple): A tuple containing three numpy arrays (p1, p2, p3) where each array 
                     has shape (N, 4) representing 4-momentum vectors (E, px, py, pz).
    
    Returns:
        list: A list of three numpy arrays [P_1, P_2, P_3] containing the momentum 
              magnitudes for each particle.
    """
    p1, p2, p3 = data
    P_Ks = np.sqrt(p1[:, 1]**2 + p1[:,2]**2 + p1[:,3]**2)
    P_pim = np.sqrt(p2[:, 1]**2 + p2[:,2]**2 + p2[:,3]**2)
    P_pip = np.sqrt(p3[:, 1]**2 + p3[:,2]**2 + p3[:,3]**2)
    return [P_Ks, P_pim, P_pip]

def p4_to_srd(data, resolution=None):
    """Convert 4-momentum vectors to Stretched Rotated Dalitz (SRD) coordinates.
    
    This function calculates invariant masses from 4-momentum vectors and transforms 
    them to SRD coordinates, with optional resolution corrections.
    
    Args:
        data (tuple): A tuple containing three numpy arrays (p1, p2, p3) where each array 
                     has shape (N, 4) representing 4-momentum vectors.
        resolution (tuple, optional): A tuple of two resolution corrections to be added 
                                    to m12 and m13 respectively. Defaults to None.
    
    Returns:
        np.ndarray: Array of shape (2, N) containing SRD coordinates.
    """
    p1, p2, p3 = data
    m12 = get_mass(p1, p2)
    m13 = get_mass(p1, p3)
    m12 = np.array(m12, dtype=np.float64)
    m13 = np.array(m13, dtype=np.float64)
    if resolution is not None:
        m12 += resolution[0]
        m13 += resolution[1]
    srd = phsp_to_srd(m12, m13)
    srd = np.array(srd, dtype=np.float64)
    return srd

def p4_to_rd(data):
    """Convert 4-momentum vectors to Rotated Dalitz (RD) coordinates.
    
    This function calculates invariant masses from 4-momentum vectors and transforms 
    them to RD coordinates.
    
    Args:
        data (tuple): A tuple containing three numpy arrays (p1, p2, p3) where each array 
                     has shape (N, 4) representing 4-momentum vectors.
    
    Returns:
        np.ndarray: Array of shape (2, N) containing RD coordinates.
    """
    p1, p2, p3 = data
    m12 = get_mass(p1, p2)
    m13 = get_mass(p1, p3)
    rd = phsp_to_rd(m12, m13)
    return rd

def p4_to_phsp(data, resolution=None):
    """Convert 4-momentum vectors to phase space (PHSP) coordinates.
    
    This function calculates invariant masses from 4-momentum vectors with optional 
    resolution corrections and returns them as phase space coordinates.
    
    Args:
        data (tuple): A tuple containing three numpy arrays (p1, p2, p3) where each array 
                     has shape (N, 4) representing 4-momentum vectors.
        resolution (tuple, optional): A tuple of two resolution corrections to be added 
                                    to m12 and m13 respectively. Defaults to None.
    
    Returns:
        np.ndarray: Array of shape (2, N) containing phase space coordinates [m12, m13].
    """
    p1, p2, p3 = data
    m12 = get_mass(p1, p2)
    m13 = get_mass(p1, p3)
    m12 = np.array(m12, dtype=np.float64)
    m13 = np.array(m13, dtype=np.float64)
    if resolution is not None:
        m12 += resolution[0]
        m13 += resolution[1]
    coords = np.array([m12, m13])
    return coords

def deg_to_rad(deg):
    """Convert degrees to radians.
    
    Args:
        deg (float or np.ndarray): Angle(s) in degrees.
    
    Returns:
        float or np.ndarray: Angle(s) in radians.
    """
    return deg*np.pi/180

def rad_to_deg(rad):
    """Convert radians to degrees.
    
    Args:
        rad (float or np.ndarray): Angle(s) in radians.
    
    Returns:
        float or np.ndarray: Angle(s) in degrees.
    """
    return rad*180/np.pi

def get_xy_xi(physics_param):
    """Calculate Cartesian coordinates from physics parameters including xi parameters.
    
    This function takes physics parameters (gamma, rB, deltaB, r_dpi, d_dpi) and converts 
    them to Cartesian coordinates (x, y) for both B+ and B- decays, along with xi parameters.
    
    Args:
        physics_param (list): A list containing [gamma, r_dk, d_dk, r_dpi, d_dpi] where:
                             - gamma: CP-violating phase (radians)
                             - r_dk: Magnitude ratio for D→K decay
                             - d_dk: Strong phase for D→K decay (radians)
                             - r_dpi: Magnitude ratio for D→π decay
                             - d_dpi: Strong phase for D→π decay (radians)
    
    Returns:
        list: [xp, yp, xm, ym, x_xi, y_xi] where:
              - xp, yp: Cartesian coordinates for B+ decay
              - xm, ym: Cartesian coordinates for B- decay
              - x_xi, y_xi: Xi parameters for D→π/D→K ratio
    """
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
    r"""Calculate Cartesian coordinates from physics parameters.
    
    This function takes physics parameters :math:`\gamma, r_B, \delta_B` and converts them to 
    Cartesian coordinates (x, y) for both B+ and B- decays.
    
    Args:
        physics_param (list): A list containing :math:`[\gamma, r_{dk}, d_{dk}]`

    Returns:
        list: [xp, yp, xm, ym] where:
              - xp, yp: Cartesian coordinates for B+ decay
              - xm, ym: Cartesian coordinates for B- decay
    """
    gamma  = physics_param[0]
    r_dk   = physics_param[1]
    d_dk   = physics_param[2]


    xm = r_dk * np.cos(d_dk - gamma)
    xp = r_dk * np.cos(d_dk + gamma)
    ym = r_dk * np.sin(d_dk - gamma)
    yp = r_dk * np.sin(d_dk + gamma)


    return [xp, yp, xm, ym] 

def amp_mask(raw_amp, raw_ampbar, raw_amp_tag=None, raw_ampbar_tag=None, max_amp=150):
    """Mask amplitudes to be within a certain range to remove outliers.

    This function filters amplitude values by applying a maximum amplitude threshold,
    removing events where any amplitude exceeds the specified limit.

    Args:
        raw_amp (np.ndarray): The raw amplitude values for signal events.
        raw_ampbar (np.ndarray): The raw amplitude bar values for signal events.
        raw_amp_tag (np.ndarray, optional): The raw amplitude values for tag events. 
                                           Defaults to None.
        raw_ampbar_tag (np.ndarray, optional): The raw amplitude bar values for tag events. 
                                              Defaults to None.
        max_amp (float, optional): The maximum amplitude value to consider. Defaults to 100.

    Returns:
        tuple: If raw_amp_tag is None, returns (masked_amp, masked_ampbar, mask).
               If raw_amp_tag is provided, returns (masked_amp, masked_ampbar, 
               masked_amp_tag, masked_ampbar_tag, mask).
               
               - masked_amp (np.ndarray): Filtered amplitude values.
               - masked_ampbar (np.ndarray): Filtered amplitude bar values.
               - masked_amp_tag (np.ndarray): Filtered tag amplitude values (if applicable).
               - masked_ampbar_tag (np.ndarray): Filtered tag amplitude bar values (if applicable).
               - mask (np.ndarray): Boolean mask indicating which values were kept.
    """
    from tfpcbpggsz.generator.data import data_mask
    raw_amp = np.array(raw_amp)
    raw_ampbar = np.array(raw_ampbar)
    absA = np.abs(raw_amp)
    absAbar = np.abs(raw_ampbar)
    mask = (absA < max_amp) & (absAbar < max_amp)
    if raw_amp_tag is not None:
        absA_tag = np.abs(raw_amp_tag)
        absAbar_tag = np.abs(raw_ampbar_tag)
        mask = mask & (absA_tag < max_amp) & (absAbar_tag < max_amp)
    # use boolean mask to remove values
    masked_amp = data_mask(raw_amp, mask)
    masked_ampbar = data_mask(raw_ampbar, mask)
    masked_amp_tag = None
    masked_ampbar_tag = None
    if raw_amp_tag is not None:
        masked_amp_tag = data_mask(raw_amp_tag, mask)
        masked_ampbar_tag = data_mask(raw_ampbar_tag, mask)
        return masked_amp, masked_ampbar, masked_amp_tag, masked_ampbar_tag, mask
    return masked_amp, masked_ampbar, mask