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

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def get_mass(p1, p2):
    """Calculates the invariant mass squared of a two-particle system.

    Args:
        p1 (np.ndarray): A numpy array of shape (N, 4) representing the 4-momentum (E, px, py, pz) of the first particle.
        p2 (np.ndarray): A numpy array of shape (N, 4) representing the 4-momentum (E, px, py, pz) of the second particle.
    Returns:
        np.ndarray: A numpy array of shape (N,) representing the invariant mass squared of the two-particle system.

    """
    # 1. Convert inputs to NumPy arrays for consistent and robust handling
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # 2. Perform robust validation using array properties, which works even for N=0
    if p1.ndim != 2 or p2.ndim != 2 or p1.shape[1] != 4 or p2.shape[1] != 4:
        raise ValueError(f"Inputs must be 2D arrays with shape (N, 4). Got {p1.shape} and {p2.shape}.")
    
    if p1.shape[0] != p2.shape[0]:
        raise ValueError(f"Input arrays must have the same number of events. Got {p1.shape[0]} and {p2.shape[0]}.")

    # 3. If there are no events, return an empty array of the correct shape
    #if p1.shape[0] == 0:
    #    print(p1.shape)
        #return np.array([])

    # 4. Calculate the sum of the 4-momenta in a vectorized way
    total_p = p1 + p2
    
    # E^2 - (px^2 + py^2 + pz^2)
    # The formula is M^2 = E_total^2 - p_total^2
    energy_sq = total_p[:, 0]**2
    momentum_sq = np.sum(total_p[:, 1:4]**2, axis=1) # Sums px^2 + py^2 + pz^2
    
    return energy_sq - momentum_sq

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

def calculate_covariance(data , bias=False):
    """
    Calculate the covariance matrix of the given data.
    
    Parameters:
    data (numpy.ndarray): The input data for which to calculate the covariance.
    
    Returns:
    numpy.ndarray: The covariance matrix of the input data.
    """
    covariance = np.cov(data, rowvar=False, bias=bias)
    correlation = np.corrcoef(data, rowvar=False, bias=bias)
    if bias:
        sigma = np.sqrt(np.diag(covariance))
        mu = np.mean(data, axis=0)
        corrected_sigma = np.sqrt(np.array(mu)**2+np.array(sigma)**2)
        corrected_covariance = corrected_sigma[:, None] * corrected_sigma[None, :] * correlation
        return corrected_covariance, correlation

    return covariance, correlation

def calculate_correlation_matrix_from_covariance_matrix(covariance):
    """
    Calculate the correlation matrix from the covariance matrix.
    
    Parameters:
    covariance (numpy.ndarray): The covariance matrix.

    Returns:
    numpy.ndarray: The correlation matrix.
    """
    stddev = np.sqrt(np.diag(covariance))
    correlation = covariance / stddev[:, None] / stddev[None, :]
    correlation[~np.isfinite(correlation)] = 0  # Handle division by zero
    return correlation
    

def print_correlation_matrix(correlation, coeff={}):
    corr_df = pd.DataFrame(correlation, columns=coeff.keys(), index=coeff.keys())
    corr_df = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    corr_df = corr_df.round(2)  # Round to 2 decimal places
    corr_df = corr_df.fillna('')  # Fill NaN with empty string for LaTeX compatibility
    corr_df.index.name = 'Parameter'
    corr_df.columns.name = 'Parameter'  
    latex_table = corr_df.to_latex(float_format="%.2f", index=True, header=True, escape=False, sparsify=True)
    latex_table = corr_df.to_latex(float_format="%.2f", index=True, header=True, escape=False)
    return latex_table


# --- Assume Gaussian function is defined like this ---
# Make sure this matches the parameters expected by curve_fit's p0
def gaussian(x, mu, sigma, amplitude):
    """Gaussian function where amplitude is the peak height."""
    # Ensure sigma is positive if curve_fit doesn't enforce bounds
    sigma = abs(sigma)
    if sigma == 0:
        # Avoid division by zero if sigma somehow becomes zero during fitting
        # Return a very small value or handle appropriately based on context
        # Returning zeros might be safe if x is not exactly mu
        return np.zeros_like(x)
    return amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2))
# ----------------------------------------------------
#Define a asymmetric gaussian function
def gaussian_asym(x, mu, sigma, amplitude, alpha):
    """Asymmetric Gaussian function."""
    # Ensure sigma is positive if curve_fit doesn't enforce bounds
    sigma = abs(sigma)
    if sigma == 0:
        return np.zeros_like(x)
    # Asymmetric Gaussian formula
    return amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2)) * (1 + alpha * (x - mu))


def fit_gaussian(data, bins, ax, key, range=None):
    """
    Fits a Gaussian function to histogram data, calculates chi2/ndf, and plots.

    Args:
        data (array-like): The raw data to be histogrammed and fitted.
        bins (int or sequence): The number of bins or the bin edges for the histogram.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        key (str): A label or title key for the plot.
        range (tuple, optional): The lower and upper range of the bins. Defaults to None.

    Returns:
        tuple: (popt, perr, chi2_ndf)
               popt: Optimal values for the parameters (mu, sigma, amplitude).
               perr: Standard deviation errors on the parameters.
               chi2_ndf: Chi-squared per degree of freedom.
               Returns (None, None, None) if fit fails.
    """
    counts, bin_edges = np.histogram(data, bins=bins, range=range)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- Prepare for weighted fit ---
    # Use sqrt(counts) as errors. Handle bins with 0 counts.
    errors = np.sqrt(counts)
    # Assign a finite error (e.g., 1) to zero-count bins.
    # These bins will have very little weight in the fit.
    errors[errors == 0] = 1.0

    # Filter out bins with zero counts *before* fitting?
    # Alternative: Keep them but use the error=1. Let's try keeping them.
    # valid_indices = counts > 0
    # bin_centres_fit = bin_centres[valid_indices]
    # counts_fit = counts[valid_indices]
    # errors_fit = errors[valid_indices]
    # If filtering, use these _fit arrays below. If not, use original arrays.
    # Let's stick to the original approach of fitting all bins, using error=1 for zero counts.

    # Initial parameter guesses
    # Ensure data is not empty / std is not zero before calculating p0
    if len(data) == 0 or np.std(data) == 0:
        print(f"Warning: Not enough data or zero standard deviation for key: {key}. Skipping fit.")
        ax.hist(data, bins=bins, histtype='step', color='black', range=range)
        ax.set_title(f'${key}$ (No Fit - Invalid Data)')
        return None, None, None

    p0 = [np.mean(data), np.std(data), np.max(counts)] # mu, sigma, amplitude

    try:
        # Perform the weighted fit
        # absolute_sigma=True means 'sigma' represents actual std deviations.
        popt, pcov = curve_fit(gaussian, bin_centres, counts, p0=p0, sigma=errors, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov)) # Standard deviation errors on parameters

        # --- Calculate Chi-squared ---
        expected_counts = gaussian(bin_centres, *popt)

        # Calculate chi-squared statistic
        # Use only bins with non-zero counts for chi2 calculation,
        # as chi2 is ill-defined for expected=0, and sqrt(0) error was problematic.
        # However, since we used errors=1 for 0-count bins in the fit,
        # it might be more consistent to include them in chi2 calculation too,
        # using the same errors used for fitting. Let's do that.
        # If a bin had count=0, error=1. If count>0, error=sqrt(count).
        chi2 = np.sum(((counts - expected_counts) / errors)**2)

        # Calculate degrees of freedom
        n_params = len(popt)
        # NDF = (Number of data points) - (Number of parameters)
        # The number of data points is the number of bins.
        ndf = len(bin_centres) - n_params

        # Calculate chi2/ndf, handle ndf <= 0
        if ndf > 0:
            chi2_ndf = chi2 / ndf
        else:
            chi2_ndf = np.inf # Or float('nan'), indicate undefined/unreliable

        # --- Plotting ---
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 5000) # More points for smoother curve
        fit_label = (f'$\mu$={popt[0]:.2f} ± {perr[0]:.2f}\n'
                     f'$\sigma$={popt[1]:.2f} ± {perr[1]:.2f}\n'
                     f'Amp={popt[2]:.1f} ± {perr[2]:.1f}') # Added Amplitude to label
        ax.plot(x_fit, gaussian(x_fit, *popt), color='red')

        # Plot histogram data (use error bars for better visualization)
        # ax.hist(data, bins=bins, histtype='step', color='black', range=range) # Original
        ax.errorbar(bin_centres, counts, yerr=errors, fmt='k.', markersize=4, capsize=2, label='Data') # Plot data points with errors

        # Add text box with parameters and chi2/ndf
        # Adjust text position (e.g., 0.05, 0.95) and alignment for clarity
        text_content = (f'Mean: {popt[0]:.2f} ± {perr[0]:.2f}\n'
                        f'Width: {popt[1]:.2f} ± {perr[1]:.2f}\n'
                        f'Amp: {popt[2]:.1f} ± {perr[2]:.1f}\n' # Added Amplitude
                        f'$\chi^2$/NDF: {chi2_ndf:.2f} ({chi2:.1f} / {ndf})') # Added chi2 info

        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=14, # Smaller font?
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f'${key}$')
        ax.set_xlabel("Value") # Add generic labels if needed
        ax.set_ylabel("Counts")
        ax.legend() # Display legend including fit label and data label

        return popt, perr, chi2_ndf

    except RuntimeError:
        print(f"Error: Fit failed to converge for key: {key}.")
        # Plot just the histogram if fit fails
        ax.hist(data, bins=bins, histtype='step', color='black', range=range)
        ax.set_title(f'${key}$ (Fit Failed)')
        ax.text(0.05, 0.95, 'Fit Failed', transform=ax.transAxes, color='red', verticalalignment='top')
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during fitting for key {key}: {e}")
        # Plot just the histogram if fit fails
        ax.hist(data, bins=bins, histtype='step', color='black', range=range)
        ax.set_title(f'${key}$ (Fit Error)')
        ax.text(0.05, 0.95, f'Fit Error:\n{e}', transform=ax.transAxes, color='red', verticalalignment='top', fontsize=8)
        return None, None, None

def fit_gaussian_asym(data, bins, ax, key, range=None):
    """
    Fits an asymmetric Gaussian function to histogram data, calculates chi2/ndf, and plots.

    Args:
        data (array-like): The raw data to be histogrammed and fitted.
        bins (int or sequence): The number of bins or the bin edges for the histogram.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        key (str): A label or title key for the plot.
        range (tuple, optional): The lower and upper range of the bins. Defaults to None.

    Returns:
        tuple: (popt, perr, chi2_ndf)
               popt: Optimal values for the parameters (mu, sigma, amplitude).
               perr: Standard deviation errors on the parameters.
               chi2_ndf: Chi-squared per degree of freedom.
               Returns (None, None, None) if fit fails.
    """
    counts, bin_edges = np.histogram(data, bins=bins, range=range)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- Prepare for weighted fit ---
    # Use sqrt(counts) as errors. Handle bins with 0 counts.
    errors = np.sqrt(counts)
    # Assign a finite error (e.g., 1) to zero-count bins.
    # These bins will have very little weight in the fit.
    errors[errors == 0] = 1.0

    # Initial parameter guesses
    # Ensure data is not empty / std is not zero before calculating p0
    if len(data) == 0 or np.std(data) == 0:
        print(f"Warning: Not enough data or zero standard deviation for key: {key}. Skipping fit.")
        ax.hist(data, bins=bins, histtype='step', color='black', range=range)
        ax.set_title(f'${key}$ (No Fit - Invalid Data)')
        return None, None, None

    p0 = [np.mean(data), np.std(data), np.max(counts), 0.5] # mu, sigma, amplitude

    try:
        # Perform the weighted fit
        # absolute_sigma=True means 'sigma' represents actual std deviations.
        popt, pcov = curve_fit(gaussian_asym, bin_centres
                               , counts, p0=p0, sigma=errors, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov)) # Standard deviation errors on parameters
        # --- Calculate Chi-squared ---
        expected_counts = gaussian_asym(bin_centres, *popt)
        # Calculate chi-squared statistic
        # Use only bins with non-zero counts for chi2 calculation,
        # as chi2 is ill-defined for expected=0, and sqrt(0) error was problematic.
        # However, since we used errors=1 for 0-count bins in the fit,
        # it might be more consistent to include them in chi2 calculation too,
        # using the same errors used for fitting. Let's do that.
        # If a bin had count=0, error=1. If count>0, error=sqrt(count).
        chi2 = np.sum(((counts - expected_counts) / errors)**2)
        # Calculate degrees of freedom
        n_params = len(popt)
        # NDF = (Number of data points) - (Number of parameters)
        # The number of data points is the number of bins.
        ndf = len(bin_centres) - n_params
        # Calculate chi2/ndf, handle ndf <= 0
        if ndf > 0:
            chi2_ndf = chi2 / ndf
        else:
            chi2_ndf = np.inf # Or float('nan'), indicate undefined/unreliable
        # --- Plotting ---
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 5000) # More points for smoother curve 
        fit_label = (f'$\mu$={popt[0]:.2f} ± {perr[0]:.2f}\n'
                     f'$\sigma$={popt[1]:.2f} ± {perr[1]:.2f}\n'
                     f'Amp={popt[2]:.1f} ± {perr[2]:.1f}') # Added Amplitude to label
        ax.plot(x_fit, gaussian_asym(x_fit, *popt), color='red')
        # Plot histogram data (use error bars for better visualization)
        # ax.hist(data, bins=bins, histtype='step', color='black', range=range) # Original
        ax.errorbar(bin_centres, counts, yerr=errors, fmt='k.', markersize=4, capsize=2, label='Data') # Plot data points with errors
        # Add text box with parameters and chi2/ndf
        # Adjust text position (e.g., 0.05, 0.95) and alignment for clarity 
        text_content = (f'Mean: {popt[0]:.2f} ± {perr[0]:.2f}\n'
                        f'Width: {popt[1]:.2f} ± {perr[1]:.2f}\n'
                        f'Amp: {popt[2]:.1f} ± {perr[2]:.1f}\n' # Added Amplitude
                        f'$\chi^2$/NDF: {chi2_ndf:.2f} ({chi2:.1f} / {ndf})') # Added chi2 info
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=14, # Smaller font?
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(f'${key}$')
        ax.set_xlabel("Value") # Add generic labels if needed
        ax.set_ylabel("Counts")
        ax.legend() # Display legend including fit label and data label
        return popt, perr, chi2_ndf
    except RuntimeError:
        print(f"Error: Fit failed to converge for key: {key}.")
        # Plot just the histogram if fit fails
        ax.hist(data, bins=bins, histtype='step', color='black', range=range)
        ax.set_title(f'${key}$ (Fit Failed)')
        ax.text(0.05, 0.95, 'Fit Failed', transform=ax.transAxes, color='red', verticalalignment='top')
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during fitting for key {key}: {e}")
        # Plot just the histogram if fit fails
        ax.hist(data, bins=bins, histtype='step', color='black', range=range)
        ax.set_title(f'${key}$ (Fit Error)')
        ax.text(0.05, 0.95, f'Fit Error:\n{e}', transform=ax.transAxes, color='red', verticalalignment='top', fontsize=8)
        return None, None, None

