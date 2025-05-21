/// ----------------------------------------------------------------------
// Common top level includes and declarations
// ----------------------------------------------------------------------

#ifndef DEFINITIONS_MODEL_BELLE2018_H
#define DEFINITIONS_MODEL_BELLE2018_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>

#include "Resonance.hh"

using namespace std;

// enum for the parameters in the Dalitz-plot fit and time-dependent measurement

  enum parameter_types {
  	SigNorm = 0,

  	// isobars amplitude and phase notation
  	nonResAmp = 1,
  	nonResPh = 2,

  	omegaAmp = 3,
  	omegaPh = 4,

  	Kstar892minus_Amp = 5,
  	Kstar892minus_Ph = 6,

  	Kstarzero1430minus_Amp = 7,
  	Kstarzero1430minus_Ph = 8,

  	Kstartwo1430minus_Amp = 9,
  	Kstartwo1430minus_Ph = 10,

  	Kstar1680minus_Amp = 11,
  	Kstar1680minus_Ph = 12,

  	Kstar1410minus_Amp = 13,
  	Kstar1410minus_Ph = 14,

  	Kstar892plus_Amp = 15,
  	Kstar892plus_Ph = 16,

  	Kstarzero1430plus_Amp = 17,
  	Kstarzero1430plus_Ph = 18,

  	Kstartwo1430plus_Amp = 19,
  	Kstartwo1430plus_Ph = 20,

  	Kstar1680plus_Amp = 21,
  	Kstar1680plus_Ph = 22,

  	Kstar1410plus_Amp = 23,
  	Kstar1410plus_Ph = 24,

  	fzero980_Amp = 25,
  	fzero980_Ph = 26,

  	ftwo1270_Amp = 27,
  	ftwo1270_Ph = 28,

  	fzero1370_Amp = 29,
  	fzero1370_Ph = 30,

  	rho1450_Amp = 31,
  	rho1450_Ph = 32,

  	sigma1_Amp = 33,
  	sigma1_Ph = 34,

  	sigma2_Amp = 35,
  	sigma2_Ph = 36,

  	// isobars real and imaginary part version
  	nonRes_realpart = 37,
  	nonRes_imaginarypart = 38,

  	omega_realpart = 39,
  	omega_imaginarypart = 40,

  	Kstar892minus_realpart = 41,
  	Kstar892minus_imaginarypart = 42,

  	Kstarzero1430minus_realpart = 43,
  	Kstarzero1430minus_imaginarypart = 44,

  	Kstartwo1430minus_realpart = 45,
  	Kstartwo1430minus_imaginarypart = 46,

  	Kstar1680minus_realpart = 47,
  	Kstar1680minus_imaginarypart = 48,

  	Kstar1410minus_realpart = 49,
  	Kstar1410minus_imaginarypart = 50,

  	Kstar892plus_realpart = 51,
  	Kstar892plus_imaginarypart = 52,

  	Kstarzero1430plus_realpart = 53,
  	Kstarzero1430plus_imaginarypart = 54,

  	Kstartwo1430plus_realpart = 55,
  	Kstartwo1430plus_imaginarypart = 56,

  	Kstar1680plus_realpart = 57,
  	Kstar1680plus_imaginarypart = 58,

  	Kstar1410plus_realpart = 59,
  	Kstar1410plus_imaginarypart = 60,

  	fzero980_realpart = 61,
  	fzero980_imaginarypart = 62,

  	ftwo1270_realpart = 63,
  	ftwo1270_imaginarypart = 64,

  	fzero1370_realpart = 65,
  	fzero1370_imaginarypart = 66,

  	rho1450_realpart = 67,
  	rho1450_imaginarypart = 68,

  	sigma1_realpart = 69,
  	sigma1_imaginarypart = 70,

  	sigma2_realpart = 71,
  	sigma2_imaginarypart = 72,

  	// masses and widths
  	sigma1_Mass = 73,
  	sigma1_Width = 74,

  	sigma2_Mass = 75,
  	sigma2_Width = 76,

  	Kstar892_Mass = 77,
  	Kstar892_Width = 78,

  	rho770_Mass = 79,
  	rho770_Width = 80,

  	omega_Mass = 81,
  	omega_Width = 82,

  	Kstarzero1430_Mass = 83,
  	Kstarzero1430_Width = 84,

  	Kstartwo1430_Mass = 85,
  	Kstartwo1430_Width = 86,

  	Kstar1680_Mass = 87,
  	Kstar1680_Width = 88,

  	Kstar1410_Mass = 89,
  	Kstar1410_Width = 90,

  	fzero980_Mass = 91,
  	fzero980_Width = 92,

  	ftwo1270_Mass = 93,
  	ftwo1270_Width = 94,

  	fzero1370_Mass = 95,
  	fzero1370_Width = 96,

  	rho1450_Mass = 97,
  	rho1450_Width = 98,

  	// Kmatrix amplitude and phase notation
  	Kmatrix_beta1_Amplitude = 99,
  	Kmatrix_beta1_Phase = 100,
  	Kmatrix_beta2_Amplitude = 101,
  	Kmatrix_beta2_Phase = 102,
  	Kmatrix_beta3_Amplitude = 103,
  	Kmatrix_beta3_Phase = 104,
  	Kmatrix_beta4_Amplitude = 105,
  	Kmatrix_beta4_Phase = 106,
  	Kmatrix_beta5_Amplitude = 107,
  	Kmatrix_beta5_Phase = 108,

  	Kmatrix_f_prod_11_Amplitude = 109,
  	Kmatrix_f_prod_11_Phase = 110,
  	Kmatrix_f_prod_12_Amplitude = 111,
  	Kmatrix_f_prod_12_Phase = 112,
  	Kmatrix_f_prod_13_Amplitude = 113,
  	Kmatrix_f_prod_13_Phase = 114,
  	Kmatrix_f_prod_14_Amplitude = 115,
  	Kmatrix_f_prod_14_Phase = 116,
  	Kmatrix_f_prod_15_Amplitude = 117,
  	Kmatrix_f_prod_15_Phase = 118,

  	// Kmatrix real and imaginary part version
  	Kmatrix_beta1_realpart = 119,
  	Kmatrix_beta2_realpart = 120,
  	Kmatrix_beta3_realpart = 121,
  	Kmatrix_beta4_realpart = 122,
  	Kmatrix_beta5_realpart = 123,

  	Kmatrix_beta1_imaginarypart = 124,
  	Kmatrix_beta2_imaginarypart = 125,
  	Kmatrix_beta3_imaginarypart = 126,
  	Kmatrix_beta4_imaginarypart = 127,
  	Kmatrix_beta5_imaginarypart = 128,

  	Kmatrix_f_prod_11_realpart = 129,
  	Kmatrix_f_prod_12_realpart = 130,
  	Kmatrix_f_prod_13_realpart = 131,
  	Kmatrix_f_prod_14_realpart = 132,
  	Kmatrix_f_prod_15_realpart = 133,

  	Kmatrix_f_prod_11_imaginarypart = 134,
  	Kmatrix_f_prod_12_imaginarypart = 135,
  	Kmatrix_f_prod_13_imaginarypart = 136,
  	Kmatrix_f_prod_14_imaginarypart = 137,
  	Kmatrix_f_prod_15_imaginarypart = 138,

  	Kmatrix_s_prod_0 = 139,

  	// LASS parameters
  	LASS_F = 140,
  	LASS_phi_F = 141,
  	LASS_R = 142,
  	LASS_phi_R = 143,
  	LASS_a = 144,
  	LASS_r = 145,

  	// fit fractions
  	FITFRAC_nonresonant = 146,
  	FITFRAC_rho = 147,
  	FITFRAC_omega = 148,
  	FITFRAC_Kstar = 149,
  	FITFRAC_Kstar0_1430 = 150,
  	FITFRAC_Kstar2_1430 = 151,
  	FITFRAC_Kstar_1680 = 152,
  	FITFRAC_Kstar_1410 = 153,
  	FITFRAC_Kstar_DCS = 154,
  	FITFRAC_Kstar0_1430_DCS = 155,
  	FITFRAC_Kstar2_1430_DCS = 156,
  	FITFRAC_Kstar_1680_DCS = 157,
  	FITFRAC_Kstar_1410_DCS = 158,
  	FITFRAC_fzero980 = 159,
  	FITFRAC_f2_1270 = 160,
  	FITFRAC_fzero1370 = 161,
  	FITFRAC_rho_1450 = 162,
  	FITFRAC_sigma1 = 163,
  	FITFRAC_sigma2 = 164,
  	FITFRAC_Kmatrix = 165,

  	mistag = 166,

  	fsig = 167,
  	fbkg_comb_rand = 168,

  	RESONANCE_NUMBER = 169,

    // CP asymmetry
    SIG_S = 170,
    SIG_C = 171,

    // lifetime and mixing fit
    SIG_tau_B0 = 172,
    SIG_delta_m_B0 = 173,
    SIG_tau_Bplus = 174,

    // prompt background
    BKG_PROMPT_FRAC = 175,
    BKG_PROMPT_MU = 176,

    // lifetime background
    BKG_LIFETIME_MU = 177,
    BKG_LIFETIME_TAU = 178,

    // background model 1
    BKGMODEL1_RES_S_MAIN = 179,
    BKGMODEL1_RES_S_TAIL = 180,
    BKGMODEL1_RES_FRAC_TAIL = 181,

	cat4_BKG_PROMPT_FRAC = 182,
	cat4_BKG_PROMPT_MU = 183,
	cat4_BKG_LIFETIME_TAU = 184,
	cat4_BKG_LIFETIME_MU = 185,

	cat5_BKG_PROMPT_FRAC = 186,
	cat5_BKG_PROMPT_MU = 187,
	cat5_BKG_LIFETIME_TAU = 188,
	cat5_BKG_LIFETIME_MU = 189,

    DUMMY = 190,

    signal_fraction = 191,

	SIG_PHI = 192,

	DeltaM = 193,

	tauB0 = 194,

	tauB0_cat2 = 195,

	tauBplus = 196,

	coefS = 197,
	coefC = 198,

	beta = 199,

	sin2beta = 200,
	cos2beta = 201,

	NUMBER_PARAMETERS = 202
  };

#endif // DEFINITIONS_MODEL_BELLE2018_H
