#ifndef D0TOKSPIPI2018_CXX
#define D0TOKSPIPI2018_CXX

#include <iostream>
#include <string>
#include <utility>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <complex>

#include "common_constants.h"
#include "D0ToKspipi2018.h"
#include "Resonance.hh"
#include "Resonance.cpp"
#include "DalitzPoint.hh"
#include "DalitzPoint.cpp"
#include "definitions_model_belle2018.h"
#include "functions.h"


#define PI 3.14159265

// using namespace std;
// const double mD0 = 1.8648399;
// const double mKs = 0.49761401;
// const double mPi = 0.13957017;

D0ToKspipi2018::~D0ToKspipi2018(){}


void D0ToKspipi2018::init(){
  // masses
  mD0 = PDG_m_Dz.value()*0.001;
  mPi = PDG_m_pi.value()*0.001;
  mKs = PDG_m_Ks.value()*0.001;
  /// parameters
  std::string parameter_filename, ratio_filename, outpath;
  parameter_filename = "/software/rj23972/safety_net/tfpcbpggsz/tfpcbpggsz/amp_masses/BELLE2018_data.txt";
  // open parameter file and read Dalitz model
  // cout << "Opening parameter file: " << parameter_filename << endl;
  FILE *pFile_fit_result;
  pFile_fit_result = fopen(parameter_filename.c_str(), "r");
  if (pFile_fit_result==NULL) {
    cout << "File error in reading parameter array." << endl;
    cout << endl;
    cout << "Can not find:  " << parameter_filename << endl;
    cout << endl;
  }
  int temp_number = -999;
  double temp_value = -999;
  double temp_error = -999;
  while (fscanf(pFile_fit_result, "%i %lf %lf\n", &temp_number, &temp_value, &temp_error) == 3 )
    {
      _parameter_number[temp_number] = temp_number;
      _parameter_value[temp_number] = temp_value;
    }
  fclose(pFile_fit_result);

  cout << "D0ToKSpipi2018 (tenfor) (July 31 2024) ==> Initialization !" << endl;
}

std::vector<std::complex<double>> D0ToKspipi2018::get_amp(double _zp, double _zm) const
{

  // init();
  // std::cout << "I'm trying something here" << std::endl;
  const double mSq0p = masses_from_rotvar(_zp,_zm)[0];
  const double mSq0n = masses_from_rotvar(_zp,_zm)[1];
  const double mSqpn = mD0 * mD0 + 2*mPi*mPi + mKs*mKs - mSq0p -mSq0n;
  std::complex<double> amp( -1.0, 0.0 );
  // Direct and conjugated Dalitz points.
  const DalitzPoint pointD0( mKs, mPi, mPi, mSq0n, mSq0p, mSqpn );
  const DalitzPoint pointD0b( mKs, mPi, mPi, mSq0p, mSq0n, mSqpn );
  // Direct and conjugated amplitudes.
  const std::complex<double> ampD0  = Amp( pointD0, _parameter_value ); 
  const std::complex<double> ampD0b = Amp( pointD0b, _parameter_value );

  const std::vector<std::complex<double>> result = {ampD0, ampD0b};
  return result;
}

  // for ( int i = 0; i<202; i++ ) {
  //   std::cout << "Param number " << i << " = " << _parameter_value[i] << std::endl;
  // }

  // const Double_t phase_correction = Phase_Correction(_zp, _zm, m_polyorder, m_phasecoeff);
  // const TComplex exp_phase(1, phase_correction, kTRUE);
  // std::cout << "_zp " << _zp << std::endl;
  // std::cout << "_zm " << _zm << std::endl;
  // for (int i = 0;i<m_phasecoeff.size();i++ ) {
  //   std::cout << "m_phasecoeff["<<i<<"] " << m_phasecoeff[i] << std::endl;
  // }
  // std::cout << "phase_correction " << phase_correction << std::endl;
  // std::cout << "exp_phase        " << exp_phase << std::endl;
  // std::cout << " " << std::endl;
  
//   if ( m_bFlavor == 1 ) {
//     // TComplex iphase( 0., m_deltaB + m_gamma, kFALSE);
//     // TComplex expo( TComplex::Exp(iphase) );
//     // amp = ampD0b + m_rB * expo * ampD0;
//     std::complex<double> cp_obs( m_xp, m_yp);
//     amp = ampD0b + cp_obs * ampD0; // * exp_phase;
//     // amp = ampD0b;    
//     // std::cout << " B is B+ or B0" << std::endl;
//   } else if ( m_bFlavor == -1 )  {
//     // TComplex iphase( 0., m_deltaB - m_gamma, kFALSE);
//     // TComplex expo( TComplex::Exp(iphase) );
//     // amp = ampD0 + m_rB * expo * ampD0b;
//     std::complex<double> cp_obs( m_xm, m_ym);
//     amp = ampD0 + cp_obs * ampD0b; // * exp_phase;
//     // amp = ampD0;
//     // std::cout << " B is B- or B0bar" << std::endl;
//   } else if ( m_bFlavor == -2 )  { // to test the D0
//     amp = ampD0;
//   } else if ( m_bFlavor ==  2 )  { // to test the D0b
//     amp = ampD0b;
//   }
//   else {
//     std::cout << "Wrong B flavor !" << std::endl;
//     amp = std::complex( -1.0, 0.0 );
//     return amp;
//   }

//   double prob = amp.Rho2();
//   return prob; 
// }


std::complex<double> D0ToKspipi2018::Amp(const DalitzPoint& point, const double* par) const {

  
  if (point.isValid() == false) {
    return 0;
  }

  const std::complex<double> TComplex_omega( par[omega_realpart], par[omega_imaginarypart]);
  const std::complex<double> TComplex_Kstar892minus( par[Kstar892minus_realpart], par[Kstar892minus_imaginarypart]);
  const std::complex<double> TComplex_Kstarzero1430minus( par[Kstarzero1430minus_realpart], par[Kstarzero1430minus_imaginarypart]);
  const std::complex<double> TComplex_Kstartwo1430minus( par[Kstartwo1430minus_realpart], par[Kstartwo1430minus_imaginarypart]);
  const std::complex<double> TComplex_Kstar1680minus( par[Kstar1680minus_realpart], par[Kstar1680minus_imaginarypart]);
  const std::complex<double> TComplex_Kstar1410minus( par[Kstar1410minus_realpart], par[Kstar1410minus_imaginarypart]);
  const std::complex<double> TComplex_Kstar892plus( par[Kstar892plus_realpart], par[Kstar892plus_imaginarypart]);
  const std::complex<double> TComplex_Kstarzero1430plus( par[Kstarzero1430plus_realpart], par[Kstarzero1430plus_imaginarypart]);
  const std::complex<double> TComplex_Kstartwo1430plus( par[Kstartwo1430plus_realpart], par[Kstartwo1430plus_imaginarypart]);
  const std::complex<double> TComplex_Kstar1410plus( par[Kstar1410plus_realpart], par[Kstar1410plus_imaginarypart]);
  const std::complex<double> TComplex_ftwo1270( par[ftwo1270_realpart], par[ftwo1270_imaginarypart]);
  const std::complex<double> TComplex_rho1450( par[rho1450_realpart], par[rho1450_imaginarypart]);

  const std::complex<double> TComplex_Kmatrix_beta1( par[Kmatrix_beta1_realpart], par[Kmatrix_beta1_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_beta2( par[Kmatrix_beta2_realpart], par[Kmatrix_beta2_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_beta3( par[Kmatrix_beta3_realpart], par[Kmatrix_beta3_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_beta4( par[Kmatrix_beta4_realpart], par[Kmatrix_beta4_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_beta5( par[Kmatrix_beta5_realpart], par[Kmatrix_beta5_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_f_prod_11( par[Kmatrix_f_prod_11_realpart], par[Kmatrix_f_prod_11_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_f_prod_12( par[Kmatrix_f_prod_12_realpart], par[Kmatrix_f_prod_12_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_f_prod_13( par[Kmatrix_f_prod_13_realpart], par[Kmatrix_f_prod_13_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_f_prod_14( par[Kmatrix_f_prod_14_realpart], par[Kmatrix_f_prod_14_imaginarypart]);
  const std::complex<double> TComplex_Kmatrix_f_prod_15( par[Kmatrix_f_prod_15_realpart], par[Kmatrix_f_prod_15_imaginarypart]);


  // pipi resonances (L!=0)
  const Resonance ResRho("BW", 1, 0, par[rho770_Mass], par[rho770_Width], 1, 1.5, 5.0, "Y");
  const Resonance ResOmega("BW", std::abs(TComplex_omega), std::arg(TComplex_omega) * 180. / PI, par[omega_Mass], par[omega_Width], 1, 1.5, 5.0, "Y");
  const Resonance Resf2_1270("BW", std::abs(TComplex_ftwo1270), std::arg(TComplex_ftwo1270) * 180. / PI, par[ftwo1270_Mass], par[ftwo1270_Width], 2, 1.5, 5.0, "Y");
  const Resonance ResRho_1450("BW", std::abs(TComplex_rho1450), std::arg(TComplex_rho1450) * 180. / PI, par[rho1450_Mass], par[rho1450_Width], 1, 1.5, 5.0, "Y");

  // K* resonances, Cabibbo-favored
  const Resonance ResKstar("BW", std::abs(TComplex_Kstar892minus), std::arg(TComplex_Kstar892minus) * 180. / PI, par[Kstar892_Mass], par[Kstar892_Width], 1, 1.5, 5.0, "X");

  const Resonance ResKstar0_1430("LASS", std::abs(TComplex_Kstarzero1430minus), std::arg(TComplex_Kstarzero1430minus) * 180. / PI, par[Kstarzero1430_Mass], par[Kstarzero1430_Width], 0, 1.5, 5.0, "X",
			   par[LASS_F],
			   par[LASS_phi_F],
			   par[LASS_R],
			   par[LASS_phi_R],
			   par[LASS_a],
			   par[LASS_r]);

  const Resonance ResKstar2_1430("BW", std::abs(TComplex_Kstartwo1430minus), std::arg(TComplex_Kstartwo1430minus) * 180. / PI, par[Kstartwo1430_Mass], par[Kstartwo1430_Width], 2, 1.5, 5.0, "X");
  const Resonance ResKstar_1680("BW" , std::abs(TComplex_Kstar1680minus), std::arg(TComplex_Kstar1680minus) * 180. / PI, par[Kstar1680_Mass], par[Kstar1680_Width], 1, 1.5, 5.0, "X");
  const Resonance ResKstar_1410("BW" , std::abs(TComplex_Kstar1410minus), std::arg(TComplex_Kstar1410minus) * 180. / PI, par[Kstar1410_Mass], par[Kstar1410_Width], 1, 1.5, 5.0, "X");

  // K* resonances, doubly Cabibbo-suppressed
  const Resonance ResKstar_DCS("BW", std::abs(TComplex_Kstar892plus), std::arg(TComplex_Kstar892plus) * 180. / PI, par[Kstar892_Mass], par[Kstar892_Width], 1, 1.5, 5.0, "Z");

  const Resonance ResKstar0_1430_DCS("LASS", std::abs(TComplex_Kstarzero1430plus), std::arg(TComplex_Kstarzero1430plus) * 180. / PI, par[Kstarzero1430_Mass], par[Kstarzero1430_Width], 0, 1.5, 5.0, "Z",
			       par[LASS_F],
			       par[LASS_phi_F],
			       par[LASS_R],
			       par[LASS_phi_R],
			       par[LASS_a],
			       par[LASS_r]);

  const Resonance ResKstar2_1430_DCS("BW", std::abs(TComplex_Kstartwo1430plus), std::arg(TComplex_Kstartwo1430plus) * 180. / PI, par[Kstartwo1430_Mass], par[Kstartwo1430_Width], 2, 1.5, 5.0, "Z");
  const Resonance ResKstar_1410_DCS("BW", std::abs(TComplex_Kstar1410plus), std::arg(TComplex_Kstar1410plus) * 180. / PI, par[Kstar1410_Mass], par[Kstar1410_Width], 1, 1.5, 5.0, "Z");

  // K-matrix for the pipi S-wave
  const Resonance ResKMatrix("KMatrix", "Y",
		       std::abs(TComplex_Kmatrix_beta1),
		       std::arg(TComplex_Kmatrix_beta1),
		       std::abs(TComplex_Kmatrix_beta2),
		       std::arg(TComplex_Kmatrix_beta2),
		       std::abs(TComplex_Kmatrix_beta3),
		       std::arg(TComplex_Kmatrix_beta3),
		       std::abs(TComplex_Kmatrix_beta4),
		       std::arg(TComplex_Kmatrix_beta4),
		       std::abs(TComplex_Kmatrix_beta5),
		       std::arg(TComplex_Kmatrix_beta5),
		       std::abs(TComplex_Kmatrix_f_prod_11),
		       std::arg(TComplex_Kmatrix_f_prod_11),
		       std::abs(TComplex_Kmatrix_f_prod_12),
		       std::arg(TComplex_Kmatrix_f_prod_12),
		       std::abs(TComplex_Kmatrix_f_prod_13),
		       std::arg(TComplex_Kmatrix_f_prod_13),
		       std::abs(TComplex_Kmatrix_f_prod_14),
		       std::arg(TComplex_Kmatrix_f_prod_14),
		       std::abs(TComplex_Kmatrix_f_prod_15),
		       std::arg(TComplex_Kmatrix_f_prod_15),
		       par[Kmatrix_s_prod_0]
		       );

/*
 * Compute the total amplitude by the coherent sum of the individual amplitudes.
 *
 * Resonance.contribution(xx, yy) returns the amplitude and phase of individual resonances as a function of the Dalitz plot position xx and yy.
 */
  std::complex<double> total_amp(0.0, 0.0);

  total_amp +=
    ResRho.contribution(point) +
    ResOmega.contribution(point) +
    ResKstar.contribution(point) +
    ResKstar0_1430.contribution(point) +
    ResKstar2_1430.contribution(point) +
    ResKstar_1680.contribution(point) +
    ResKstar_1410.contribution(point) +
    ResKstar_DCS.contribution(point) +
    ResKstar0_1430_DCS.contribution(point) +
    ResKstar2_1430_DCS.contribution(point) +
    ResKstar_1410_DCS.contribution(point) +
    Resf2_1270.contribution(point) +
    ResRho_1450.contribution(point) +
    ResKMatrix.contribution(point);

  /*
   * Return the total amplitude
   */
  return total_amp;
}


#endif
