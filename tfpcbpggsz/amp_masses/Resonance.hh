#ifndef RESONANCE_H
#define RESONANCE_H

#include "DalitzPoint.hh"
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <assert.h>
#include "common_constants.h"

using namespace std;
using std::string;

#include <complex>

/*
 * Class to compute the complex amplitude of single Breit-Wigner resonances, the LASS parameterization, and the K-matrix.
 */

class Resonance {
      
public:

  // Constructor for Breit-Wigner resonances
  Resonance(std::string RName, // std::string to identify the resonances type. Needs to be BW" for the constructor of Breit-Wigner resonances.
	    double RAmp,
	    double RPhase,
	    double RMass,
            double RWidth,
	    int RSpin,
	    double RRadii,
	    double DRadii,
	    std::string axis
	    );

  // K-matrix constructor
  Resonance( std::string RName, // std::string to identify the resonances type. Needs to be "KMatrix" for the K-matrix constructor.
	     std::string axis,
	     double Kmatrix_beta1_Amplitude,
	     double Kmatrix_beta1_Phase,
	     double Kmatrix_beta2_Amplitude,
	     double Kmatrix_beta2_Phase,
	     double Kmatrix_beta3_Amplitude,
	     double Kmatrix_beta3_Phase,
	     double Kmatrix_beta4_Amplitude,
	     double Kmatrix_beta4_Phase,
	     double Kmatrix_beta5_Amplitude,
	     double Kmatrix_beta5_Phase,
	     double Kmatrix_f_prod_11_Amplitude,
	     double Kmatrix_f_prod_11_Phase,
	     double Kmatrix_f_prod_12_Amplitude,
	     double Kmatrix_f_prod_12_Phase,
	     double Kmatrix_f_prod_13_Amplitude,
	     double Kmatrix_f_prod_13_Phase,
	     double Kmatrix_f_prod_14_Amplitude,
	     double Kmatrix_f_prod_14_Phase,
	     double Kmatrix_f_prod_15_Amplitude,
	     double Kmatrix_f_prod_15_Phase,
	     double Kmatrix_s_prod_0 );

  // LASS constructor
  Resonance(std::string RName, // std::string to identify the resonances type. Needs to be "LASS" for the LASS constructor.
	    double RAmp, double RPhase, double RMass,
	    double RWidth, int RSpin, double RRadii, double DRadii, std::string axis,
	    double LASS_F,
	    double LASS_phi_F,
	    double LASS_R,
	    double LASS_phi_R,
	    double LASS_a,
	    double LASS_r
	    );

  // Access methods ----*

  // Utilities ----*
  double momentum(double m);
  double Barrier(double ratio2_p);
  double RelativeBlattWeisskopf(double p2, double pR2, int spin, double radius);
  double BlattWeisskopf(double p2, int spin, double radius);
  double FD(double mab2);
  double FR_rel(double mab2);
  double FR(double mab2);
  std::complex<double> denominator(double mab2);
  std::complex<double> BreitWigner(double mab2);
  std::complex<double> Sakurai(double mab2);
  double angular(double mab2, double mca2); 

  // Functions to compute the amplitudes
  std::complex<double> amplitude_BW(double v1, double v2, double v3) const ; // For Breit-Wigner isobars.
  std::complex<double> amplitude_KMatrix(double mab2) const ; // K-matrix.
  std::complex<double> amplitude_LASS(double mab2) const ; // LASS parameterization.

  // Function to return the amplitude as a function of the Dalitz-plot position
  std::complex<double> contribution(const DalitzPoint& point) const ;

  const int spinD0 = 0;
  double  mD0 = PDG_m_Dz.value()*0.001;
  double  mPi = PDG_m_pi.value()*0.001;
  double  mKs = PDG_m_Ks.value()*0.001;
  // const double mD0 = EvtPDL::getMass( EvtPDL::getId( "D0" ) );  
  // const double mPi = EvtPDL::getMass( EvtPDL::getId( "pi+" ) ); 
  // const double mKs = EvtPDL::getMass( EvtPDL::getId( "K_S0" ) );

private:


  int _verbose;

  // Sting to identify resonance type in the constructor.
  std::string _name;

  // Relative amplitude and phases
  double _amp;
  double _phase;

  // Resonance mass, width and spin
  double _massR;
  double _gammaR;
  int _spinR;

  double _radiusR;
  double _radiusD;

  // These are initialized by setAxis()
  double _ma, _mb, _mc;

  // X=1, Y=2, Z=3
  int _axis;
  std::string _axislabel;

  // The fixed momentum at mR (set in constructor)
  double _momentum_at_mR;
  double _xmin, _xmax;

  // K matrix parameters
  double _Kmatrix_beta1_Amplitude;
  double _Kmatrix_beta1_Phase;
  double _Kmatrix_beta2_Amplitude;
  double _Kmatrix_beta2_Phase;
  double _Kmatrix_beta3_Amplitude;
  double _Kmatrix_beta3_Phase;
  double _Kmatrix_beta4_Amplitude;
  double _Kmatrix_beta4_Phase;
  double _Kmatrix_beta5_Amplitude;
  double _Kmatrix_beta5_Phase;
  double _Kmatrix_f_prod_11_Amplitude;
  double _Kmatrix_f_prod_11_Phase;
  double _Kmatrix_f_prod_12_Amplitude;
  double _Kmatrix_f_prod_12_Phase;
  double _Kmatrix_f_prod_13_Amplitude;
  double _Kmatrix_f_prod_13_Phase;
  double _Kmatrix_f_prod_14_Amplitude;
  double _Kmatrix_f_prod_14_Phase;
  double _Kmatrix_f_prod_15_Amplitude;
  double _Kmatrix_f_prod_15_Phase;

  double _Kmatrix_s_prod_0;

  // LASS parameters
  double _LASS_F;
  double _LASS_phi_F;
  double _LASS_R;
  double _LASS_phi_R;
  double _LASS_a;
  double _LASS_r;

  // Sets particle ordering for axis
  void setAxis(std::string axisLabel);

};

#endif

