#ifndef D0TOKSPIPI2018_H
#define D0TOKSPIPI2018_H

#include "DalitzPoint.hh"
#include "DalitzPoint.cpp"
#include <cmath>
#include <complex>
#include <regex.h>
#include "common_constants.h"
#include "definitions_model_belle2018.h"

class Resonance;

class D0ToKspipi2018 {
public:
  // D0ToKspipi2018(double xp, double xm, double yp, double ym, int bFlavor) :
  //   m_xp{xp}, m_xm{xm}, m_yp{yp}, m_ym{ym}, m_bFlavor{bFlavor} {
  ~D0ToKspipi2018();

  void init(const char* data_file_path);  
  std::vector<std::complex<double>> get_amp(double _zm, double _zp) const;
  //std::complex<double> get_amp(double _x, double _y, int D_flag);
  std::vector<std::vector<std::complex<double>>> AMP(vector<double> _zm, vector<double> _zp) const;
  // arrays for the Dalitz model results without systematics
  std::complex<double> Amp(const DalitzPoint& point, const double* par) const;
  //TComplex Amp(double *x, double *par);
    

  // Masses of the relevant particles.
  double mD0;
  double mKs;
  double mPi;

  int _parameter_number[NUMBER_PARAMETERS];
  double _parameter_value[NUMBER_PARAMETERS];

private:

  // double* _parameter_number[NUMBER_PARAMETERS];
  // double* _parameter_value[NUMBER_PARAMETERS];
  //static double _parameter_error[NUMBER_PARAMETERS];

  // double m_xp;
  // double m_xm;
  // double m_yp;
  // double m_ym;
  // int  m_bFlavor;
  
};
#endif
