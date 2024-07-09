#ifndef D0TO2KSPIPI2018_H
#define D0TO2KSPIPI2018_H

#include <vector>
#include <complex>
using namespace std;

class D0ToKSpipi2018{

public:
  
  D0ToKSpipi2018() {}
  virtual ~D0ToKSpipi2018();

  void init();

  complex<double> Amp_PFT(vector<double> k0l, vector<double> pip, vector<double> pim);

protected:


private:

  complex<double> K_matrix(vector<double> p_pip, vector<double> p_pim);
  complex<double> amplitude_LASS(vector<double> p_k0l, vector<double> p_pip, vector<double> p_pim, std::string reso, double A_r, double Phi_r);
  complex<double> Resonance2(vector<double> p4_p, vector<double> p4_d1, const vector<double> p4_d2, double mag, double theta, double gamma, double bwm, int spin);

  int _nd;

  float ar[13], phir[13];
  //vector < complex<double> > CP_mult, beta, fprod;
  complex<double> CP_mult[5], beta[5], fprod[5];
  double tan2thetaC;
  double pi180inv;
  double mass_R[13], width_R[13];
  int spin_R[13];
  double frac1[3], frac2[3], frac3[3];
  double rd[4], deltad[4], Rf[4];
  double ma[5], g[5][5];  // Kmatrix_couplings
};

#endif

