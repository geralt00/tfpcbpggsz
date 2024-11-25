#ifndef INCLUDES_FUNCTIONS_H 
#define INCLUDES_FUNCTIONS_H 1

#include <complex>
#include "common_constants.h"
#include "DalitzPoint.hh"


std::vector<double> masses_from_rotvar(double zp_p, double zm_pp) {
  /// Compute the s+ and s- variables from the rotated-stretched variables
  std::vector<double> res = {0,0};
  double Ap = QMI_zpmax_Kspi + QMI_zpmin_Kspi; /// warning these variable are in MeV
  double Bp = QMI_zpmax_Kspi - QMI_zpmin_Kspi; /// warning these variable are in MeV
  double Am = QMI_zmmax_Kspi + QMI_zmmin_Kspi; /// warning these variable are in MeV
  double Bm = QMI_zmmax_Kspi - QMI_zmmin_Kspi; /// warning these variable are in MeV
  double zp = (Bp*zp_p + Ap)/2.;
  double zm = (Bm*zm_pp*(zp_p+2.)/2. + Am)/2.;
  res[0] = (zp+zm)/2.;
  res[1] = (zp-zm)/2.;
  res[0] = res[0]/1000000; // put it it GeV
  res[1] = res[1]/1000000; // put it it GeV
  return res;
}


DalitzPoint DalitzP_D2Kspipi(double q0p,double q0n) {
  if (q0p > 3 && q0n > 3) {
    std::cout << "Careful, in DalitzP_D2Kspipi, one of the momentum is greater than 3GeV squared, so probably not in GeV !" << std::endl;
  }
  double mD0 = PDG_m_Dz.value()*0.001;
  double mPi = PDG_m_pi.value()*0.001;
  double mKs = PDG_m_Ks.value()*0.001;
  double qpn = mD0 * mD0 + 2*mPi*mPi + mKs*mKs - q0p -q0n;
  DalitzPoint point( mKs, mPi, mPi, q0p, q0n, qpn );
  return point;
}

DalitzPoint DalitzP_D2Kspipi_rotvar(double zp_p,double zm_pp) {
  double q0p = masses_from_rotvar(zp_p, zm_pp)[0];
  double q0n = masses_from_rotvar(zp_p, zm_pp)[1];
  if (q0p > 3 && q0n > 3) {
    std::cout << "Careful, in DalitzP_D2Kspipi, one of the momentum is greater than 3GeV squared, so probably not in GeV !" << std::endl;
  }
  double mD0 = PDG_m_Dz.value()*0.001;
  double mPi = PDG_m_pi.value()*0.001;
  double mKs = PDG_m_Ks.value()*0.001;
  double qpn = mD0 * mD0 + 2*mPi*mPi + mKs*mKs - q0p -q0n;
  DalitzPoint point( mKs, mPi, mPi, q0p, q0n, qpn );
  return point;
}


std::vector < double > func_var_rotated(double sp, double sm, double zpmax, double zpmin, double zmmax, double zmmin) {
  double zp = sp + sm;
  double zm = sp - sm;
  double nump = 2*zp - zpmax - zpmin;
  double denomp = zpmax - zpmin;
  double zp_p = nump / denomp;
  double numm = 2*zm - zmmax - zmmin;
  double denomm = zmmax - zmmin;
  double zm_p = numm / denomm;
  std::vector< double > res = {zp_p, zm_p};
  return res;
}

std::vector < double > func_var_rotated_stretched(double sp, double sm, double zpmax, double zpmin, double zmmax, double zmmin) {
  double zp = sp + sm;
  double zm = sp - sm;
  double nump = 2*zp - zpmax - zpmin;
  double denomp = zpmax - zpmin;
  double zp_p = nump / denomp;
  double numm = 2*zm - zmmax - zmmin;
  double denomm = zmmax - zmmin;
  double zm_p = numm / denomm;
  double zm_pp = 2*zm_p/(2+zp_p);
  std::vector< double > res = {zp_p, zm_pp};
  return res;
}

double Dalitz_upper_limit(double* skpip, double* par) { // in GeV !!
  return (0.394465 + 1.8821*skpip[0] - 0.5* pow(skpip[0],2) + 1.86484* sqrt(0.044744 - 0.485412*skpip[0] + 1.13203*pow(skpip[0],2) - 0.541203*pow(skpip[0],3) + 0.0718881*pow(skpip[0],4)))/skpip[0];
}

double Dalitz_lower_limit(double* skpip, double* par) { // in GeV !!
  return (0.394465 + 1.8821*skpip[0] - 0.5* pow(skpip[0],2) - 1.86484* sqrt(0.044744 - 0.485412*skpip[0] + 1.13203*pow(skpip[0],2) - 0.541203*pow(skpip[0],3) + 0.0718881*pow(skpip[0],4)))/skpip[0];
}

bool in_Dalitz_plot(double skpip, double skpim) {
  if ( (skpip < QMI_smin_Kspi.value()/( (double) 1000000)) || (skpip > QMI_smax_Kspi.value()/( (double) 1000000)) ) { return false; }
  double zero = 0;
  double UP = Dalitz_upper_limit(&skpip, &zero);
  double LOW = Dalitz_lower_limit(&skpip, &zero);
  if ( (skpim > UP) ) {
    // std::cout << "DEBUG ------ " << std::endl;
    // std::cout << "DEBUG ------ Rejected because skpim > UP " << std::endl;
    // std::cout << "DEBUG ------ skpip = " << skpip << std::endl;
    // std::cout << "DEBUG ------ skpim = " << skpim << std::endl;
    // std::cout << "DEBUG ------    UP = " << UP << std::endl;
    // std::cout << "DEBUG ------   LOW = " << LOW << std::endl;
    return false;
  }
  if ( (skpim < LOW) ) {
    // std::cout << "DEBUG ------ " << std::endl;
    // std::cout << "DEBUG ------ Rejected because skpim < LOW " << std::endl;
    // std::cout << "DEBUG ------ skpip = " << skpip << std::endl;
    // std::cout << "DEBUG ------ skpim = " << skpim << std::endl;
    // std::cout << "DEBUG ------    UP = " << UP << std::endl;
    // std::cout << "DEBUG ------   LOW = " << LOW << std::endl;
    return false;
  }  
  return true;
}


/* double Dalitz_upper_limitFUNC(double skpip) { // in GeV !! */
/*   return (0.394465 + 1.8821*skpip - 0.5* pow(skpip,2) + 1.86484* TMath::Sqrt(0.044744 - 0.485412*skpip + 1.13203*pow(skpip,2) - 0.541203*pow(skpip,3) + 0.0718881*pow(skpip,4)))/skpip; */
/* } */

/* double Dalitz_lower_limitFUNC(double skpip) { // in GeV !! */
/*   return (0.394465 + 1.8821*skpip - 0.5* pow(skpip,2) + 1.86484* TMath::Sqrt(0.044744 - 0.485412*skpip + 1.13203*pow(skpip,2) - 0.541203*pow(skpip,3) + 0.0718881*pow(skpip,4)))/skpip; */
/* } */


#endif 
