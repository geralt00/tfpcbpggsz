#ifndef D0TOKSPIPI2018_H
#define D0TOKSPIPI2018_H
#include <vector>
#include <complex>
using namespace std;

class D0ToKSpipi2018 {
public:
    // Destructor
    ~D0ToKSpipi2018();

    // Initialization method
    void init();
    std::complex<double> AMP(vector<double> ks, vector<double> pi1, vector<double> pi2);
private:
    // Member functions declared with complex computations
    std::complex<double> D0_K_0_s_1430_p_GLASS__K0S0_pip__pim_(double const* x0, double const* x1);
    std::complex<double> D0_K_0_s_1430_p_GLASS__K0S0_pip__pim__wParams(double const* x1);
    std::complex<double> D0_K_0_s_1430_barm_GLASS__K0S0_pim__pip_(double const* x0, double const* x1);
    std::complex<double> D0_K_0_s_1430_barm_GLASS__K0S0_pim__pip__wParams(double const* x1);
    std::complex<double> D0_K_2_s_1430_p_K0S0_pip__pim_(double const* x0, double const* x1);
    std::complex<double> D0_K_2_s_1430_p_K0S0_pip__pim__wParams(double const* x1);
    std::complex<double> D0_K_2_s_1430_barm_K0S0_pim__pip_(double const* x0, double const* x1);
    std::complex<double> D0_K_2_s_1430_barm_K0S0_pim__pip__wParams(double const* x1);
    std::complex<double> D0_Ks_1410_p_K0S0_pip__pim_(double const* x0, double const* x1);
    std::complex<double> D0_Ks_1410_p_K0S0_pip__pim__wParams(double const* x1);
    std::complex<double> D0_Ks_1410_barm_K0S0_pim__pip_(double const* x0, double const* x1);
    std::complex<double> D0_Ks_1410_barm_K0S0_pim__pip__wParams(double const* x1);
    std::complex<double> D0_Ks_1680_barm_K0S0_pim__pip_(double const* x0, double const* x1);
    std::complex<double> D0_Ks_1680_barm_K0S0_pim__pip__wParams(double const* x1);
    std::complex<double> D0_Ks_892_p_K0S0_pip__pim_(double const* x0, double const* x1);
    std::complex<double> D0_Ks_892_p_K0S0_pip__pim__wParams(double const* x1);
    std::complex<double> D0_Ks_892_barm_K0S0_pim__pip_(double const* x0, double const* x1);
    std::complex<double> D0_Ks_892_barm_K0S0_pim__pip__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_0__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_0__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_1__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_1__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_2__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_2__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_3__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_pole_3__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_0__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_0__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_1__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_1__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_2__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_2__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_3__pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_PiPi00_kMatrix_prod_3__pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_f_2__1270_0_pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_f_2__1270_0_pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_omega_782_0_pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_omega_782_0_pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_rho_1450_0_pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_rho_1450_0_pip_pim__K0S0__wParams(double const* x1);
    std::complex<double> D0_rho_770_0_pip_pim__K0S0_(double const* x0, double const* x1);
    std::complex<double> D0_rho_770_0_pip_pim__K0S0__wParams(double const* x1);
};

#endif // D0TOKSPIPI2018_H
