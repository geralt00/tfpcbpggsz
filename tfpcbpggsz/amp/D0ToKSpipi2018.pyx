# D0ToKSpipi2018.pyx
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.complex cimport complex as c_complex
from libcpp.string cimport string
cdef extern from  "D0ToKSpipi2018.cxx":
    pass

cdef extern from "D0ToKSpipi2018.h":
    cdef cppclass D0ToKSpipi2018:
        D0ToKSpipi2018()  # Constructor
        void init()

        vector[c_complex[double]] all_amplitudes(double[:] x1)        
        c_complex[double] AMP(vector[double] k0l, vector[double] pip, vector[double] pim);

        c_complex[double] D0_K_0_s_1430_p_GLASS__K0S0_pip__pim_(double[:] x0, double[:] x1);
        c_complex[double] D0_K_0_s_1430_p_GLASS__K0S0_pip__pim__wParams(double[:] x1);
        c_complex[double] D0_K_0_s_1430_barm_GLASS__K0S0_pim__pip_(double[:] x0, double[:] x1);
        c_complex[double] D0_K_0_s_1430_barm_GLASS__K0S0_pim__pip__wParams(double[:] x1);
        c_complex[double] D0_K_2_s_1430_p_K0S0_pip__pim_(double[:] x0, double[:] x1);
        c_complex[double] D0_K_2_s_1430_p_K0S0_pip__pim__wParams(double[:] x1);
        c_complex[double] D0_K_2_s_1430_barm_K0S0_pim__pip_(double[:] x0, double[:] x1);
        c_complex[double] D0_K_2_s_1430_barm_K0S0_pim__pip__wParams(double[:] x1);
        c_complex[double] D0_Ks_1410_p_K0S0_pip__pim_(double[:] x0, double[:] x1);
        c_complex[double] D0_Ks_1410_p_K0S0_pip__pim__wParams(double[:] x1);
        c_complex[double] D0_Ks_1410_barm_K0S0_pim__pip_(double[:] x0, double[:] x1);
        c_complex[double] D0_Ks_1410_barm_K0S0_pim__pip__wParams(double[:] x1);
        c_complex[double] D0_Ks_1680_barm_K0S0_pim__pip_(double[:] x0, double[:] x1);
        c_complex[double] D0_Ks_1680_barm_K0S0_pim__pip__wParams(double[:] x1);
        c_complex[double] D0_Ks_892_p_K0S0_pip__pim_(double[:] x0, double[:] x1);
        c_complex[double] D0_Ks_892_p_K0S0_pip__pim__wParams(double[:] x1);
        c_complex[double] D0_Ks_892_barm_K0S0_pim__pip_(double[:] x0, double[:] x1);
        c_complex[double] D0_Ks_892_barm_K0S0_pim__pip__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_0__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_0__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_1__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_1__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_2__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_2__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_3__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_pole_3__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_0__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_0__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_1__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_1__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_2__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_2__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_3__pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_PiPi00_kMatrix_prod_3__pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_f_2__1270_0_pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_f_2__1270_0_pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_omega_782_0_pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_omega_782_0_pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_rho_1450_0_pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_rho_1450_0_pip_pim__K0S0__wParams(double[:] x1);
        c_complex[double] D0_rho_770_0_pip_pim__K0S0_(double[:] x0, double[:] x1);
        c_complex[double] D0_rho_770_0_pip_pim__K0S0__wParams(double[:] x1);

cdef class PyD0ToKSpipi2018:
    """
    This is a Python wrapper for the C++ class D0ToKSpipi2018.
    """

    cdef D0ToKSpipi2018* thisptr  # Pointer to the C++ class instance

    def __cinit__(self):
        self.thisptr = new D0ToKSpipi2018()

    def __dealloc__(self):
        del self.thisptr

    def init(self):
        """
        Initialize the D0ToKSpipi2018 class.
        """
        self.thisptr.init()

    def AMP(self, list k0, list pip, list pim):
        """
        Calculate the amplitude for the D0 -> K0S0 pi+ pi- decay.

        Parameters:
        k0 (list): List of K0S0 particle momenta.
        pip (list): List of pi+ particle momenta.
        pim (list): List of pi- particle momenta.

        Returns:
        complex: The calculated amplitude.
        """
        cdef vector[double] ck0 = k0
        cdef vector[double] cpip = pip
        cdef vector[double] cpim = pim

        return self.thisptr.AMP(ck0, cpip, cpim)
