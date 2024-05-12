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
        c_complex[double] Amp_PFT(vector[double] k0l, vector[double] pip, vector[double] pim)
        c_complex[double] K_matrix(vector[double] p_pip, vector[double] p_pim)  # Assuming private methods might need wrapping
        c_complex[double] amplitude_LASS(vector[double] p_k0l, vector[double] p_pip, vector[double] p_pim, string reso, double A_r, double Phi_r)
        c_complex[double] Resonance2(vector[double] p4_p, vector[double] p4_d1, const vector[double] p4_d2, double mag, double theta, double gamma, double bwm, int spin)

cdef class PyD0ToKSpipi2018:
    cdef D0ToKSpipi2018* thisptr  # Pointer to the C++ class instance

    def __cinit__(self):
        self.thisptr = new D0ToKSpipi2018()

    def __dealloc__(self):
        del self.thisptr

    def init(self):
        self.thisptr.init()

    def Amp_PFT(self, list k0l, list pip, list pim):
        cdef vector[double] ck0l = k0l
        cdef vector[double] cpip = pip
        cdef vector[double] cpim = pim
        return self.thisptr.Amp_PFT(ck0l, cpip, cpim)
