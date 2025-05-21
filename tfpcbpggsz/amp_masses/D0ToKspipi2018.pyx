# D0ToKspipi2018.pyx
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.complex cimport complex as c_complex
from libcpp.string cimport string
cdef extern from  "D0ToKspipi2018.cxx":
    pass

cdef extern from "D0ToKspipi2018.h":
    cdef cppclass D0ToKspipi2018:
        D0ToKspipi2018() # Constructor
        void init()
        vector[c_complex[double]] get_amp(double _x, double _y) # the one we need


cdef class PyD0ToKspipi2018:
    cdef D0ToKspipi2018* thisptr  # Pointer to the C++ class instance

    def __cinit__(self):
        self.thisptr = new D0ToKspipi2018()

    def init(self):
        self.thisptr.init()

    def get_amp(self, double _zm, double _zp):
        cdef vector[c_complex[double]] result
        result = self.thisptr.get_amp(_zm, _zp)
        return result

    def AMP(self, list[double] _zm, list[double] _zp):
        """
        Calculate the amplitude for the D0 -> K0S0 pi+ pi- decay., old version, not tensorized
        """
        cdef vector[double] czp
        cdef vector[double] czm
        cdef vector[vector[c_complex[double]]] result

        for i in range(len(_zp)):
            result.push_back(self.get_amp(_zm[i], _zp[i]))

        return result


