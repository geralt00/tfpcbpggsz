# distutils: language = c++

from libcpp.complex cimport complex as cpp_complex
from libc.stdint cimport int32_t

from libcpp.vector cimport vector
from libcpp.complex cimport complex as c_complex
from libcpp.string cimport string
cdef extern from  "D0ToKSpipi2018.cxx":
    pass


# Cython wrapper for the C++ class
cdef class PyD0ToKSpipi2018:
    cdef D0ToKSpipi2018* c_d0_to_kspipi  # Pointer to the C++ class instance

    def __cinit__(self):
        self.c_d0_to_kspipi = new D0ToKSpipi2018()
        self.c_d0_to_kspipi.init()

    def __dealloc__(self):
        del self.c_d0_to_kspipi

    def AMP(self, double[:] x0, int x1):
        cdef double* px0 = &x0[0]  # Get a pointer to the data
        cdef cpp_complex[double] result = self.c_d0_to_kspipi.AMP(px0, x1)
        return result.real, result.imag


# Example of using this class from Python code
def example_use(double[:] x0, int x1):
    cdef PyD0ToKSpipi2018 d0_to_kspipi = PyD0ToKSpipi2018()
    return d0_to_kspipi.AMP(x0, x1)
