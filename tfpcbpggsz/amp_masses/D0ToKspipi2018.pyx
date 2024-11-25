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

    def get_amp(self, double _zp, double _zm):
        cdef vector[c_complex[double]] result
        result = self.thisptr.get_amp(_zp, _zm)
        return result


