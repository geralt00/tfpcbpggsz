# D0ToKspipi2018.pyx
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.complex cimport complex as c_complex # Standard alias for std::complex
from libcpp.string cimport string # Not used in the provided snippet, but kept if needed elsewhere
import importlib.resources

import numpy as np
cimport numpy as np

# This line is often not strictly necessary if all declarations are in the .h file,
# but it doesn't harm.
cdef extern from "D0ToKspipi2018.cxx":
    pass

cdef extern from "D0ToKspipi2018.h":
    cdef cppclass D0ToKspipi2018:
        D0ToKspipi2018() except + nogil # Constructor with exception handling and nogil
        void init(const char* data_file_path) except + nogil
        # Assuming get_amp might also throw or could be long-running
        vector[c_complex[double]] get_amp(double zm, double zp) except + nogil  # get_amp method with exception handling and nogil
        vector[c_complex[double]] get_amp_vec(const double* zm, const double* zp) except + nogil
        vector[vector[c_complex[double]]] AMP(vector[double] zm, vector[double] zp) except + nogil  # AMP method with exception handling and nogil
        vector[c_complex[double]] AMP_bulk(const double* zm,
                                           const double* zp,
                                           size_t N) except + nogil  # New bulk AMP method


cdef class PyD0ToKspipi2018:
    cdef D0ToKspipi2018* thisptr  # Pointer to the C++ class instance

    def __cinit__(self):
        # Allocate the C++ object
        self.thisptr = new D0ToKspipi2018()
        if self.thisptr == NULL: # Check for allocation failure
            raise MemoryError("Failed to allocate D0ToKspipi2018 instance")

    def __dealloc__(self):
        # Important: Free the C++ object when the Python object is garbage collected
        if self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL # Good practice to nullify after delete

    def init(self, file_path=None):
        """Wraps the C++ init method."""
        # Consider GIL release if init() is potentially long and thread-safe
        # with nogil:

        file_ref = (importlib.resources.files('tfpcbpggsz')
                    .joinpath('external')
                    .joinpath('BELLE2018_data.txt'))
        with importlib.resources.as_file(file_ref) as data_path:
            # data_path is a pathlib.Path object. It is only valid inside this block.
            # Convert the path to bytes for the const char* argument.
            path_bytes = str(data_path).encode('utf-8')
            if file_path is not None:
                path_bytes = file_path.encode('utf-8')

            # 3. Call the C++ init method with the guaranteed path.
            #    The GIL is released automatically due to the 'nogil' declaration.
            self.thisptr.init(path_bytes)

    def get_amp(self, double zm, double zp):
        """
        Wraps the C++ get_amp method.
        This method is kept if you need to call the C++ get_amp directly from Python
        for individual values, separately from the main AMP method.
        """
        cdef double czm = zm
        cdef double czp = zp

        return self.thisptr.get_amp(czm, czp)

    def AMP(self,
            np.ndarray[np.double_t, ndim=1, mode="c"] zm,
            np.ndarray[np.double_t, ndim=1, mode="c"] zp):

        cdef Py_ssize_t N = zm.shape[0]
        cdef const double* ptr_zm  = <const double*> zm.data
        cdef const double* ptr_zp = <const double*> zp.data

        cdef vector[c_complex[double]] result = self.thisptr.AMP_bulk(ptr_zm, ptr_zp, N)
        # Convert vector to numpy array
        cdef np.ndarray[np.complex128_t, ndim=2] arr = np.empty((N, 2), dtype=np.complex128)
        cdef np.npy_intp shape[2]
        shape[0] = N
        shape[1] = 2
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_COMPLEX128, &result[0])
        arr = np.array(arr, copy=True)  # make Python own a copy
        return arr

'''
    def AMP(self, vector[double] zm, vector[double] zp):
        """
        Calculates the amplitude by calling the C++ AMP method,
        which processes vectors of zm and zp values.
        """

        cdef vector[double] czm = zm
        cdef vector[double] czp = zp

        return self.thisptr.AMP(czm, czp)

'''