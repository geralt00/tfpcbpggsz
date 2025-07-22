# D0ToKspipi2018.pyx
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.complex cimport complex as c_complex # Standard alias for std::complex
from libcpp.string cimport string # Not used in the provided snippet, but kept if needed elsewhere
import importlib.resources

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
        vector[vector[c_complex[double]]] AMP(vector[double] zm, vector[double] zp) except + nogil  # AMP method with exception handling and nogil


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

    def init(self):
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

    def AMP(self, vector[double] zm, vector[double] zp):
        """
        Calculates the amplitude by calling the C++ AMP method,
        which processes vectors of zm and zp values.
        """

        cdef vector[double] czm = zm
        cdef vector[double] czp = zp

        return self.thisptr.AMP(czm, czp)