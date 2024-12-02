from tfpcbpggsz.Includes.selections import *
from tfpcbpggsz.Includes.ntuples import *
from tfpcbpggsz.Includes.variables import *
from tfpcbpggsz.Includes.common_constants import *
from tfpcbpggsz.Includes.functions import *
from tfpcbpggsz.Includes.VARDICT import VARDICT
from tfpcbpggsz.Includes.common_classes import *

from tfpcbpggsz.core import *

# def norm_pdf_2d(x, y, pdf):
#     # Flatten the tensors for sorting purposes
#     x_flat  = tf.reshape(x, [-1])
#     px_flat = tf.reshape(pdf(x), [-1])
#     y_flat  = tf.reshape(y, [-1])
#     py_flat = tf.reshape(pdf(y), [-1])
#     # Get the sorted indices and sort both x and y
#     sorted_indices = tf.argsort(x_flat)
#     sorted_x = tf.gather(x_flat, sorted_indices)
#     sorted_y = tf.gather(y_flat, sorted_indices)
#     # Perform integration using tfp.math.trapz along the second axis
#     norm_const = tfp.math.trapz(sorted_y, sorted_x)
#     a = tf.convert_to_tensor(pdf(x)/norm_const)
#     return a


def Flat(zp_p, zm_pp, Bsign, variables=None, shared_variables=None):
    res = np.ones(zp_p.shape)
    res = np.where(
        in_Dalitz_plot_SRD(zp_p, zm_pp) == True,
        res,
        0
    )
    return tf.cast(res, tf.float64)


class DalitzPDF:

    def __init__(self, name_function, component, Bsign, isSignal=False):
        self.name         = name_function
        self.component    = component
        self.function     = self.get_function()
        self.Bsign        = 1 if (Bsign=="Bplus") else -1
        self.isSignal     = isSignal
        return

    def get_function(self):
        function = Flat
        if   (self.component == "DK_Kspipi" or self.component == "DK_Kspipi_misID"):
            function = prob_totalAmplitudeSquared_XY
        elif (self.component == "Dpi_Kspipi_misID" or self.component == "Dpi_Kspipi"):
            function = prob_totalAmplitudeSquared_DPi_XY
        elif (self.name == "Legendre_2_2"):
            function = Legendre_2_2
        else:
            print("ERROR ---------------------- get_function ")
            print(" component ", self.component, " undefined in DalitzPDF")
        return function

    def get_normalisation(self, norm_ampD0, norm_ampD0bar, variables=None, shared_variables=None):
        N_normalisation = len(norm_ampD0)
        if ( not (N_normalisation == len(norm_ampD0bar) ) ):
            print("ERROR ----------------------- in B2DK_Kspipi_norm")
            print("not the same number of D0 and D0bar amplitudes, something shady's going on")
            return -1
        res = 0
        ############
        # if we decide to use normalisation events generated with a non-flat model
        # we need to add this amplitude here
        gen_amplitude = 1
        ####
        ampSq = self.function(norm_ampD0, norm_ampD0bar, self.Bsign, variables=variables, shared_variables=shared_variables)
        # print("shared_variables:")
        # print(shared_variables)
        print("variables:")
        print(variables)
        # print(" ")
        # print("nan in ampSq in normalisation:")
        # print(np.isnan(ampSq).any())
        #### 
        res = tf.math.reduce_mean(ampSq / gen_amplitude)
        res *= tf.constant(
            (QMI_smax_Kspi-QMI_smin_Kspi)*(QMI_smax_Kspi-QMI_smin_Kspi),
            tf.float64
        )
        # print("res in normalisation:")
        # print(res)
        return res ## is a number


    # @tf.function
    def get_dalitz_pdf(self, norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp, variables=None, shared_variables=None):
        # print(" PRINTING some stuff !")
        # print(self.name)
        # print(self.component)
        # print(self.Bsign)
        if (self.isSignal == True):
            self.norm_constant = self.get_normalisation(
                norm_ampD0         ,
                norm_ampD0bar      ,
                variables=variables,
                shared_variables=shared_variables
            )
        else:
            self.norm_constant = self.get_normalisation(
                norm_zp_p          ,
                norm_zm_pp         ,
                variables=variables,
                shared_variables=shared_variables
            )
            pass
        # print("norm_constant")
        # print("norm_constant")
        # print("norm_constant")
        # print("norm_constant :", self.norm_constant)
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        pdf = lambda ampD0, ampD0bar: self.function(ampD0, ampD0bar, self.Bsign, variables=variables, shared_variables=shared_variables) / self.norm_constant #
        self.pdf = pdf
        return

