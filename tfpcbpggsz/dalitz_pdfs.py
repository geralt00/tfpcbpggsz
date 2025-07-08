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

DICT_EFFICIENCY_FUNCTIONS = {
    "Flat"        : Flat,
    "Legendre_2_2": Legendre_2_2,
    "Legendre_5_5": Legendre_5_5,
}

class DalitzPDF:

    def __init__(self, name_function, component, Bsign, isSignal=False):
        self.name         = name_function
        self.component    = component
        self.functions    = self.get_functions()
        self.Bsign        = 1 if (Bsign=="Bplus") else -1
        self.isSignal     = isSignal
        return

    def get_functions(self):
        # function   = Flat
        model      = Flat
        efficiency = Flat
        if   (self.component == "DK_Kspipi"):
            model = prob_totalAmplitudeSquared_XY
            pass
        elif (self.component == "Dpi_Kspipi"):
            model = prob_totalAmplitudeSquared_DPi_XY
            pass
        elif (self.component == "Dpi_Kspipi_misID"):
            ### unshared_variables is necesary because the "normal" function
            # uses the shared_variables input.
            # For the misID DPi, we are not using the same values
            # but some mock input determined from MC, that are given as input in
            # VARDICT under "Bplus_model" and "Bminus_model".
            model = prob_totalAmplitudeSquared_XY_unshared_variables
            pass
        elif (self.component == "Dpi_Kspipi_misID_PHSP"):
            model = Flat
            pass
            
        efficiency = DICT_EFFICIENCY_FUNCTIONS[self.name]
        ### note: it's called efficiency here, but for backgrounds it is really
        # the model. It's just more practical to call it that way
        # if (self.name == "Legendre_2_2"):
        #     print("Using Legendre_2_2 for the efficiency")
        #     efficiency = Legendre_2_2
        # else:
        #     print("ERROR ---------------------- get_function ")
        #     print(" component ", self.component, " undefined in DalitzPDF")
        #     pass
        return model, efficiency

    def get_normalisation(self, norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp, variables_eff=None, variables_model=None, shared_variables=None):
        N_normalisation = len(norm_ampD0)
        if ( not (N_normalisation == len(norm_ampD0bar) ) ):
            print("ERROR ----------------------- in B2DK_Kspipi_norm")
            print("not the same number of D0 and D0bar amplitudes, something shady's going on")
            return -1
        res = 0
        ############
        # if we decide to use normalisation events generated with a non-flat model
        # we need to add this amplitude here
        gen_amplitude = 1.
        ####
        model = self.functions[0](norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp, self.Bsign, variables=variables_model, shared_variables=shared_variables)
        # print("model")
        # print(model)
        efficiency = self.functions[1](norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp, self.Bsign, variables=variables_eff, shared_variables=shared_variables)
        ampSq = model * efficiency
        # print("shared_variables:")
        # print(shared_variables)
        # print("variables:")
        # print(variables)
        # print(" ")
        # print("nan in ampSq in normalisation:")
        # print(np.isnan(ampSq).any())
        #### 
        res = tf.math.reduce_mean(ampSq / gen_amplitude)
        res *= tf.constant(
            (QMI_smax_Kspi-QMI_smin_Kspi)*(QMI_smax_Kspi-QMI_smin_Kspi),
            tf.float64
        )
        # tf.print("res in normalisation:")
        # tf.print("res: ", res)
        return res ## is a number


    # @tf.function
    def get_dalitz_pdf(self, norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp, variables_eff=None, variables_model=None, shared_variables=None):
        # print(" PRINTING some stuff !")
        # print(self.name)
        # print(self.component)
        # print(self.Bsign)
        self.norm_constant = self.get_normalisation(
            norm_ampD0            ,
            norm_ampD0bar         ,
            norm_zp_p             ,
            norm_zm_pp            ,
            variables_eff  =variables_eff  ,
            variables_model=variables_model,
            shared_variables=shared_variables
        )
        # print("norm_constant")
        # print("norm_constant")
        # print("norm_constant")
        # print("variables :", variables)
        # print("shared_variables :", shared_variables)
        # print("shared_variables :", shared_variables)
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        def pdf(ampD0, ampD0bar, zp_p, zm_pp):
            model = self.functions[0](ampD0, ampD0bar, zp_p, zm_pp, self.Bsign, variables=variables_model, shared_variables=shared_variables)
            efficiency = self.functions[1](ampD0, ampD0bar, zp_p, zm_pp, self.Bsign, variables=variables_eff, shared_variables=shared_variables)
            # tf.print("efficiency: ", efficiency)
            # tf.print("model     : ", model)
            ret_pdf = model*efficiency / self.norm_constant
            return ret_pdf
        self.pdf = pdf
        return

