import numpy as np
from sympy import Ci, denom
from tfpcbpggsz.amp.evtgen.D0ToKspipi2018 import PyD0ToKspipi2018
from tfpcbpggsz.amp.ampgen.D0ToKSpipi2018 import PyD0ToKSpipi2018
from tfpcbpggsz.amp.bes.D0ToKspipi2025 import PyD0ToKspipi2025
from tfpcbpggsz.amp.bes_test import D0ToKSpipi
from tfpcbpggsz.ulti import p4_to_phsp, p4_to_srd
from tfpcbpggsz.tensorflow_wrapper import tf


class Amplitude:
    def __init__(self, model='evtgen', **kwargs):
        """
        Initializes the Amplitude class.

        Args:
            model (str): The model to use for amplitude calculation ('evtgen' or 'ampgen').
            **kwargs: Additional keyword arguments for the model.
        """
        self.model_name = model
        self.model_instance_alt = None
        if model == 'evtgen':
            self.model_instance = PyD0ToKspipi2018()
        elif model == 'ampgen':
            self.model_instance = PyD0ToKSpipi2018()
        elif model == 'bes':
            self.model_instance = PyD0ToKspipi2025()
        elif model == 'bes_test':
            self.model_instance = D0ToKSpipi

        else:
            raise ValueError("Model must be either 'evtgen', 'ampgen', or 'bes'.")
        self.kwargs = kwargs
        self.res = False
        self.pc = None
        self._data = None
        self._amp_file = None
        
        if 'amp_file' in kwargs:
            if model == 'evtgen':
                self.model_instance_alt = PyD0ToKspipi2018()
            elif model == 'bes_test':
                self.model_instance_alt = D0ToKSpipi
            else:
                self.model_instance_alt = PyD0ToKspipi2025()
            
            self._amp_file = kwargs['amp_file']
            print(f"Using amplitude file: {self._amp_file}")

    def init(self):
        """        Initializes the amplitude model instance.
        """
        if hasattr(self.model_instance, 'init'):
            if self._amp_file is None:
                self.model_instance.init()
            else:
                self.model_instance.init()
                self.model_instance_alt.init(self._amp_file)
        else:
            print("Model instance does not have an init method.")

    def set_res_params(self, res_params):
        """
        Sets the resonance parameters for the amplitude calculation.

        Args:
            res_params (list): List of resonance parameters.
        """

        self.res = True
        self.res_params = res_params

    def replace_magnitude(self, original_A: complex, source_A: complex) -> complex:
        """
        Creates a new complex number with the magnitude of source_A and the phase of original_A.

        Args:
            original_A (complex): The complex number providing the phase.
            source_A (complex): The complex number providing the magnitude.

        Returns:
            complex: The newly constructed complex number.
        """
        # 1. Get the new magnitude from the source vector
        new_magnitude = np.abs(source_A)

        # 2. Get the original phase from the original vector
        original_phase = np.angle(original_A)

        # 3. Reconstruct the new complex number in polar form
        new_A = new_magnitude * np.exp(1j * original_phase)

        return new_A


    def _ensure_amplitudes_computed(self, data_input):
        """
        Ensures that Kspipi.AMP is called only if necessary and caches its raw result.
        This is the core of the optimization.
        """
        
        phsp_points = []
        raw_A_output, raw_Abar_output = None, None
        self._data = data_input
        phsp_points = p4_to_phsp(data_input)
        if self.res is True:
            phsp_points += self.res_params
        if self.model_name == 'bes':
            #swap pi+ and pi- for bes model
            phsp_points = [phsp_points[1], phsp_points[0]]

        raw_A_total = self.model_instance.AMP(phsp_points[0].tolist(), phsp_points[1].tolist())
        raw_A_total = tf.cast(raw_A_total, tf.complex128)
        raw_A_output, raw_Abar_output = raw_A_total[:, 0], raw_A_total[:, 1]
        if self.model_instance_alt is not None:
            raw_A_total_alt = self.model_instance_alt.AMP(phsp_points[0].tolist(), phsp_points[1].tolist())
            raw_A_total_alt = tf.cast(raw_A_total_alt, tf.complex128)
            raw_A_output_alt, raw_Abar_output_alt = raw_A_total_alt[:, 0], raw_A_total_alt[:, 1]
            raw_A_output = self.replace_magnitude(raw_A_output, raw_A_output_alt)
            raw_Abar_output = self.replace_magnitude(raw_Abar_output, raw_Abar_output_alt)



        return raw_A_output, raw_Abar_output
    
    def _get_amplitudes_computed(self, data_input):
        """
        Ensures that Kspipi.AMP is called only if necessary and caches its raw result.
        This is the core of the optimization.
        """
        
        p1, p2, p3 = data_input
        p1_flatten, p2_flatten, p3_flatten = [], [], []
        raw_A_output, raw_Abar_output = None, None
        self._data = data_input

        if not isinstance(p1, tf.Tensor):
            p1_flatten = p1.tolist()
            p2_flatten = p2.tolist()
            p3_flatten = p3.tolist()
        else:
            p1_flatten = p1.numpy().tolist()
            p2_flatten = p2.numpy().tolist()
            p3_flatten = p3.numpy().tolist()
        raw_A_total = self.model_instance.AMP(p1_flatten, p2_flatten, p3_flatten)
        raw_A_total = tf.cast(raw_A_total, tf.complex128)
        raw_A_output, raw_Abar_output = raw_A_total[:, 0], raw_A_total[:, 1]
        if self.model_instance_alt is not None:
            raw_A_total_alt = self.model_instance_alt.AMP(p1_flatten, p2_flatten, p3_flatten)
            raw_A_total_alt = tf.cast(raw_A_total_alt, tf.complex128)
            raw_A_output_alt, raw_Abar_output_alt = raw_A_total_alt[:, 0], raw_A_total_alt[:, 1]
            raw_A_output = self.replace_magnitude(raw_A_output, raw_A_output_alt)
            raw_Abar_output = self.replace_magnitude(raw_Abar_output, raw_Abar_output_alt)



        return raw_A_output, raw_Abar_output

    def amp(self, data):
        """
        Calculates the amplitude of the decay from momenta.
        """

        Kspipi = self.model_instance
        if self.model_name == 'evtgen' or self.model_name == 'bes':
            raw_tensor = self._ensure_amplitudes_computed(data)
            if raw_tensor is None:
                # Handle cases where _ensure_amplitudes_computed returns None (e.g., _amp_from_mass is False)
                # Return a default complex tensor, an empty tensor, or raise an error.
                return tf.constant([], dtype=tf.complex128) 

            # The original `amp` function returned the entire result from Kspipi.AMP, cast to complex128.
            # The bug in the original `amp_i = tf.cast(amp_i, ...)` is fixed by casting `raw_tensor`.
            amp_result = tf.cast(raw_tensor[0], tf.complex128)
            return amp_result 
        elif self.model_name == 'bes_test':
            raw_tensor = self._get_amplitudes_computed(data)
            if raw_tensor is None:
                # Handle cases where _ensure_amplitudes_computed returns None (e.g., _amp_from_mass is False)
                # Return a default complex tensor, an empty tensor, or raise an error.
                return tf.constant([], dtype=tf.complex128) 

            # The original `amp` function returned the entire result from Kspipi.AMP, cast to complex128.
            # The bug in the original `amp_i = tf.cast(amp_i, ...)` is fixed by casting `raw_tensor`.
            amp_result = tf.cast(raw_tensor[0], tf.complex128)
            return amp_result
        else:
            p1,p2,p3 = data
            if not isinstance(p1, tf.Tensor):
                amp_i = Kspipi.AMP(p1.tolist(), p2.tolist(), p3.tolist())     
            else:
                amp_i = Kspipi.AMP(p1.numpy().tolist(), p2.numpy().tolist(), p3.numpy().tolist())    
            amp_i = tf.cast(amp_i, tf.complex128)
            return amp_i
    
    def ampbar(self, data):
        #"""Calculate the amplitude of the decay from momenta."""
        Kspipi = self.model_instance
        if self.model_name == 'evtgen' or self.model_name == 'bes':
            raw_tensor = self._ensure_amplitudes_computed(data)

            if raw_tensor is None:
                return tf.constant([], dtype=tf.complex128)

            # `ampbar` takes the second "column" of the results.
            # This implies Kspipi.AMP returns a 2D structure (e.g., list of [amp, amp_bar] pairs).
            # Slicing `[:, 1]` extracts all rows from the second column.
            #ampbar_slice = raw_tensor[:, 1]
        
            # Cast the slice to tf.complex128
            ampbar_result = tf.cast(raw_tensor[1], tf.complex128)
            return ampbar_result 
        elif self.model_name == 'bes_test':
            raw_tensor = self._get_amplitudes_computed(data)
            if raw_tensor is None:
                return tf.constant([], dtype=tf.complex128)

            # `ampbar` takes the second "column" of the results.
            # This implies Kspipi.AMP returns a 2D structure (e.g., list of [amp, amp_bar] pairs).
            # Slicing `[:, 1]` extracts all rows from the second column.
            ampbar_result = tf.cast(raw_tensor[1], tf.complex128)
            return ampbar_result
        else:
            p1,p2,p3 = data
            p1bar, p2bar, p3bar = tf.concat([p1[:, :1], tf.negative(p1[:, 1:])], axis=1), tf.concat([p2[:, :1], tf.negative(p2[:, 1:])], axis=1), tf.concat([p3[:, :1], tf.negative(p3[:, 1:])], axis=1)
            ampbar_i = Kspipi.AMP(p1bar.numpy().tolist(), p3bar.numpy().tolist(), p2bar.numpy().tolist())
            ampbar_i = tf.cast(tf.negative(ampbar_i), tf.complex128)
            return ampbar_i
        
    def DeltadeltaD(self, amp, ampbar):
        """
        Calculates the difference in phase between the amplitude and its conjugate.
        This function is used to compute the Strong phase difference in the D amplitude.
        Args:
            amp (tf.Tensor): The amplitude tensor.
            ampbar (tf.Tensor): The conjugate amplitude tensor.
        Returns:
            tf.Tensor: The phase difference tensor.
        """

        from tfpcbpggsz.core import DeltadeltaD, DeltadeltaD_old
        DeltadeltaD_val = None
        if self.model_name == 'evtgen' or self.model_name == 'bes' or self.model_name == 'bes_test':
            # Use the DeltadeltaD function from tfpcbpggsz.core
            DeltadeltaD_val = DeltadeltaD(amp, ampbar)
        else:
            # Use the DeltadeltaD_old function from tfpcbpggsz.core
            DeltadeltaD_val = DeltadeltaD_old(amp, ampbar)

        if self.pc is not None:
            # Apply phase correction if available
            print("Applying phase correction") 
            DeltadeltaD_val += self.pc.eval_corr_norm(p4_to_srd(self._data))

        return DeltadeltaD_val
            

from scipy.spatial import cKDTree
class GetCiSi:
    def __init__(self,Amplitude,**kwargs):
        self.Amplitude = Amplitude
        self.kwargs = kwargs
        self.binning_file = kwargs.get('binning_file',None)
        if self.binning_file:
            bin_data = self.read_binning()
            self.s01_list = bin_data[:, 0].astype(float)
            self.s02_list = bin_data[:, 1].astype(float)
            self.binList = bin_data[:, 2].astype(float)
        self.tree = cKDTree(np.column_stack((self.s01_list, self.s02_list)))


    def read_binning(self):
        """
        Reads data from a txt file using the map function.

        Args:
          filename: The name of the txt file.

        Returns:
          A NumPy array containing the data.
        """

        with open(self.binning_file, 'r') as f:
          data = list(map(lambda line: np.array(line.strip().split(' ')), f))
        return np.array(data)


    
    def find_bin_numpy(self, event):
        """
        Finds the bin index on the Dalitz plane using NumPy.

        Args:
            event: The event object with s(i, j) method for calculating invariant mass.
            s01_list: NumPy array of s01 values for the binning scheme.
            s02_list: NumPy array of s02 values for the binning scheme.
            binList: List of bin indices corresponding to the binning scheme.

        Returns:
            The bin index.
        """

        s01, s02 = event['s12'], event['s13']
        _, idx = self.tree.query(np.column_stack((s01, s02)))
        bins = self.binList[idx]
        return np.where(s01 < s02, bins, -bins)
    
    def get_cisi(self, event):
        """Compute Ci and Si efficiently per bin."""
        bin_list = self.find_bin_numpy(event)
        phase    = event['model_phase']
        absAmp   = np.abs(event['amp'])
        absAmpbar = np.abs(event['ampbar'])

    # Precompute numerators and denominators
        num_cos = absAmp * absAmpbar * np.cos(phase)
        num_sin = absAmp * absAmpbar * np.sin(phase)
        den_A   = absAmp**2
        den_B   = absAmpbar**2

    # Group by bin index
        unique_bins, inv = np.unique(bin_list, return_inverse=True)
        sum_num_cos = np.bincount(inv, weights=num_cos)
        sum_num_sin = np.bincount(inv, weights=num_sin)
        sum_den_A   = np.bincount(inv, weights=den_A)
        sum_den_B   = np.bincount(inv, weights=den_B)

    # Avoid divide-by-zero
        denom = np.sqrt(sum_den_A * sum_den_B)
        denom[denom == 0] = np.nan

        Ci_vals = sum_num_cos / denom
        Si_vals = sum_num_sin / denom

        # Return as dicts for consistency
        Ci = {str(b): c for b, c in zip(unique_bins, Ci_vals)}
        Si = {str(b): s for b, s in zip(unique_bins, Si_vals)}

        return Ci, Si, unique_bins





if __name__ == "__main__":
    # Example usage
    model = 'bes_test'  
    amplitude = Amplitude(model=model)
    amplitude.init()
    k0 = [0.66786802, -0.37050274, -0.07949114,  0.23417914]
    pim = [0.65777907, 0.14400027, -0.23772302, -0.57960771]
    pip = [0.5391929, 0.22650247, 0.31721417, 0.34542857]

    data = [np.array([k0]), np.array([pim]), np.array([pip])]
    amp = amplitude.amp(data)
    ampbar = amplitude.ampbar(data)
    print(np.abs(amp), np.angle(amp))
    print(np.abs(ampbar), np.angle(ampbar))

