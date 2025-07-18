import numpy as np
from tfpcbpggsz.amp.evtgen.D0ToKspipi2018 import PyD0ToKspipi2018
from tfpcbpggsz.amp.ampgen.D0ToKSpipi2018 import PyD0ToKSpipi2018
from tfpcbpggsz.ulti import p4_to_phsp
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
        if model == 'evtgen':
            self.model_instance = PyD0ToKspipi2018()
        elif model == 'ampgen':
            self.model_instance = PyD0ToKSpipi2018()
        else:
            raise ValueError("Model must be either 'evtgen' or 'ampgen'.")
        self.kwargs = kwargs
        self.res = False


    def init(self):
        """        Initializes the amplitude model instance.
        """
        if hasattr(self.model_instance, 'init'):
            self.model_instance.init()
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


    def _ensure_amplitudes_computed(self, data_input):
        """
        Ensures that Kspipi.AMP is called only if necessary and caches its raw result.
        This is the core of the optimization.
        """
        
        phsp_points = []
        raw_A_output, raw_Abar_output = None, None
        phsp_points = p4_to_phsp(data_input)
        if self.res is True:
            phsp_points += self.res_params

        raw_A_total = self.model_instance.AMP(phsp_points[0].tolist(), phsp_points[1].tolist())
        raw_A_total = tf.cast(raw_A_total, tf.complex128)
        raw_A_output, raw_Abar_output = raw_A_total[:, 0], raw_A_total[:, 1]


        return raw_A_output, raw_Abar_output

    def amp(self, data):
        """
        Calculates the amplitude of the decay from momenta.
        """

        Kspipi = self.model_instance
        if self.model_name == 'evtgen':
            raw_tensor = self._ensure_amplitudes_computed(data)
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
        if self.model_name == 'evtgen':
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
        if self.model_name == 'evtgen':
            # Use the DeltadeltaD function from tfpcbpggsz.core
            return DeltadeltaD(amp, ampbar)
        else:
            # Use the DeltadeltaD_old function from tfpcbpggsz.core
            return DeltadeltaD_old(amp, ampbar)
            


class GetCiSi:
    def __init__(self,D0ToKSpipi2018,**kwargs):
        self.D0ToKSpipi2018 = D0ToKSpipi2018
        self.kwargs = kwargs
        self.binning_file = kwargs.get('binning_file',None)



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

        s01 = event['s12']
        s02 = event['s13']
        s01_list, s02_list, binList = self.read_binning()[:,0], self.read_binning()[:,1], self.read_binning()[:,2]
        binList = binList.astype(float)

        # Calculate squared distances using NumPy broadcasting
        s01_list = np.array(s01_list)
        s02_list = np.array(s02_list)

        # Broadcasting to get distances:
        s01_expanded = s01[:, np.newaxis]  # Add a new axis to s01
        s02_expanded = s02[:, np.newaxis] 
        chunk_size = 1000  # Adjust this based on your memory constraints
        min_index = []

        for i in range(0, len(s01), chunk_size):
            chunk_s01 = s01_expanded[i:i + chunk_size]
            chunk_s02 = s02_expanded[i:i + chunk_size]

            distances = (s01_list - chunk_s01)**2 + (s02_list - chunk_s02)**2
            chunk_min_index = np.argmin(distances, axis=1)
            min_index.append(chunk_min_index)

        min_index = np.concatenate(min_index)

        bin = binList[min_index]
        bin = np.where(s01 < s02, bin, np.negative(bin))
        return bin
    
    def get_cisi(self, event):
        """
        Gets the Ci and Si values for an event.

        Args:
            event: The event object with s(i, j) method for calculating invariant mass.

        Returns:
            A tuple containing the Ci and Si values.
        """

        bin_list = self.find_bin_numpy(event)
        phase = event['model_phase']
        absAmp = np.abs(event['amp'])
        absAmpbar = np.abs(event['ampbar'])

        unique_bins = np.unique(bin_list)
        #Ci = np.zeros_like(unique_bins, dtype=float)
        #Si = np.zeros_like(unique_bins, dtype=float)
        Ci = {}
        Si = {}

        for i, bin_index in enumerate(unique_bins):
            mask = (bin_list == bin_index)  # Boolean mask for events in the current bin
            Ci[f'{bin_index}'] = np.sum(absAmp[mask] * absAmpbar[mask] * np.cos(phase[mask]))/np.sqrt(np.sum(absAmp[mask]**2) * np.sum(absAmpbar[mask]**2))
            Si[f'{bin_index}'] = np.sum(absAmp[mask] * absAmpbar[mask] * np.sin(phase[mask]))/np.sqrt(np.sum(absAmp[mask]**2) * np.sum(absAmpbar[mask]**2))

        return Ci, Si, unique_bins





if __name__ == "__main__":
    # Example usage
    model = 'evtgen'  # or 'ampgen'
    amplitude = Amplitude(model=model)
    amplitude.init()