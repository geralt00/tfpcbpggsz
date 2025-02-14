from tfpcbpggsz.tensorflow_wrapper import tf
from tfpcbpggsz.amp_up.D0ToKSpipi2018 import PyD0ToKSpipi2018
from tfpcbpggsz.ulti import get_mass, phsp_to_srd


class data_io:
    """Do something with the momentumfiles
    """
    def __init__(self, data):
        """Do something with the momentumfiles

        Args:
            data (float64): list of the momentums
        """
        self.data = data 
        self.Kspipi = PyD0ToKSpipi2018()
        self.variables = {}

    def load_all(self):
        """Load all variables from the data

        Returns:
            dic: dictionary with all variables
        """
        if len(self.data) == 3:
            self.variables['amp'], self.variables['ampbar'] = self.get_amplitude(self.data)
            self.variables['s12'], self.variables['s13'], self.variables['srd'] = self.get_mass(self.data)
            return self.variables

        elif len(self.data) == 2:
            self.variables['amp'], self.variables['ampbar'], self.variables['s12'], self.variables['s13'], self.variables['srd'] = {}, {}, {}, {}, {}
            for i, key in enumerate(['sig', 'tag']):
                self.variables['amp'][key], self.variables['ampbar'][key] = self.get_amplitude(self.data[i])
                self.variables['s12'][key], self.variables['s13'][key], self.variables['srd'][key] = self.get_mass(self.data[i])

            return self.variables


    def get_amplitude(self, data):
        """Get the amplitude of the data

        Returns:
            float64: the amplitude of the data
            amp, ampbar: the amplitude of the data and the conjugate of the amplitude
        """
        amp = self.amp(data)
        ampbar = self.ampbar(data)

        return amp, ampbar
    
    def amp(self, data):
        """Calculate the amplitude of the decay from momenta."""

        Kspipi = self.Kspipi
        #time_cal_amp_start = time.time()
        p1,p2,p3 = data
        if not isinstance(p1, tf.Tensor):
            amp_i = Kspipi.AMP(p1.tolist(), p2.tolist(), p3.tolist())
        else:
            amp_i = Kspipi.AMP(p1.numpy().tolist(), p2.numpy().tolist(), p3.numpy().tolist())
        return tf.cast(amp_i, tf.complex128)
   
    def ampbar(self, data):
        """Calculate the amplitude of the decay from momenta."""

        Kspipi = self.Kspipi
        #time_cal_amp_start = time.time()
        p1,p2,p3 = data
        p1bar, p2bar, p3bar = tf.concat([p1[:, :1], tf.negative(p1[:, 1:])], axis=1), tf.concat([p2[:, :1], tf.negative(p2[:, 1:])], axis=1), tf.concat([p3[:, :1], tf.negative(p3[:, 1:])], axis=1)
        ampbar_i = Kspipi.AMP(p1bar.numpy().tolist(), p3bar.numpy().tolist(), p2bar.numpy().tolist())
        ampbar_i = tf.negative(ampbar_i)
        return ampbar_i
    
    def get_mass(self, data):
        """Get the s12, s13, srd from the data

        Returns:
            s12, s13, srd: list of Dalitz variables
        """

        p1,p2,p3 = data
        s12 = get_mass(p1,p2)
        s13 = get_mass(p1,p3)
        srd = phsp_to_srd(s12,s13)

        return s12, s13, srd