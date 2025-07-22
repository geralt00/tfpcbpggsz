from tfpcbpggsz.ulti import get_mass, phsp_to_srd, amp_mask


class data_io:
    """Do something with the momentumfiles
    """
    def __init__(self):
        """Do something with the momentumfiles

        Args:
            data (float64): list of the momentums
        """
        self.amp = None
        self.variables = {}
        self._amp_i = None
        self.mask = None

    def load_all(self, data):
        """Load all variables from the data

        Returns:
            dic: dictionary with all variables
        """
        if len(data) == 3:
            self.variables['amp'], self.variables['ampbar'] = self.get_amplitude(data)
            self.variables['s12'], self.variables['s13'], self.variables['srd'] = self.get_mass(data)
            return self.variables

        elif len(data) == 2:
            self.variables['amp'], self.variables['ampbar'], self.variables['s12'], self.variables['s13'], self.variables['srd'] = {}, {}, {}, {}, {}
            self.variables['amp']['sig'], self.variables['ampbar']['sig'], self.variables['amp']['tag'], self.variables['ampbar']['tag'] = self.get_amplitude(data[0],data[1])
            self.variables['s12']['sig'], self.variables['s13']['sig'], self.variables['srd']['sig'], self.variables['s12']['tag'], self.variables['s13']['tag'], self.variables['srd']['tag'] = self.get_mass(data[0],data[1])
            return self.variables


    def get_amplitude(self, data, data_tag=None):
        """Get the amplitude of the data

        Returns:
            float64: the amplitude of the data
            amp, ampbar: the amplitude of the data and the conjugate of the amplitude
        """
        if self.amp is None:
            raise ValueError("Amplitude is not set. Please set the amplitude before calling this function.")
        
        amp = self.amp.amp(data)
        ampbar = self.amp.ampbar(data)
        amp_tag, ampbar_tag = None, None
        if data_tag is not None:
            amp_tag = self.amp.amp(data_tag)
            ampbar_tag = self.amp.ampbar(data_tag)
            amp, ampbar, amp_tag, ampbar_tag, self.mask = amp_mask(amp, ampbar, amp_tag, ampbar_tag)
            return amp, ampbar, amp_tag, ampbar_tag
        else:
            amp, ampbar, self.mask = amp_mask(amp, ampbar)
            return amp, ampbar
    
    def get_mass(self, data, data_tag=None):
        """Get the s12, s13, srd from the data

        Returns:
            s12, s13, srd: list of Dalitz variables
        """
        from tfpcbpggsz.generator.data import data_mask

        p1,p2,p3 = data
        s12 = get_mass(p1,p2)
        s13 = get_mass(p1,p3)
        srd = phsp_to_srd(s12,s13)
        s12_tag, s13_tag, srd_tag = None, None, None
        if data_tag is not None:
            p1_tag, p2_tag, p3_tag = data_tag
            s12_tag = get_mass(p1_tag,p2_tag)
            s13_tag = get_mass(p1_tag,p3_tag)
            srd_tag = phsp_to_srd(s12_tag,s13_tag)
            s12, s13, srd, s12_tag, s13_tag, srd_tag = data_mask(s12, self.mask), data_mask(s13, self.mask), data_mask(srd, self.mask), data_mask(s12_tag, self.mask), data_mask(s13_tag, self.mask), data_mask(srd_tag, self.mask)
            return s12, s13, srd, s12_tag, s13_tag, srd_tag
        else:
            s12, s13, srd = data_mask(s12, self.mask), data_mask(s13, self.mask), data_mask(srd, self.mask)
            return s12, s13, srd