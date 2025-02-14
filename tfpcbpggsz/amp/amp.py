from tfpcbpggsz.amp.D0ToKSpipi2018 import *

#Kspipi = PyD0ToKSpipi2018()
#Kspipi.init()


class Amplitude:
    """
    Class for calculating the amplitude
    """
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.amplitude = None
        self.amplitudebar = None
        self.mass = None
        self.massbar = None
        self.eff = None
        self.effbar = None
        self.normalisation = None
        self.fracDD = None
        self.mass_pdf = None
        self.mass_pdfbar = None
        self.mass_pdf_data = None
        self.mass_pdfbar_data