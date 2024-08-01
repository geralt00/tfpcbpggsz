from tfpcbpggsz.tensorflow_wrapper import *
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, deg_to_rad
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_test import *
from tfpcbpggsz.generator.generator import single_sampling2, multi_sampling
from tfpcbpggsz.amp_test import PyD0ToKSpipi2018
from tfpcbpggsz.core import DeltadeltaD
class pcbpggsz_generator:
    """
    A generator for the decay D0 -> Ks0 pi+ pi-.
    type: str
        The type of the generator. Can be flav, flavbar, cp_even, cp_odd, cp_mix, d2dh
    """
    def __init__(self, **kwargs):
        self.type = type
        self.gen = None

        self.Gamma=[]
        self.phsp = PhaseSpaceGenerator()
        self.Kspipi = PyD0ToKSpipi2018()
        self.Kspipi.init()
        self.charge = 1


    def generate(self, N=1000, type="b2dh", **kwargs):
        """
        PCBPGGSZ generator
        Usage:
        gen = pcbpggsz_generator()
        """
        phsp = PhaseSpaceGenerator().generate

        self.type = type
        if type=="b2dh":
            self.rb = kwargs['rb']
            self.deltaB = kwargs['deltaB']
            self.gamma = kwargs['gamma']   
            self.charge = kwargs['charge']


        if kwargs.get('max_N') is not None:
            max_N = kwargs['max_N']

        fun = self.formula()
        ret, status = multi_sampling(
            phsp,
            fun,
            N,
            force=True,
        )
        return ret



    def amp(self, data):
        """
        Calculate the amplitude (Kspipi model) of the decay from momenta.
        """
        Kspipi = self.Kspipi
        #time_cal_amp_start = time.time()
        p1,p2,p3 = data
        amp_i = Kspipi.AMP(p1.numpy().tolist(), p2.numpy().tolist(), p3.numpy().tolist())    
        amp_i = tf.cast(amp_i, tf.complex128)
        return amp_i
    
    def ampbar(self, data):
        """
        Calculate the amplitude (Kspipi model) of the decay from momenta.
        """
        Kspipi = self.Kspipi
        #time_cal_amp_start = time.time()
        p1,p2,p3 = data
        p1bar, p2bar, p3bar = tf.concat([p1[:, :1], tf.negative(p1[:, 1:])], axis=1), tf.concat([p2[:, :1], tf.negative(p2[:, 1:])], axis=1), tf.concat([p3[:, :1], tf.negative(p3[:, 1:])], axis=1)
        amp_i_bar = Kspipi.AMP(p1bar.numpy().tolist(), p2bar.numpy().tolist(), p3bar.numpy().tolist())
        amp_i_bar = tf.cast(amp_i_bar, tf.complex128)
        return amp_i_bar
    
    
    def formula(self):
        if self.type[:4] == 'flav':
            return  self.flavour
        elif self.type == 'cp_even' or self.type == 'cp_odd':
            return  self.cp_tag
        elif self.type == 'cp_mix':
            return  self.cp_mixed
        elif self.type == 'b2dh':
            return self.b2dh

    
    def flavour(self, data):
 

        if self.type == 'flav':
            absAmp = tf.abs(self.amp(data))**2
            Gamma = absAmp
            self.Gamma = Gamma

        elif self.type == 'flavbar':
            absAmp = tf.abs(self.ampbar(data))**2
            Gamma = absAmp
            self.Gamma = Gamma

    def cp_tag(self, data):
        """
        Decay rate for CP tag
        """

        phase = DeltadeltaD(self.amp(data), self.ampbar(data))
        absAmp = tf.abs(self.amp(data))
        absAmpbar = tf.abs(self.ampbar(data))
        cp_sign=1 if self.type == 'cp_even' else -1
        Gamma = absAmp**2 + absAmpbar**2 + 2*cp_sign* absAmp * absAmpbar * tf.math.cos(phase)
        self.Gamma = Gamma

    def cp_mixed(self, data):
        """
        Decay rate for CP mixed tag
        """

        phase_sig = DeltadeltaD(self.amp, self.ampbar)
        absAmp_sig = tf.abs(self.amp)
        absAmpbar_sig = tf.abs(self.ampbar)
        phase_tag = DeltadeltaD(self.amp, self.ampbar)
        absAmp_tag = tf.abs(self.amp)
        absAmpbar_tag = tf.abs(self.ampbar)

        Gamma = (absAmp_sig*absAmpbar_tag)**2 + (absAmpbar_sig*absAmp_tag)**2 - 2*absAmp_sig*absAmpbar_tag*absAmpbar_sig*absAmp_tag*tf.math.cos(phase_sig-phase_tag)
        self.Gamma = Gamma

    def b2dh(self, data):
        """
        Decay rate for B -> D h
        """
        rb, deltaB, gamma = self.rb, deg_to_rad(self.deltaB), deg_to_rad(self.gamma)
        absAmp = tf.abs(self.amp(data))
        absAmpbar = tf.abs(self.ampbar(data))
        phase = DeltadeltaD(self.amp(data), self.ampbar(data))

        
        if self.charge==1:
            Gamma = absAmp**2*rb**2 + absAmpbar**2 + 2*rb*absAmp*absAmpbar*tf.math.cos(phase + (deltaB + gamma))
        else:
            Gamma = absAmp**2 + absAmpbar**2*rb**2 + 2*rb*absAmp*absAmpbar*tf.math.cos(phase - (deltaB - gamma))

        self.Gamma = Gamma
        return Gamma
