from tfpcbpggsz.tensorflow_wrapper import *
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, deg_to_rad, p4_to_phsp, p4_to_srd
from tfpcbpggsz.generator.generator import GenTest, BaseGenerator, ARGenerator
from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape
from tfpcbpggsz.amp_up import *
from tfpcbpggsz.generator.generator import single_sampling2, multi_sampling, multi_sampling2
from tfpcbpggsz.core import DeltadeltaD
from tfpcbpggsz.phasecorrection import PhaseCorrection
from tfpcbpggsz.core import eff_fun

class pcbpggsz_generator:
    r"""
    PCBPGGSZ generator
    """
    def __init__(self, **kwargs):
        self.type = type
        self.gen = None

        self.Gamma=[]
        self.phsp = PhaseSpaceGenerator()
        self.Kspipi = PyD0ToKSpipi2018()
        self.Kspipi.init()
        self.charge = 1
        self.fun = None
        self.pc = None
        self.DEBUG = False
        self.apply_eff = False

    def add_bias(self, correctionType="singleBias"):
        """Adding the bias in different type"""
        self.pc = PhaseCorrection()
        self.pc.DEBUG = self.DEBUG
        self.pc.correctionType=correctionType
        self.pc.PhaseCorrection()

    def add_eff(self, charge, decay):
        """Calling the efficiency map for decay"""
        self.charge = charge
        self.decay = decay

        print(f'Efficiency applied with: {decay}_{charge}')

    def eval_bias(self, data):
        """Getting the bias value for given 4 momentum"""
        return self.pc.eval_bias(p4_to_phsp(data))
    
    def eval_eff(self, data):
        """Getting the efficiency value for given 4 momentum"""
        return eff_fun(p4_to_srd(data), self.charge, self.decay)
    
    def make_eff_fun(self):
        return self.eval_eff 

    def make_fun(self):
        """Making prod function for decay rate and efficiency"""
        return  lambda data:   self.make_eff_fun()(data) * self.formula()(data) 

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
            self.deltaB = kwargs['dB']
            self.gamma = kwargs['gamma']   
            self.charge = kwargs['charge']


        if kwargs.get('max_N') is not None:
            max_N = kwargs['max_N']

        self.fun = self.formula()
        self.prod_fun = self.make_fun() if self.apply_eff else self.formula()



        if type != 'cp_mixed':
            ret, status = multi_sampling(
                phsp,
                self.prod_fun,
                N,
                force=True,
            )

            return ret
        else:
            ret_sig, ret_tag, status = multi_sampling2(
                phsp,
                self.prod_fun,
                N,
                force=True,
            )

            return ret_sig, ret_tag




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
        ampbar_i = Kspipi.AMP(p1bar.numpy().tolist(), p3bar.numpy().tolist(), p2bar.numpy().tolist())
        ampbar_i = tf.cast(tf.negative(ampbar_i), tf.complex128)
        return ampbar_i

    @tf.function
    def amp_ag(self, data):
        """
        Calculate the amplitude (Kspipi model) of the decay from momenta.
        """    
        Kspipi = self.Kspipi
        #time_cal_amp_start = time.time()
        p1,p2,p3 = data
        # Convert TensorFlow tensors to DLPack capsules
        tf.print(p1.shape )
        # wrap with tf.py_function
        amp_i = Kspipi.AMP(p1, p2, p3)
        return amp_i
    
    @tf.function
    def ampbar_ag(self, data):
        """
        Calculate the amplitude (Kspipi model) of the decay from momenta.
        """
        Kspipi = self.Kspipi
        #time_cal_amp_start = time.time()
        p1,p2,p3 = data
        p1bar, p2bar, p3bar = tf.concat([p1[:, :1], tf.negative(p1[:, 1:])], axis=1), tf.concat([p2[:, :1], tf.negative(p2[:, 1:])], axis=1), tf.concat([p3[:, :1], tf.negative(p3[:, 1:])], axis=1)
        p1bar, p2bar, p3bar = tf.stack([tf.unstack(p1bar, axis=1)]), tf.stack([tf.unstack(p2bar, axis=1)]), tf.stack([tf.unstack(p3bar, axis=1)])
        amp_i_bar = Kspipi.AMP(p1bar, p3bar, p2bar)  # Pass tensors directly
        #amp_i_bar = Kspipi.AMP(p1bar.numpy().tolist(), p3bar.numpy().tolist(), p2bar.numpy().tolist())
        amp_i_bar = tf.cast(tf.negative(amp_i_bar), tf.complex128)
        return amp_i_bar 
    
    def formula(self):
        if self.type[:4] == 'flav':
            return  self.flavour
        elif self.type == 'cp_even' or self.type == 'cp_odd':
            return  self.cp_tag
        elif self.type == 'cp_mixed':
            return  self.cp_mixed
        elif self.type == 'b2dh':
            return self.b2dh
        elif self.type == 'phsp':
            return self.phsp_fun
        else:
            print('Invalid type')

    
    def flavour(self, data):
        """Decay rate for flav tags"""

        if self.type == 'flav':
            absAmp = tf.abs(self.ampbar(data))**2
            Gamma = absAmp
            self.Gamma = Gamma

        elif self.type == 'flavbar':
            absAmp = tf.abs(self.amp(data))**2
            Gamma = absAmp
            self.Gamma = Gamma

        return Gamma

    def cp_tag(self, data):
        """
        Decay rate for CP tag
        """

        DD_sign=-1
        phase = DeltadeltaD(self.amp(data), self.ampbar(data))
        phase_correction = tf.zeros_like(phase) if self.pc is None else self.eval_bias(data)
        phase = phase + phase_correction
        absAmp = tf.abs(self.amp(data))
        absAmpbar = tf.abs(self.ampbar(data))
        cp_sign=1 if self.type == 'cp_even' else -1
        Gamma = absAmp**2 + absAmpbar**2 + 2*DD_sign*cp_sign* absAmp * absAmpbar * tf.math.cos(phase)
        self.Gamma = Gamma
        return Gamma


    def cp_mixed(self, data_sig, data_tag):
        """
        Decay rate for CP mixed tag
        """
#        phase_sig = DeltadeltaD(self.amp_ag(data_sig), self.ampbar_ag(data_sig))
#        absAmp_sig = tf.abs(self.amp_ag(data_sig))
#        absAmpbar_sig = tf.abs(self.ampbar_ag(data_sig))
#        phase_tag = DeltadeltaD(self.amp_ag(data_tag), self.ampbar_ag(data_tag))
#        absAmp_tag = tf.abs(self.amp_ag(data_tag))
#        absAmpbar_tag = tf.abs(self.ampbar_ag(data_tag))


        phase_sig = DeltadeltaD(self.amp(data_sig), self.ampbar(data_sig))
        phase_correction_sig = tf.zeros_like(phase_sig) if self.pc is None else self.eval_bias(data_sig)
        print(phase_correction_sig) if self.DEBUG else None
        phase_sig = phase_sig + phase_correction_sig
        absAmp_sig = tf.abs(self.amp(data_sig))
        absAmpbar_sig = tf.abs(self.ampbar(data_sig))
        phase_tag = DeltadeltaD(self.amp(data_tag), self.ampbar(data_tag))
        phase_correction_tag = tf.zeros_like(phase_tag) if self.pc is None else self.eval_bias(data_tag)
        phase_tag = phase_tag + phase_correction_tag
        absAmp_tag = tf.abs(self.amp(data_tag))
        absAmpbar_tag = tf.abs(self.ampbar(data_tag))

        Gamma = (absAmp_sig*absAmpbar_tag)**2 + (absAmpbar_sig*absAmp_tag)**2 - 2*absAmp_sig*absAmpbar_tag*absAmpbar_sig*absAmp_tag*tf.math.cos(phase_sig-phase_tag)
        self.Gamma = Gamma

        return Gamma

    def b2dh(self, data):
        r"""
        Decay rate for 

        .. math::

          B^{\pm} \rightarrow D^0 h^{\mp}

        """
        rb, deltaB, gamma = self.rb, deg_to_rad(self.deltaB), deg_to_rad(self.gamma)
        absAmp = tf.abs(self.amp(data))
        absAmpbar = tf.abs(self.ampbar(data))
        phase = DeltadeltaD(self.amp(data), self.ampbar(data))
        phase_correction = tf.zeros_like(phase) if self.pc is None else self.eval_bias(data)
        phase = phase + phase_correction
        print(phase_correction) if self.DEBUG else None

        
        if self.charge==1:
            Gamma = absAmp**2*rb**2 + absAmpbar**2 + 2*rb*absAmp*absAmpbar*tf.math.cos(phase + (deltaB + gamma))
        else:
            Gamma = absAmp**2 + absAmpbar**2*rb**2 + 2*rb*absAmp*absAmpbar*tf.math.cos(phase - (deltaB - gamma))

        self.Gamma = Gamma
        return Gamma
    
    def phsp_fun(self, data):
        """
        Phase space decay rate
        """

        Gamma = tf.ones_like(data[0][:,0], dtype=tf.float64)
        return Gamma
