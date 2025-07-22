import tensorflow as tf
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import  deg_to_rad, p4_to_phsp, p4_to_srd, p4_to_mag, amp_mask
from tfpcbpggsz.generator.generator import multi_sampling, multi_sampling2
from tfpcbpggsz.core import DeltadeltaD
from tfpcbpggsz.phasecorrection import PhaseCorrection
from tfpcbpggsz.core import eff_fun
from tfpcbpggsz.variable import VarsManager as vm
from tfpcbpggsz.amp.amplitude import Amplitude

class pcbpggsz_generator:
    #"""
    #Class for PCBPGGSZ generator
    #"""

    def __init__(self, amplitude, **kwargs):
        self.type = type
        self.gen = None
            
        self.Gamma=[]
        self.phsp = PhaseSpaceGenerator()
        self.charge = 'p'
        self.fun = None
        self.pc = None
        self.DEBUG = False
        self._corr_from_fit = False
        # Cache attributes
        self._cached_s12 = None
        self._cached_s13 = None
        self._raw_amp_tensor_cache = None
        self.amplitude = amplitude
        self.amplitude.init()
        self.model_name = 'evtgen'
        if not isinstance(amplitude, Amplitude):
            raise TypeError("Amplitude must be initialise before passing to the generator")
            
    

    def add_bias(self, correctionType="singleBias", **kwargs):
        

        self.pc = PhaseCorrection(vm=vm())
        self.pc.DEBUG = self.DEBUG
        self.pc.correctionType=correctionType
        if kwargs.get('coefficients') is  None:
            self.pc.PhaseCorrection()

        else:
            self.pc.order = kwargs['order']
            self.pc.PhaseCorrection()
            self.pc.set_coefficients(coefficients=kwargs['coefficients'])
            self._corr_from_fit = True




    def add_eff(self, charge, decay):
        #"""Calling the efficiency map for decay"""

        self.charge = charge
        self.decay = decay

        print(f'Efficiency applied with: {decay}_{charge}')

    def add_res(self, params):
        #"""Adding the resolution parameters"""
        self.amplitude.set_res_params(params)
        #self.amplitude.res = True
        #self.amplitude.res_params = params

    def eval_bias(self, data):
        """
        Getting the bias value for given 4 momentum
        
        
        """
        if self._corr_from_fit:
            return self.pc.eval_corr_gen(p4_to_srd(data))
        else:
            return self.pc.eval_bias(p4_to_phsp(data))
    
    def eval_eff(self, data):
        #"""Getting the efficiency value for given 4 momentum"""
        return eff_fun(p4_to_srd(data), self.charge, self.decay)
    
    def eval_res(self, data):
        #"""Getting the resolution value for given 4 momentum"""
        P_Ks, P_pim, P_pip = p4_to_mag(data)
        return 
    
    def make_eff_fun(self):
        return self.eval_eff 

    def make_fun(self):
        #"""Making prod function for decay rate and efficiency"""
        return  lambda data:   self.make_eff_fun()(data) * self.formula()(data) 

    def generate(self, N=1000, type="b2dh", **kwargs):
        #"""
        #PCBPGGSZ generator
        #Usage:
        #gen = pcbpggsz_generator()
        #"""

        phsp = PhaseSpaceGenerator().generate
        apply_eff = False
        self.type = type
        if type=="b2dh":
            self.rb = kwargs['rb']
            self.deltaB = kwargs['dB']
            self.gamma = kwargs['gamma']   
            self.charge = kwargs['charge']


        if kwargs.get('max_N') is not None:
            max_N = kwargs['max_N']
        
        if kwargs.get('apply_eff') is not None:
            apply_eff = kwargs['apply_eff']
            self.add_eff(kwargs['charge'], kwargs['decay'])


        self.fun = self.formula()
        self.prod_fun = self.make_fun() if apply_eff == True else self.formula()



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

    def formula(self):
        #"""Decay rate formula"""

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
        #"""Decay rate for flav tags"""

        if self.type == 'flav':
            absAmp = tf.abs(self.amplitude.ampbar(data))**2
            Gamma = absAmp
            self.Gamma = Gamma

        elif self.type == 'flavbar':
            absAmp = tf.abs(self.amplitude.amp(data))**2
            Gamma = absAmp
            self.Gamma = Gamma

        return Gamma

    def amp(self, data):
        #"""Amplitude for the decay"""
        return self.amplitude.amp(data)
  
    def ampbar(self, data):
        #"""Amplitude bar for the decay"""
        return self.amplitude.ampbar(data)
    
    def cp_tag(self, data):
        #"""Decay rate for CP tags"""

        DD_sign=-1
        phase = self.amplitude.DeltadeltaD(self.amplitude.amp(data), self.amplitude.ampbar(data))
        phase_correction = tf.zeros_like(phase) if self.pc is None else self.eval_bias(data)
        phase = phase + phase_correction
        absAmp = tf.abs(self.amplitude.amp(data))
        absAmpbar = tf.abs(self.amplitude.ampbar(data))
        cp_sign=1 if self.type == 'cp_even' else -1
        Gamma = absAmp**2 + absAmpbar**2 + 2*DD_sign*cp_sign* absAmp * absAmpbar * tf.math.cos(phase)
        self.Gamma = Gamma
        return Gamma


    def cp_mixed(self, data_sig, data_tag):
        #"""Decay rate for CP mixed tags"""

        phase_sig = self.amplitude.DeltadeltaD(self.amplitude.amp(data_sig), self.amplitude.ampbar(data_sig))
        phase_correction_sig = tf.zeros_like(phase_sig) if self.pc is None else self.eval_bias(data_sig)
        #print(phase_correction_sig) if self.DEBUG else None
        phase_sig = phase_sig + phase_correction_sig
        absAmp_sig = tf.abs(self.amplitude.amp(data_sig))
        absAmpbar_sig = tf.abs(self.amplitude.ampbar(data_sig))
        phase_tag = self.amplitude.DeltadeltaD(self.amplitude.amp(data_tag), self.amplitude.ampbar(data_tag))
        phase_correction_tag = tf.zeros_like(phase_tag) if self.pc is None else self.eval_bias(data_tag)
        #print(phase_correction_tag) if self.DEBUG else None
        phase_tag = phase_tag + phase_correction_tag
        absAmp_tag = tf.abs(self.amplitude.amp(data_tag))
        absAmpbar_tag = tf.abs(self.amplitude.ampbar(data_tag))
        Gamma = (absAmp_sig*absAmpbar_tag)**2 + (absAmpbar_sig*absAmp_tag)**2 - 2*absAmp_sig*absAmpbar_tag*absAmpbar_sig*absAmp_tag*tf.math.cos(phase_sig-phase_tag)
        self.Gamma = Gamma

        return Gamma

    def b2dh(self, data):
        #"""Decay rate for B2Dh decay"""

        rb, deltaB, gamma = self.rb, deg_to_rad(self.deltaB), deg_to_rad(self.gamma)
        absAmp = tf.abs(self.amplitude.amp(data))
        absAmpbar = tf.abs(self.amplitude.ampbar(data))
        phase = DeltadeltaD(self.amplitude.amp(data), self.amplitude.ampbar(data))
        phase_correction = tf.zeros_like(phase) if self.pc is None else self.eval_bias(data)
        phase = phase + phase_correction
        print(phase_correction) if self.DEBUG else None

        
        if self.charge=='p':
            Gamma = absAmp**2*rb**2 + absAmpbar**2 + 2*rb*absAmp*absAmpbar*tf.math.cos(phase + (deltaB + gamma))
        else:
            Gamma = absAmp**2 + absAmpbar**2*rb**2 + 2*rb*absAmp*absAmpbar*tf.math.cos(phase - (deltaB - gamma))

        self.Gamma = Gamma
        return Gamma
    
    def phsp_fun(self, data):
        #"""Phase space decay rate, uniform distribution"""

        Gamma = tf.ones_like(data[0][:,0], dtype=tf.float64)
        return Gamma
