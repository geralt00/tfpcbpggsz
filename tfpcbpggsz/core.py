from tfpcbpggsz.tensorflow_wrapper import tf
import numpy as np
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.phasecorrection import * 


#Common functions
_PI = tf.constant(np.pi, dtype=tf.float64)
def DeltadeltaD(A, Abar):
    """
    Function to calculate the phase difference between the amplitude and the conjugate amplitude
    
    Args:
        A (Amplitude): the amplitude from sample
        Abar (Amplitude Bar): the conjugate amplitude from sample

    Returns:
        float64: phase difference between the amplitude and the conjugate amplitude
    """
    var_a = tf.math.angle(A*np.conj(Abar))+ _PI
    var_b = tf.where(var_a > _PI, var_a - 2*_PI, var_a)
    var = tf.where(var_b < -_PI, var_b + 2*_PI, var_b)

    return var

def name_convert(decay_str='b2dk_LL_p'):
    """
    Function to convert the name of the decay to the name used in the dataset
    """


    decay_str = decay_str.split('_')[0]+'_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    if decay_str.split('_')[0] == 'b2dpi':
        return 'DPi_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    elif decay_str.split('_')[0] == 'b2dk':
        return 'DK_KsPiPi_'+decay_str.split('_')[1]+'_'+decay_str.split('_')[2]
    
def clip_log(x, _epsilon=1e-6):
    """clip log to allowed large value"""
    x_cut = tf.where(x > _epsilon, x, tf.ones_like(x) * _epsilon)
    b_t = tf.math.log(x_cut)

    delta_x = x - _epsilon
    b_f = (
       np.log(_epsilon) + delta_x / _epsilon - (delta_x / _epsilon) ** 2 / 2.0
    )
    return tf.where(x > _epsilon, b_t, b_f)

def dalitz_transform(x_valid, y_valid):
    rotatedSymCoord = (y_valid + x_valid)/2  #z_+
    rotatedAntiSymCoord = (y_valid - x_valid)/2 #z_-

    m1_ = 2.23407421671132946
    c1_ = -3.1171885586526695
    m2_ = 0.8051636393861085
    c2_ = -9.54231895051727e-05

    stretchedSymCoord = m1_ * rotatedSymCoord + c1_
    stretchedAntiSymCoord = m2_ * rotatedAntiSymCoord + c2_
    antiSym_scale = 2.0
    antiSym_offset = 2.0
    stretchedAntiSymCoord_dp = (antiSym_scale * (stretchedAntiSymCoord)) / (antiSym_offset + stretchedSymCoord)
    return np.array([stretchedSymCoord, stretchedAntiSymCoord_dp])

def eff_fun(x, charge='p', decay='dk_LL'):
    """
    Function to calculate the efficiency value for the B2DK and B2Dpi decays
    
    """
    # in GeV !!
    res = 0
    zp_p = x[0]  # $z_{+}^{\prime}
    zm_pp = x[1]  # $z_{-}^{\prime\prime}
    Legendre_zp_1 = zp_p
    Legendre_zp_2 = (3 * np.power(zp_p, 2) - 1) / 2.
    Legendre_zp_3 = (5 * np.power(zp_p, 3) - 3 * zp_p) / 2.
    Legendre_zp_4 = (35 * np.power(zp_p, 4) - 30 * np.power(zp_p, 2) + 3) / 8.
    Legendre_zp_5 = (36 * np.power(zp_p, 5) - 70 * np.power(zp_p, 3) + 15 * zp_p) / 8.
    Legendre_zp_6 = (231 * np.power(zp_p, 6) - 315 * np.power(zp_p, 4) + 105 * np.power(zp_p, 2) - 5) / 16.
    Legendre_zm_2 = (3 * np.power(zm_pp, 2) - 1) / 2.
    Legendre_zm_4 = (35 * np.power(zm_pp, 4) - 30 * np.power(zm_pp, 2) + 3) / 8.
    Legendre_zm_6 = (231 * np.power(zm_pp, 6) - 315 * np.power(zm_pp, 4) + 105 * np.power(zm_pp, 2) - 5) / 16.
    params = {}
    offset = {}
    mean = {}
    params['bp_b2dk_LL'] = [-9.53264, 30.6442, -70.1242, -171.659, 20.997, 111.455, -25.7043, -208.538, -131.828, 5.30961, 38.3281, 52.9287, -45.1955, -92.5608, -53.7144, -6.41475]
    params['bm_b2dk_LL'] = [-11.6916, 36.6738, -75.5949, -177.263, 24.6625, 128.041, -29.1374, -222.949, -133.375, 5.42292, 50.5985, 63.6018, -44.7697, -94.6231, -65.164, -7.16556]
    params['bp_b2dpi_LL'] = [-11.726, 36.3723, -79.9267, -178.024, 26.746, 125.069, -31.2218, -233.282, -133.757, 6.55028, 48.6604, 59.0497, -43.2816, -99.2002, -65.8039, -6.45939]
    params['bm_b2dpi_LL'] = [-11.2904, 33.8601, -76.9433, -178.023, 25.2621, 118.649, -30.6079, -226.689, -135.131, 5.66233, 46.1495, 57.1009, -45.8845, -98.3241, -65.0337, -6.8956]
    params['bp_b2dk_DD'] = [-30.8217, 98.3102, -201.671, -450.003, 67.5668, 336.798, -77.9681, -580.061, -337.139, 17.1748, 135.283, 166.737, -108.95, -248.926, -160.071, -16.7914]
    params['bm_b2dk_DD'] = [-28.5793, 92.1537, -192.655, -445.199, 66.4849, 315.014, -76.3822, -547.47, -336.686, 17.0215, 130.277, 144.332, -112.366, -224.548, -150.207, -19.5028]
    params['bp_b2dpi_DD'] = [-25.1893, 83.6515, -191.42, -437.186, 59.6098, 286.787, -79.8558, -540.406, -331.55, 13.7906, 114.149, 135.039, -114.09, -219.42, -167.023, -22.0707 ]
    params['bm_b2dpi_DD'] = [-27.3281, 77.6543, -181.93, -451.338, 51.6441, 278.849, -66.5276, -522.833, -353.474, 11.865, 98.5612, 138.88, -121.508, -228.435, -139.317, -15.4476 ]
    offset['bp_b2dk_LL'] = 57.060094380400386
    offset['bm_b2dk_LL'] = 58.17587468590285
    offset['bp_b2dk_DD'] = 144.4818375757835
    offset['bm_b2dk_DD'] = 149.4970954503175
    offset['bp_b2dpi_LL'] = 58.02066608312892
    offset['bm_b2dpi_LL'] = 57.2679149533671
    offset['bp_b2dpi_DD'] = 143.85275330726384
    offset['bm_b2dpi_DD'] = 145.64578485313336
    mean['bp_b2dk_LL'] = 94.71806024672841
    mean['bm_b2dk_LL'] = 95.67192393339243
    mean['bp_b2dk_DD'] = 236.7559605125362
    mean['bm_b2dk_DD'] = 243.96305838750337
    mean['bp_b2dpi_LL'] = 95.5957545077163
    mean['bm_b2dpi_LL'] = 95.47377527333916
    mean['bp_b2dpi_DD'] = 238.0281142413688
    mean['bm_b2dpi_DD'] = 241.6462132232453

    decay = 'b'+charge+'_'+decay

    res = (
        params[decay][0]
        + params[decay][1] * Legendre_zp_1
        + params[decay][2] * Legendre_zp_2
        + params[decay][3] * Legendre_zm_2
        + params[decay][4] * Legendre_zp_3
        + params[decay][5] * Legendre_zp_1 * Legendre_zm_2
        + params[decay][6] * Legendre_zp_4
        + params[decay][7] * Legendre_zp_2 * Legendre_zm_2
        + params[decay][8] * Legendre_zm_4
        + params[decay][9] * Legendre_zp_5
        + params[decay][10] * Legendre_zm_2 * Legendre_zp_3
        + params[decay][11] * Legendre_zm_4 * Legendre_zp_1
        + params[decay][12] * Legendre_zm_6
        + params[decay][13] * Legendre_zm_4 * Legendre_zp_2
        + params[decay][14] * Legendre_zm_2 * Legendre_zp_4
        + params[decay][15] * Legendre_zp_6
    )

    return( res+offset[decay])/mean[decay]

def prob_totalAmplitudeSquared_XY(Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0,0), pc=None):
    """
    Function to calculate the amplitude squared for the B2DK and B2Dpi decays

    Args:
        Bsign (int, optional): the charge of B meson. Defaults to 1.
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        x (tuple, optional): the fitted params. Defaults to (0,0,0,0,0,0).
        pc (float, optional): the phase correction class. Defaults to None.

    Returns:
        float64: the amplitude squared
    """

    phase = DeltadeltaD(amp, ampbar)
    if pc is not None:
        phase = phase + pc

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)


    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)

        return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus * tf.cos(phase) - yPlus * tf.sin(phase)))
    
    elif Bsign == -1:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)

        return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus * tf.cos(phase) + yMinus * tf.sin(phase)))

def prob_totalAmplitudeSquared_CP_mix(amp_sig=[], ampbar_sig=[],amp_tag=[], ampbar_tag=[], pc_sig=None, pc_tag=None):
    """
    Function to calculate the amplitude squared for the D0->KsPiPi and D0bar->KsPiPi decays

    Args:
        amp_sig (Amplitude): the amplitude from signal sample
        ampbar_sig (Amplitude Bar): the conjugate amplitude from signal sample
        amp_tag (Amplitude): the amplitude from tag sample
        ampbar_tag (Amplitude Bar): the conjugate amplitude from tag sample
        pc_sig (float, optional): the phase correction for the signal. Defaults to None.
        pc_tag (float, optional): the phase correction for the tag. Defaults to None.

    Returns:
        float64: the amplitude squared
    """


    phase_sig = DeltadeltaD(amp_sig, ampbar_sig)
    phase_tag = DeltadeltaD(amp_tag, ampbar_tag)

    if pc_sig is not None:
        phase_sig = phase_sig + pc_sig
    if pc_tag is not None:
        phase_tag = phase_tag + pc_tag

    absA_sig = tf.cast(tf.abs(amp_sig), tf.float64)
    absAbar_sig = tf.cast(tf.abs(ampbar_sig), tf.float64)
    absA_tag = tf.cast(tf.abs(amp_tag), tf.float64)
    absAbar_tag = tf.cast(tf.abs(ampbar_tag), tf.float64)

    return (absA_sig*absAbar_tag)**2 + (absAbar_sig*absA_tag)**2 - 2*absA_sig*absAbar_tag*absAbar_sig*absA_tag*tf.math.cos(phase_sig-phase_tag)

def prob_totalAmplitudeSquared_CP_tag(CPsign=1, amp=[], ampbar=[], pc=None):
    """
    Function to calculate the amplitude squared for the CP tag decay

    Args:
        CPsign (int, optional): the CP sign for the decay. Defaults to +.
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        pc (float, optional): the phase correction class. Defaults to None.

    Returns:
        float64: the amplitude squared
    """



    DDsign = -1
    phase = DeltadeltaD(amp, ampbar)
    if pc is not None:
        phase = phase + pc

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)



    return (absA**2  + absAbar **2  + 2.0 * DDsign * CPsign * (absA * absAbar) * tf.cos(phase))

def prob_totalAmplitudeSquared_DPi_XY( Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0,0,0,0,0,0,0,0,0,0)):
    """
    Function to calculate the amplitude squared for the B2Dpi decay, the ratio between B2DK and B2Dpi is used, then two additional parameters are added

    Args:
        Bsign (int, optional): the charge of B meson. Defaults to 1.
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        x (tuple, optional): the fitted params. Defaults to (0,0,0,0,0,0,0,0,0,0,0,0,0,0).

    Returns:
        float64: the amplitude squared
    """

    phase = DeltadeltaD(amp, ampbar)
    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)

    xXi = tf.cast(x[4], tf.float64)
    yXi = tf.cast(x[5], tf.float64)

    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
        yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)
        rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)

        return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus_DPi * tf.cos(phase) - yPlus_DPi * tf.sin(phase)))
    
    elif Bsign == -1:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
        yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)
        rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)


        return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus_DPi * tf.cos(phase) + yMinus_DPi * tf.sin(phase)))


def prob_comb(amp=[], ampbar=[], normA=1.2, normAbar=1.2, fracDD=0.82, eff1=[], eff2=[]):
    """
    Function to calculate the amplitude squared for the combinatorial background in the B2DK and B2Dpi decays

    Args:
        amp (Amplitude): the amplitude from sample
        ampbar (Amplitude Bar): the conjugate amplitude from sample
        normA (float, optional): the normalisation of the amplitude squared for the decay. Defaults to 1.2.
        normAbar (float, optional): the normalisation of the amplitude squared for the decay. Defaults to 1.2.
        fracDD (float, optional): the fraction of the combinatorial background. Defaults to 0.82.
        eff1 (float, optional): the efficiency value for the B2DK decay. Defaults to [].
        eff2 (float, optional): the efficiency value for the B2Dpi decay. Defaults to [].

    Returns:
        float64: the amplitude squared
    """

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)
    normA = tf.cast(normA, tf.float64)
    normAbar = tf.cast(normAbar, tf.float64)
    frac1 = fracDD*0.5
    frac2 = 1.0 - fracDD
    prob1 = eff1*absA**2 /normA
    prob2 = eff2*absAbar**2 /normAbar 
    prob3 = tf.ones_like(absA)


    return (prob1 * frac1 + prob2 * frac1 + prob3 * frac2)


#Probabily just for validation stage, should be more general for the Normalisation class
class Normalisation:
    """
    Normalisation class, contains the formula we need with charm dataset
    Args:
        amp_MC: dict, dictionary of the amplitude for the MC
        ampbar_MC: dict, dictionary of the conjugate amplitude for the MC
        name: str, decay name, e.g. b2dk

    """
    def __init__(self, amp_MC, ampbar_MC, name='b2dk'):

        self._name = name
        if self._name[3] == 'k':
            self._name_misid = self._name.replace('k', 'pi')
        elif self._name[3] == 'p':
            self._name_misid = self._name.replace('pi', 'k')
        else:
            self._name_misid = None

        self.amp_MC = amp_MC
        self.ampbar_MC = ampbar_MC
        self._normA = None
        self._normAbar = None
        self._normA_misid = None
        self._normAbar_misid = None
        self._A = None
        self._Abar = None
        self._A_tag = None
        self._Abar_tag = None
        self._phase = None
        self._phase_tag = None
        self._phase_misid = None
        self._crossTerms = [None, None]
        self._crossTerms_misid = [None, None]
        self._crossTerms_complex = None
        self._BacTerms = [None, None]
        self._BacTerms_misid = [None, None]
        self._BacTerms_bkg = [None, None]
        self._AAbar = None
        self._AAbar_tag = None
        self._AAbar_misid = None
        self._params = None
        self._DEBUG = False
        self._phaseCorrection = None
        self._phaseCorrection_tag = None
        self.tagged_i = None

    def add_pc(self, pc,**kwargs):
        """
        Function to add the phase correction to the phase

        Args:
            pc (float): the phase correction
        """
        if kwargs.get('pc_tag') is not None:
            self._phaseCorrection_tag = kwargs.get('pc_tag')

        self._phaseCorrection = pc

    def debug(self):
        self._DEBUG = True
        
    def initialise(self):
        """
        Function to initialise the normalisation terms
        """

        print('Initialising normalisation for decay:', self._name)
        self.normA()
        self.normAbar()
        self.AAbar()
        self.phase()
        if self._DEBUG:
            print('Normalisation terms:\n |A|^2:', self._normA, '\n |Abar|^2:', self._normAbar, '\n |A||Abar|cos(phase):', self._crossTerms[0], '\n |A||Abar| sin(phase):', self._crossTerms[1])

    def initialise_misid(self):     
        """
        Function to initialise the normalisation terms for the misid decay
        """

        print('Initialising misid normalisation for decay:', self._name_misid)
        if self.amp_MC.get(self._name_misid) is None:
            raise ValueError('Misid amplitude not found')
        
        self.normA_misid()
        self.normAbar_misid()
        self.AAbar_misid()
        self.phase_misid()
        if self._DEBUG:
            print('Normalisation terms:\n |A|^2:', self._normA_misid, '\n |Abar|^2:', self._normAbar_misid, '\n |A||Abar| cos(phase):', self._crossTerms_misid[0], '\n |A||Abar| sin(phase):', self._crossTerms_misid[1])


    def setParams(self, x):
        self._params = x

    def phase(self):
        r"""
        This function calculates the phase related terms, including:
        .. math::

           \begin{aligned}
           \delta_D &= \\
           |A||\bar{A}| \cos(\delta_D) &= \\
           |A||\bar{A}| \sin(\delta_D) &= \\
           |A_{sig}|^2 |\bar{A}_{tag}|^2 + |\bar{A}_{sig}|^2 |A_{tag}|^2 - 2 |A||\bar{A}| |\bar{A}_{sig}||A_{tag}| \cos(\delta_D - \delta_{D_{tag}})
           \end{aligned} 

        Returns:
        float64: phase between the amplitude and the conjugate amplitude
        """
        if self._phase is None:
            self._phase = DeltadeltaD(self.amp_MC[self._name], self.ampbar_MC[self._name])
            self._crossTerms[0] = tf.math.reduce_mean(self._AAbar*tf.cos(self._phase))
            self._crossTerms[1] = tf.math.reduce_mean(self._AAbar*tf.sin(self._phase))
            self._phase_tag =  DeltadeltaD(self.amp_MC[self._name.replace('_sig', '_tag')], self.ampbar_MC[self._name.replace('_sig', '_tag')]) #tf.gather(self._phase, self.tagged_i) #+ phase_correction.eval_tf(tf.gather(events, tagged_i))
            self._crossTerms_complex =  tf.math.reduce_mean(
                (tf.abs(self._A) * tf.abs(self._Abar_tag)) ** 2
                + (tf.abs(self._Abar) * tf.abs(self._A_tag)) ** 2
                - 2 * self._AAbar * self._AAbar_tag * tf.cos(self._phase - self._phase_tag)
            )

        return self._phase

    def normA(self):
        """
        This function calculates the normalisation of the amplitude squared for the decay.

        Returns:
        float64: 
        
        .. math: |A|^2
        """
        if self._normA is None:
            self._normA = tf.math.reduce_mean(tf.abs(self.amp_MC[self._name])**2)
            self._BacTerms[0] = self._normA
            self._A = self.amp_MC[self._name]
            #Old way to tag the events
            #self.tagged_i = (tf.range(self._A.shape[0]) + self._A.shape[0] // 2) % self._A.shape[0]
            #self._A_tag = tf.gather(self._A, self.tagged_i)
            self._A_tag = self.amp_MC[self._name.replace('_sig', '_tag')]


            #if self._name[3] == 'p' and self._name[-1] == 'm':
            #    self._BacTerms_bkg[0] = self._normA

        return self._normA
    
    def normAbar(self):
        r"""
        This function calculates the normalisation of the amplitude squared for the decay.

        Returns:
        float64: 
        
        .. math: |Abar|^2

        """
        if self._normAbar is None:
            self._normAbar = tf.math.reduce_mean(tf.abs(self.ampbar_MC[self._name])**2)
            self._BacTerms[1] = self._normAbar
            self._Abar = self.ampbar_MC[self._name]
            self._Abar_tag = self.ampbar_MC[self._name.replace('_sig', '_tag')]
            #self._Abar_tag = tf.gather(self._Abar, self.tagged_i)
            #if self._name[3] == 'p' and self._name[-1] == 'p':
            #    self._BacTerms_bkg[1] = self._normAbar

        return self._normAbar

    def AAbar(self):
        r"""
        This function calculates the product of the amplitude and the conjugate amplitude for the decay.

        Returns:
            
        float64:

        .. math:: |A| |\bar{A}| 

        """

        if self._AAbar is None:
            self._AAbar = tf.abs(self.amp_MC[self._name]) * tf.abs(self.ampbar_MC[self._name])
            self._AAbar_tag = tf.abs(self._A_tag) * tf.abs(self._Abar_tag) 

            #tagged_i = (tf.range(self.AAbar.shape[0]) + self.AAbar.shape[0] // 2) % self.AAbar.shape[0]
            #self._AAbar_tag = tf.abs(tf.gather(self.amp_MC[self._name],tagged_i)) * tf.abs(tf.gather(self.ampbar_MC[self._name],tagged_i))

        return self._AAbar, self._AAbar_tag

    def Update_crossTerms(self):
        r"""
        This function calculates the cross terms between the amplitude and the conjugate amplitude.
        if there is a phase correction, it will be added to the phase

        Returns:
        float64: 

        .. math:: |A| |\bar{A}| \cos(\delta_D), |A| |\bar{A}| \sin(\delta_D)
        """
        if self._crossTerms[0] is not None and self._crossTerms[1] is not None and self._phaseCorrection is not None:

            self._crossTerms[0] = tf.math.reduce_mean(self._AAbar*tf.cos(self._phase+self._phaseCorrection))
            self._crossTerms[1] = tf.math.reduce_mean(self._AAbar*tf.sin(self._phase+self._phaseCorrection))
            if self._phaseCorrection_tag is not None:
                self._crossTerms_complex =  tf.math.reduce_mean(
                (tf.abs(self._A) * tf.abs(self._Abar_tag)) ** 2
                + (tf.abs(self._Abar) * tf.abs(self._A_tag)) ** 2
                    - 2 * self._AAbar * self._AAbar_tag * tf.cos(self._phase + self._phaseCorrection - self._phase_tag-self._phaseCorrection_tag)
                )
    
        return self._crossTerms

    def Integrated_BacTerms(self):
        r"""
        This function calculates the Bachelor terms for the decay 

        .. math:: |A|^2, |Abar|^2

        Returns:
        float64: 

        .. math:: |A|^2, |Abar|^2

        """
        if self._BacTerms[0] is None or self._BacTerms[1] is None:
            self._BacTerms[0] = tf.math.reduce_mean(tf.abs(self.amp_MC))
            self._BacTerms[1] = tf.math.reduce_mean(tf.abs(self.ampbar_MC))

        return self._BacTerms
    
    def Integrated_4p(self, Bsign=1):
        r"""

        .. math:: A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)
    
        .. math:: A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)

        Args:
        Bsign: int, 1 for B+ and -1 for B-

        Returns:
        float64: Integrated amplitude squared, the normalisation is taking from the misid decay
        """
        if self._normA is None or self._normAbar is None or self._crossTerms is None:
            raise ValueError('Please calculate the normalisation and cross terms first')
        
        normA = self._normA_misid
        normAbar = self._normAbar_misid
        crossTerm = self._crossTerms_misid
        x = self._params


        if Bsign == 1:
            xPlus = tf.cast(x[0], tf.float64)
            yPlus = tf.cast(x[1], tf.float64)
            rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)
    
            return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus *crossTerm[0] - yPlus * crossTerm[1]), tf.float64)
        
        else:
            xMinus = tf.cast(x[2], tf.float64)
            yMinus = tf.cast(x[3], tf.float64)
            rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)
    
            return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus *crossTerm[0] + yMinus * crossTerm[1]), tf.float64)    

    def Integrated_4p_sig(self, Bsign=1):
        r"""

        .. math:: A^2 * rb^2 + \bar{A}^2 + 2*|A||\bar{A}| * rb * cos(\delta_B + \gamma + \delta_D)

        .. math:: A^2 + \bar{A}^2 * rb^2 + 2*|A||\bar{A}| * rb * cos(\delta_B + \gamma - \delta_D)

        Args:
        Bsign: int, 1 for B^+ and -1 for B^-

        Returns:
        float64: Integrated amplitude squared, the normalisation is taking from the signal decay

        """
        if self._normA is None or self._normAbar is None or self._crossTerms is None:
            raise ValueError('Please calculate the normalisation and cross terms first')
        
        normA = self._normA
        normAbar = self._normAbar
        crossTerm = self._crossTerms
        x = self._params


        if Bsign == 1:
            xPlus = tf.cast(x[0], tf.float64)
            yPlus = tf.cast(x[1], tf.float64)
            rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)
    
            return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus *crossTerm[0] - yPlus * crossTerm[1]), tf.float64)
        
        else:
            xMinus = tf.cast(x[2], tf.float64)
            yMinus = tf.cast(x[3], tf.float64)
            rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)
    
            return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus *crossTerm[0] + yMinus * crossTerm[1]), tf.float64)    


    def Integrated_CP_tag(self, CPsign=1):
        r"""

        .. math:: 

        \begin{aligned}
        A^2 * rb^2 + \bar{A}^2 + DDsign*CPsign* 2*|A||Abar| * rb * cos(\delta_B + \gamma + \delta_D)
        \end{aligned}

        - With Psi(3770), DDsign = -1


        Args:
            CPsign (int, optional): the CP sign for the decay. Defaults to +.

        Returns:
            float64: Integrated amplitude squared, the normalisation is taking from the signal decay
        """
        if self._normA is None or self._normAbar is None or self._crossTerms is None:
            raise ValueError('Please calculate the normalisation and cross terms first')
        
        DD_sign=-1
        normA = self._normA
        normAbar = self._normAbar
        crossTerm = self._crossTerms

        return tf.cast(normA + normAbar + 2.0 * DD_sign* CPsign * crossTerm[0], tf.float64)


    def Integrated_4p_a(self, Bsign=1, x=(0,0,0,0)):
        r"""
        .. math::  A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)


        Args:
            Bsign (int, optional): the charge of B meson. Defaults to 1.
            x (tuple, optional): the fitted params. Defaults to (0,0,0,0).

        Returns:
            float64: Integrated amplitude squared, the normalisation is taking from the signal decay
            
        """
        if self._normA is None or self._normAbar is None or self._crossTerms is None:
            raise ValueError('Please calculate the normalisation and cross terms first')
        
        normA = self._normA
        normAbar = self._normAbar
        crossTerm = self._crossTerms


        if Bsign == 1:
            xPlus = tf.cast(x[0], tf.float64)
            yPlus = tf.cast(x[1], tf.float64)
            rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)
    
            return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus *crossTerm[0] - yPlus * crossTerm[1]), tf.float64)
        
        else:
            xMinus = tf.cast(x[2], tf.float64)
            yMinus = tf.cast(x[3], tf.float64)
            rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)
    
            return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus *crossTerm[0] + yMinus * crossTerm[1]), tf.float64) 


    def Integrated_fullchain(self, Bsign=1):
        r"""

        .. math:: 

        \begin{aligned}
        A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)
        \end{aligned}

        Args:
            Bsign (int, optional): the charge of B meson. Defaults to 1.

        Returns:
            float64: Integrated amplitude squared, the normalisation is taking from the signal decay
        """
        x = self._params
        absA = self.amp_MC
        absAbar = self.ampbar_MC
        phase = DeltadeltaD(absA, absAbar)
        
        if Bsign == 1:
            xPlus = tf.cast(x[0], tf.float64)
            yPlus = tf.cast(x[1], tf.float64)
            rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)
    
            Gp = tf.cast(absA**2 * rB2 + absAbar**2 + 2.0 * (absA * absAbar) *(xPlus * tf.cos(phase) - yPlus * tf.sin(phase)), tf.float64)
    
            return tf.math.reduce_mean(Gp)
        
        else:
            xMinus = tf.cast(x[2], tf.float64)
            yMinus = tf.cast(x[3], tf.float64)
            rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)
            Gm = tf.cast(absA**2 + absAbar**2  * rB2 + 2.0 * (absA * absAbar) *(xMinus * tf.cos(phase) + yMinus * tf.sin(phase)), tf.float64)
    
            return tf.math.reduce_mean(Gm)
        
    def phase_misid(self):
        """
        This function calculates the phase between the amplitude and the conjugate amplitude for the misid decay.

        Returns:
        float64: phase between the amplitude and the conjugate amplitude for the misid decay
        """

        if self._phase_misid is None:
            self._phase_misid = DeltadeltaD(self.amp_MC[self._name_misid], self.ampbar_MC[self._name_misid])
            self._crossTerms_misid[0] = tf.math.reduce_mean(self._AAbar_misid*tf.cos(self._phase_misid))
            self._crossTerms_misid[1] = tf.math.reduce_mean(self._AAbar_misid*tf.sin(self._phase_misid))

        return self._phase_misid
    

    def normA_misid(self):
        """
        This function calculates the normalisation of the amplitude squared for the decay.

        Returns:
        float64: 
        
        .. math:: |A|^2
        """
        if self._normA_misid is None:
            self._normA_misid = tf.math.reduce_mean(tf.abs(self.amp_MC[self._name_misid])**2)
            self._BacTerms_misid[0] = self._normA_misid
            #if self._name[3] == 'k' and self._name[-1] == 'm':
            #    self._BacTerms_bkg[0] = self._normA_misid

        return self._normA_misid
    
    def normAbar_misid(self):
        r"""
        This function calculates the normalisation of the amplitude squared for the decay.

        Returns:

        float64: 

        .. math:: |\bar{A}|^2

        """
        if self._normAbar_misid is None:
            self._normAbar_misid = tf.math.reduce_mean(tf.abs(self.ampbar_MC[self._name_misid])**2)
            self._BacTerms_misid[1] = self._normAbar_misid
            #if self._name[3] == 'k' and self._name[-1] == 'p':
            #    self._BacTerms_bkg[1] = self._normAbar_misid

        return self._normAbar_misid

    def AAbar_misid(self):
        r"""
        This function calculates the product of the amplitude and the conjugate amplitude for the decay.

        Returns:

        float64:    

        .. math:: |A||\bar{A}|

        """

        if self._AAbar_misid is None:
            self._AAbar_misid = tf.abs(self.amp_MC[self._name_misid]) * tf.abs(self.ampbar_MC[self._name_misid])
    
        return self._AAbar_misid

    def Integrated_crossTerms_misid(self):
        r"""
        This function calculates the cross terms between the amplitude and the conjugate amplitude for the misid decay.

        Returns:
        
        float64: 

        .. math:: 

        \begin{aligned}
        | A || \bar{A}| cos(\delta_D)
        | A || \bar{A}| sin(\delta_D)
        \end{aligned}

        """

        if self._crossTerms_misid[0] is None or self._crossTerms[1] is None:

            self._crossTerms_misid[0] = tf.math.reduce_mean(self._AAbar_misid*tf.cos(self._phase_misid))
            self._crossTerms_misid[1] = tf.math.reduce_mean(self._AAbar_misid*tf.sin(self._phase_misid))
    
        return self._crossTerms_misid

    def Integrated_BacTerms_misid(self):
        r"""
        This function calculates the Bachelor terms for the misid decay 
        
        .. math:: |A|^{2}, |Abar|^{2}

        Returns:
        float64: 

        .. math:: |A|^{2}, |Abar|^{2}

        """

        if self._BacTerms_misid[0] is None or self._BacTerms_misid[1] is None:
            self._BacTerms_misid[0] = tf.math.reduce_mean(tf.abs(self.amp_MC[self._name_misid]))
            self._BacTerms_misid[1] = tf.math.reduce_mean(tf.abs(self.ampbar_MC[self._name_misid]))

        return self._BacTerms_misid
    
        
    def Integrated_6p(self, Bsign=1):
        r"""
        .. math::     
            \Gamma_{+} =  A^{2} * rb^{2} + Abar^{2} + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)

        .. math::     
            \Gamma_{-} =  A^{2} + Abar^{2} * rb^{2} + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)


        Args:
        Bsign: int, 1 for B+ and -1 for B-

        Returns:

        float64: Integrated amplitude squared, the normalisation is taking from the misid decay


        """
        x = self._params
        if len(x) <6:
            raise ValueError('x should have 6 elements')
        
        xXi = tf.cast(x[4], tf.float64)
        yXi = tf.cast(x[5], tf.float64)
        normA = self._normA_misid
        normAbar = self._normAbar_misid
        crossTerm = self._crossTerms_misid

        if Bsign == 1:
            xPlus = tf.cast(x[0], tf.float64)
            yPlus = tf.cast(x[1], tf.float64)
            xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
            yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)
    
            rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)
            return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus_DPi *crossTerm[0] - yPlus_DPi * crossTerm[1]), tf.float64)
        
        else:
            xMinus = tf.cast(x[2], tf.float64)
            yMinus = tf.cast(x[3], tf.float64)
            xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
            yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)
    
            rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)
    
            return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus_DPi *crossTerm[0] + yMinus_DPi * crossTerm[1]), tf.float64)

    def Integrated_6p_sig(self, Bsign=1):
        r"""
        .. math:: A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)
    
        .. math:: A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)

        Args:
        Bsign: int, 1 for B+ and -1 for B-

        Returns:
        float64: Integrated amplitude squared, the normalisation is taking from the signal decay


        """
        x = self._params
        if len(x) <6:
            raise ValueError('x should have 6 elements')
        
        xXi = tf.cast(x[4], tf.float64)
        yXi = tf.cast(x[5], tf.float64)
        normA = self._normA
        normAbar = self._normAbar
        crossTerm = self._crossTerms

        if Bsign == 1:
            xPlus = tf.cast(x[0], tf.float64)
            yPlus = tf.cast(x[1], tf.float64)
            xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
            yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)
    
            rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)
            return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus_DPi *crossTerm[0] - yPlus_DPi * crossTerm[1]), tf.float64)
        
        else:
            xMinus = tf.cast(x[2], tf.float64)
            yMinus = tf.cast(x[3], tf.float64)
            xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
            yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)
    
            rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)
    
            return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus_DPi *crossTerm[0] + yMinus_DPi * crossTerm[1]), tf.float64)



class DecayNLLCalculator:
    """
    Class to calculate negative log-likelihood for various decay types.
    """

    def __init__(self, amp_data, ampbar_data, normalisations, mass_pdfs, eff_arr, fracDD, params,  name='B2DK'):
        self._name = name
        self.amp_data = amp_data
        self.ampbar_data = ampbar_data
        self.mass_pdfs = mass_pdfs
        self.eff_arr = eff_arr
        self.params = params
        self.fracDD = fracDD
        self.normalisations = normalisations
        self._normalisations = {}

        self._nll = {}
        self._prob = {}
        self._prod_prob = {}
        self._ret = {}
        self._phase = {}
        self._absA = {}
        self._absAbar = {}
        self._phase_correction = {}

    def initialise(self):
        """
        Initialise the amplitude and phase.
        """
        for charge in ['p', 'm']:
            name = self._name + '_' + charge
            if len(self._absA.keys()) < 2:
                print('Initialising amplitude and phase for decay:', name)

                self._absA[name] = tf.abs(self.amp_data[name])
                self._absAbar[name] = tf.abs(self.ampbar_data[name])
                self._phase[name] = DeltadeltaD(self.amp_data[name], self.ampbar_data[name])


        self.norm_bkg()
        self.norm()
        self.make_prob()
        self.make_prod_prob()
        self.nll()
        #print('Negative log-likelihood:', self._nll)

    def norm_bkg(self):

        for charge in ['p', 'm']:
            decay = self._name + '_' + charge
            decay_p = self._name + '_p'
            decay_m = self._name + '_m'
            #Making sure it is always the DPi events as normalisation
            if self._name[3] == 'k':
                self.normalisations[decay]._BacTerms_bkg[0], self.normalisations[decay]._BacTerms_bkg[1] = (
                    self.normalisations[decay_m]._BacTerms_misid[0], self.normalisations[decay_p]._BacTerms_misid[1]
                )
            elif self._name[3] == 'p':#For Dpi, prepare for the upgrated version
                self.normalisations[decay]._BacTerms_bkg[0], self.normalisations[decay]._BacTerms_bkg[1] = (
                    self.normalisations[decay_m]._BacTerms[0], self.normalisations[decay_p]._BacTerms[1]
                )





    def norm(self):
        """
        Calculate the normalisation of the amplitude squared.
        """
        for charge in ['p', 'm']:
            decay = self._name + '_' + charge
            charge_flag = 1 if charge == 'p' else -1
            self._normalisations[decay] = {} if self._normalisations.get(decay) is None else self._normalisations[decay]
            self.normalisations[decay].setParams(self.params) 
            self._normalisations[decay]['sig'] = self.normalisations[decay].Integrated_4p_sig(charge_flag)

            if len(self.params) > 4:
                if self._name[3] == 'k':
                    self._normalisations[decay]['misid'] = self.normalisations[decay].Integrated_6p(charge_flag)
                if self._name[3] == 'p':
                    self._normalisations[decay]['sig'] = self.normalisations[decay].Integrated_6p_sig(charge_flag)
                    self._normalisations[decay]['misid'] = self.normalisations[decay].Integrated_4p(charge_flag)


            if len(self._normalisations[decay].keys()) <= 4:
                #print('Calculating normalisation for decay:', decay)
                self._normalisations[decay]['comb_a'], self._normalisations[decay]['comb_abar'] = (
                    self.normalisations[decay]._BacTerms_bkg[0], self.normalisations[decay]._BacTerms_bkg[1]
                )
                if charge == 'p':
                    self._normalisations[decay]['low'] = self.normalisations[decay]._BacTerms_bkg[1]
                    if self._name[3] == 'k':#Only for DK mode
                        self._normalisations[decay]['low_misID'] = self.normalisations[decay]._BacTerms_bkg[1]
                        self._normalisations[decay]['low_Bs2DKPi'] = self.normalisations[decay]._BacTerms_bkg[0]
                else:
                    self._normalisations[decay]['low'] = self.normalisations[decay]._BacTerms_bkg[0]
                    if self._name[3] == 'k':#Only for DK mode
                        self._normalisations[decay]['low_misID'] = self.normalisations[decay]._BacTerms_bkg[0]
                        self._normalisations[decay]['low_Bs2DKPi'] = self.normalisations[decay]._BacTerms_bkg[1]

                    

    def make_prob(self):
        """
        Calculate the Amplitude PDF
        """
   
        for charge in ['p', 'm']:
            decay = self._name + '_' + charge
            charge_flag = 1 if charge == 'p' else -1
            self._prob[decay] = {} if self._prob.get(decay) is None else self._prob[decay]
            self._prob[decay]['sig'] = self.eff_arr[decay]['sig'] * self.totalAmplitudeSquared_XY(charge_flag) / self._normalisations[decay]['sig']
            self._prob[decay]['comb'] = self.totalAmplitudeSquared_comb(charge_flag)
            self._prob[decay]['low'] = self.eff_arr[decay]['low'] * self.totalAmplitudeSquared_low(charge_flag) / self._normalisations[decay]['low']
            self._prob[decay]['misid'] = self.eff_arr[decay]['misid'] * self.totalAmplitudeSquared_DPi_XY(charge_flag)/self._normalisations[decay]['misid']

            if len(self.params) > 4:
                if self._name[3] == 'k':
                    self._prob[decay]['misid'] = self.eff_arr[decay]['misid'] * self.totalAmplitudeSquared_DPi_XY(charge_flag)/self._normalisations[decay]['misid']

                elif self._name[3] == 'p':
                    self._prob[decay]['sig'] = self.eff_arr[decay]['sig'] * self.totalAmplitudeSquared_DPi_XY(charge_flag)/self._normalisations[decay]['sig']
                    self._prob[decay]['misid'] = self.eff_arr[decay]['misid'] * self.totalAmplitudeSquared_XY(charge_flag)/self._normalisations[decay]['misid']
                    
            if self._name[3] == 'k':
                self._prob[decay]['low_misID'] = self.eff_arr[decay]['low'] * self.totalAmplitudeSquared_low(charge_flag) / self._normalisations[decay]['low_misID']
                if charge == 'p':
                    self._prob[decay]['low_Bs2DKPi'] = self.eff_arr[decay]['comb_a'] * self.totalAmplitudeSquared_low_Bs2DKPi(charge_flag) / self._normalisations[decay]['low_Bs2DKPi']
                else:
                    self._prob[decay]['low_Bs2DKPi'] = self.eff_arr[decay]['comb_abar'] * self.totalAmplitudeSquared_low_Bs2DKPi(charge_flag) / self._normalisations[decay]['low_Bs2DKPi']


    def make_prod_prob(self):
        """
        Calculate the producted negative log-likelihood.
        """
        for charge in ['p', 'm']:
                decay = self._name + '_' + charge
                self._prod_prob[decay] = {}
                self._ret[decay] = 0
                for keys in self.mass_pdfs[decay].keys():
                    if np.sum(self.mass_pdfs[decay][keys]) >= 1:
                        self._prod_prob[decay][keys] = self.mass_pdfs[decay][keys] * self._prob[decay][keys]
                        self._ret[decay] += self.mass_pdfs[decay][keys] * self._prob[decay][keys]
                    #else:
                    #    print(f'Mass PDF {keys} is empty for decay {decay}')

    def nll(self):
        """
        Calculate the negative log-likelihood.
        """
        for charge in ['p', 'm']:
            decay = self._name + '_' + charge
            self._nll[decay] = tf.reduce_sum(-2 * clip_log(self._ret[decay]))

    def totalAmplitudeSquared_XY(self, Bsign=1):
    
        name = self._name+'_p' if Bsign == 1 else self._name+'_m'
        phase = self._phase[name]
        absA, absAbar = tf.cast(tf.abs(self._absA[name]), tf.float64), tf.cast(tf.abs(self._absAbar[name]), tf.float64)

        if Bsign == 1:
            xPlus, yPlus = self.params[0], self.params[1]
            rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)
    
            return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus * tf.cos(phase) - yPlus * tf.sin(phase)))
        
        elif Bsign == -1:
            xMinus, yMinus = self.params[2], self.params[3]
            rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)
    
            return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus * tf.cos(phase) + yMinus * tf.sin(phase)))
    
    def totalAmplitudeSquared_low(self, Bsign=1):
        """
        Calculate the amplitude squared for the low mass contribution.

        Args:
        Bsign: int: 1 or -1
        norm: float: normalisation factor for the amplitude squared

        Returns:
        float: probability for the low mass contribution
        """
        name = self._name+'_p' if Bsign == 1 else self._name+'_m'#If Bsign is -1, then for the low_Bs2DKPi, it will take the amplitude from opposite charge B, which is wrong
        absA, absAbar = tf.cast(tf.abs(self._absA[name]), tf.float64), tf.cast(tf.abs(self._absAbar[name]), tf.float64)

    
        if Bsign ==1:
            return absAbar**2
        elif Bsign == -1:
            return absA**2

    def totalAmplitudeSquared_low_Bs2DKPi(self, Bsign=1):
        """
        Calculate the amplitude squared for the low mass contribution.

        Args:
        Bsign: int: 1 or -1
        norm: float: normalisation factor for the amplitude squared

        Returns:
        float: probability for the low mass contribution
        """
        name = self._name+'_p' if Bsign == 1 else self._name+'_m'
        absA, absAbar = tf.cast(tf.abs(self._absA[name]), tf.float64), tf.cast(tf.abs(self._absAbar[name]), tf.float64)

    
        if Bsign ==1:
            return absA**2
        elif Bsign == -1:
            return absAbar**2

    def totalAmplitudeSquared_DPi_XY(self, Bsign=1):

        name = self._name+'_p' if Bsign == 1 else self._name+'_m'
        phase = self._phase[name]
        absA, absAbar = tf.cast(tf.abs(self._absA[name]), tf.float64), tf.cast(tf.abs(self._absAbar[name]), tf.float64)

        xXi, yXi = self.params[4], self.params[5]
        if Bsign == 1:
            xPlus, yPlus = self.params[0], self.params[1]
            xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
            yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)
            rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)
    
            return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus_DPi * tf.cos(phase) - yPlus_DPi * tf.sin(phase)))#/self._normalisations[name]['misid']
        
        elif Bsign == -1:


            xMinus, yMinus = self.params[2], self.params[3]
            xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
            yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)
            rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)
    
    
            return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus_DPi * tf.cos(phase) + yMinus_DPi * tf.sin(phase)))#/self._normalisations[name]['misid']
        
    def totalAmplitudeSquared_comb(self, Bsign = 1):

        name = self._name+'_p' if Bsign == 1 else self._name+'_m'
        fracDD = self.fracDD[name]
        absA, absAbar = tf.cast(tf.abs(self._absA[name]), tf.float64), tf.cast(tf.abs(self._absAbar[name]), tf.float64)
        normA, normAbar = self._normalisations[name]['comb_a'], self._normalisations[name]['comb_abar']
        eff1, eff2 = self.eff_arr[name]['comb_a'], self.eff_arr[name]['comb_abar']

        frac1 = fracDD*0.5
        frac2 = 1.0 - fracDD
        prob1 = eff1*absA**2 /normA
        prob2 = eff2*absAbar**2 /normAbar 
        prob3 = tf.ones_like(absA)    
    
        return (prob1 * frac1 + prob2 * frac1 + prob3 * frac2)

#Isolate the BESIII code first
class DecayNLL_Charm:
    """
    Class to calculate negative log-likelihood for BESIII decay types.
    """
    def __init__(self, **kwargs):
        self._nll = {}
        self.type = kwargs.get('type')
        self.sig_model = None
        self.model = self.formula()


    def formula(self):
        #"""Decay rate formula"""

        if self.type[:4] == 'flav':
            print('flavour')
            return  self.flavour
        elif self.type == 'cp_even' or self.type == 'cp_odd':
            print('cp_even')
            return  self.cp_tag
        elif self.type == 'cp_mixed':
            print('cp_mixed')
            return  self.cp_mixed
        else:
            print('Invalid type')

    def sig_model(self):
        """
        Signal model for the decay.
        """
        _params = model.trainable_variables


        return self.sig_model
    
    def nll_cfit(self):
        """
        Calculate the negative log-likelihood with the cfit method.

        .. math:: 
            P_{i} = (1 - f_{bkg}) * P_{sig} + f_{bkg}(q_{i}) * P_{bkg}(q_{i})

        .. math::
            P_{sig} = \\frac{\epsilon |A|^{2}(q_{i})}{\\sum \epsilon |A|^{2}(q_{i})dq_{i}}
            
        .. math::
            -2 \ln L = -2 \sum_{i} \ln P_{i}
            
    
        """

        #Load from config_loader
        prob_bkg = []
        w_bkg = []


        ln_data = tf.math.log(
            (1 - w_bkg) * self.sig_model + w_bkg * prob_bkg
        )

        nll = -2 * tf.reduce_sum(ln_data)

        return nll




