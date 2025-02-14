from tfpcbpggsz.tensorflow_wrapper import tf
import tfpcbpggsz.core as core
from tfpcbpggsz.phasecorrection import PhaseCorrection as pc
from tfpcbpggsz.core import Normalisation as normalisation


class BaseModel(object):
    def __init__(self, config):
        self.norm = {}
        self.config_loader = config
        self.pc = pc(vm=self.config_loader.vm)
        self.vm = self.pc.vm
        self.tags = self.config_loader.idx
        self.load_norm()
        self._nll = {}


    def load_norm(self):
        
        for tag in self.tags:
            if tag in ["full", "misspi", "misspi0"]:
                self.norm[tag] = normalisation({f'{tag}_sig': self.config_loader.get_phsp_amp(tag, 'sig'), f'{tag}_tag': self.config_loader.get_phsp_amp(tag, 'tag')}, {f'{tag}_sig': self.config_loader.get_phsp_ampbar(tag, 'sig'), f'{tag}_tag': self.config_loader.get_phsp_ampbar(tag, 'tag')}, f'{tag}_sig')
                
                self.norm[tag].initialise()                
            else:
                self.norm[tag] = normalisation({tag: self.config_loader.get_phsp_amp(tag)}, {tag: self.config_loader.get_phsp_ampbar(tag)}, tag)
                self.norm[tag].initialise()


    #@tf.function
    def NLL_Kspipi(self, tag):

        params = self.pc.coefficients.values()
        phase_correction_sig = self.pc.eval_corr(self.config_loader.get_data_srd(tag,'sig'))
        phase_correction_tag = self.pc.eval_corr(self.config_loader.get_data_srd(tag,'tag'))
        self.norm[tag].setParams(params)
        phase_correction_MC_sig = self.pc.eval_corr(self.config_loader.get_phsp_srd(tag,'sig'))
        phase_correction_MC_tag = self.pc.eval_corr(self.config_loader.get_phsp_srd(tag,'tag'))
        self.norm[tag].add_pc(phase_correction_MC_sig, pc_tag=phase_correction_MC_tag)
        self.norm[tag].Update_crossTerms()

        #need to be flexible with the function name
        prob = core.prob_totalAmplitudeSquared_CP_mix(self.config_loader.get_data_amp(tag,'sig'), self.config_loader.get_data_ampbar(tag,'sig'), self.config_loader.get_data_amp(tag,'tag'), self.config_loader.get_data_ampbar(tag,'tag'), phase_correction_sig, phase_correction_tag)
        norm = self.norm[tag]._crossTerms_complex
        prob_bkg = self.config_loader.get_data_bkg(tag)
        frac_bkg = self.config_loader.get_bkg_frac(tag)
        prob_bkg = (prob_bkg)*frac_bkg
        bkg_part = tf.reduce_sum(prob_bkg, axis=0)
        sig_part = (prob/norm)*(1.0-tf.reduce_sum(frac_bkg))


        nll = tf.reduce_sum(-2*core.clip_log( sig_part + bkg_part))

        return nll
    
    #@tf.function    
    def NLL_CP(self, tag, Dsign):

        params = self.pc.coefficients.values()
        phase_correction = self.pc.eval_corr(self.config_loader.get_data_srd(tag))
        self.norm[tag].setParams(params)
        phase_correction_MC = self.pc.eval_corr(self.config_loader.get_phsp_srd(tag))
        self.norm[tag].add_pc(phase_correction_MC)
        self.norm[tag].Update_crossTerms()

        if tag == 'pipipi0':
            prob = core.prob_totalAmplitudeSquared_CP_tag(Dsign,self.config_loader.get_data_amp(tag), self.config_loader.get_data_ampbar(tag), pc=phase_correction, Fplus=0.9406)
            norm = self.norm[tag].Integrated_CP_tag(Dsign, Fplus=0.9406)

        else:
            prob = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config_loader.get_data_amp(tag), self.config_loader.get_data_ampbar(tag), pc=phase_correction)
            norm = self.norm[tag].Integrated_CP_tag(Dsign)
        prob_bkg = self.config_loader.get_data_bkg(tag)
        frac_bkg = self.config_loader.get_bkg_frac(tag)
        prob_bkg = (prob_bkg)*frac_bkg
        bkg_part = tf.reduce_sum(prob_bkg, axis=0)
        sig_part = (prob/norm)*(1.0-tf.reduce_sum(frac_bkg))

        #test_term = frac_bkg*tf.ones_like(prob)
        #print(test_term.shape)
        #print(f"bkg_part: {bkg_part.shape}")
        nll = tf.reduce_sum(-2*tf.math.log( sig_part + bkg_part))

        return nll
    
    def NLL_selector(self, tag):
        if tag in ["full", "misspi", "misspi0"]:
            return self.NLL_Kspipi(tag)
        elif tag in ["kspi0", "kseta_gamgam", "ksetap_pipieta", "kseta_3pi", "ksetap_gamrho", "ksomega", "klpi0pi0"]:
            return self.NLL_CP(tag, -1)
        elif tag in ["kk", "pipi", "pipipi0", "kspi0pi0", "klpi0"]:
            return self.NLL_CP(tag, 1)

    @tf.function
    def nll_dks(self):
        #nll = []
        ret = 0
        for tag in self.tags:
            if tag not in ["full", "misspi", "misspi0"]: continue
            self._nll[tag] = self.NLL_selector(tag)
            ret += self._nll[tag]
        return ret
    
    @tf.function
    def nll_cpeven(self):
        #nll = []
        ret = 0
        for tag in self.tags:
            if tag not in ["kk", "pipi", "pipipi0", "kspi0pi0", "klpi0"]: continue
            self._nll[tag] = self.NLL_selector(tag)
            ret += self._nll[tag]
        return ret
    
    @tf.function
    def nll_cpodd(self):
        #nll = []
        ret = 0
        for tag in self.tags:
            if tag not in ["kspi0", "kseta_gamgam", "ksetap_pipieta", "kseta_3pi", "ksetap_gamrho", "ksomega", "klpi0pi0"]: continue
            self._nll[tag] = self.NLL_selector(tag)
            ret += self._nll[tag]
        
        return ret
    
    @tf.function
    def fun(self, x):
        self.set_params(x)
        ret = self.nll_dks() + self.nll_cpeven() + self.nll_cpodd()

        return ret
        
    def set_params(self, x={}):
        self.pc.set_coefficients(coefficients=x)
