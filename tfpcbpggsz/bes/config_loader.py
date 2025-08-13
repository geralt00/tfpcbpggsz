import yaml
import numpy as np
from tfpcbpggsz.core import Normalisation
from tfpcbpggsz.variable import VarsManager
from tfpcbpggsz.amp.amplitude import Amplitude

from tfpcbpggsz.bes.yields import yields, D02KsPiPi
from tfpcbpggsz.bes.data import load_data

class ConfigLoader:
    """
    Class for loading data/mc with the configuration file yaml

    """
    def __init__(self, config_file):

        self.file_path = config_file
        self._config_data = None
        self._amp = {}
        self._ampbar = {}
        self.norm = Normalisation
        self.vm = VarsManager()
        self.load_config()
        self.amp = Amplitude(self._config_data.get('model'))
        self.amp.init()
        self.idx = {}
        self.data = load_data(self)
        self.data.amp = self.amp
        self._data = {}
        self._mc = {}
        self._pdf = {}
        self._bkg_frac = {}
        self._yields = {}
        self._vary = self._config_data.get('vary', False)
        if self._vary:
            self.random_seed = self._config_data.get('random_seed', 100)
            self.random = np.random.default_rng(self.random_seed)
            print(f"INFO:: Varying the yields with random seed: {self.random_seed} \n")



        self.D02KsPiPi = D02KsPiPi()
        self.yields = yields(self.D02KsPiPi)
        self.mass_fit_results = {}
        self.new_mass_fit_results = {}

    def load_config(self):

        if isinstance(self.file_path, str):
            with open(self.file_path) as f:
                self._config_data = yaml.load(f, Loader=yaml.FullLoader)

            return self._config_data
        
    def get(self, key):
        return self._config_data.get(key)
    
    def get_order(self):
        """Get the order of the data in the configuration file

        Returns:
            dic: dictionary with the order of the data with tag name and index
        """
        for key in self._config_data['data'].get('tag_list'):
            self.idx[key] = self._config_data['data'].get('tag_list').index(key)
        return self.idx
    
    def get_all_data(self):
        datafile = ['data', 'phsp', 'pdf', 'qcmc', 'dpdm', 'qqbar', 'sigmc_um', 'qcmc_oth']
        self.get_order()
        if 'qcmc' in self._config_data['data'].keys():
            self._data, self._mc['phsp'], self._pdf, self._mc['qcmc'], self._mc['dpdm'], self._mc['qqbar'], self._mc['sigmc_um'], self._mc['qcmc_oth'] = [self.data.get_data(i) for i in datafile]
        else:
            datafile = ['data', 'phsp']
            self._data, self._mc['phsp'] = [self.data.get_data(i) for i in datafile]
        return self._data, self._mc, self._pdf

    def get_data(self, type):
        self.get_order()
        self._data = self.data.get_data(type)
        return self._data

    def get_data_srd(self, tag, key=None):
        if isinstance(self._data[tag]['srd'], dict):
            return self._data[tag]['srd'][key]
        else:
            return self._data[tag]['srd']
    
    def get_data_mass(self, tag, key=None):
        if isinstance(self._data[tag]['s12'], dict):
            return self._data[tag]['s12'][key], self._data[tag]['s13'][key]
        else:
            return self._data[tag]['s12'], self._data[tag]['s13']
    
    def get_data_amp(self, tag, key=None):
        if isinstance(self._data[tag]['amp'], dict):
            return self._data[tag]['amp'][key]
        else:
            return self._data[tag]['amp']

    def get_data_ampbar(self, tag, key=None):
        if isinstance(self._data[tag]['ampbar'], dict):
            return self._data[tag]['ampbar'][key]
        else:
            return self._data[tag]['ampbar']
        
    def get_mc_mass(self, tag, key, key_tag=None):
        #print(f"INFO:: Getting MC mass for tag: {tag}, key: {key}, key_tag: {key_tag}")
        if isinstance(self._mc[key][tag]['s12'], dict):
            return self._mc[key][tag]['s12'][key_tag], self._mc[key][tag]['s13'][key_tag]
        else:
            return self._mc[key][tag]['s12'], self._mc[key][tag]['s13']
        
    def get_phsp_srd(self, tag, key=None):
        if isinstance(self._mc['phsp'][tag]['srd'], dict):
            return self._mc['phsp'][tag]['srd'][key]
        else:
            return self._mc['phsp'][tag]['srd']
    
    def get_phsp_mass(self, tag, key=None):
        if isinstance(self._mc['phsp'][tag]['srd'], dict):
            return self._mc['phsp'][tag]['s12'][key], self._mc['phsp'][tag]['s13'][key]
        else:    
            return self._mc['phsp'][tag]['s12'], self._mc['phsp'][tag]['s13']
    
    
    def get_phsp_amp(self, tag, key=None):
        if isinstance(self._mc['phsp'][tag]['amp'], dict):
            return self._mc['phsp'][tag]['amp'][key]
        else:
            return self._mc['phsp'][tag]['amp']
        
    def get_phsp_ampbar(self, tag, key=None):
        if isinstance(self._mc['phsp'][tag]['ampbar'], dict):
            return self._mc['phsp'][tag]['ampbar'][key]
        else:
            return self._mc['phsp'][tag]['ampbar']
        
    def get_data_bkg(self, tag):
        """Get the probability of the background

        Args:
            ret (float64): the probability of the background in shape (n,)
        """
        ret = None

        if not self._pdf:
            ret = np.array([0.0], dtype=np.float64)
            
        else:
            ret = np.array([self._pdf[tag][key] for key in self._pdf[tag].keys()])
        return ret


    def get_bkg_frac(self):

        for tag in self._pdf.keys():
            self._bkg_frac[tag] = {}
            self.get_tag_bkg_frac(tag)

        return self._bkg_frac
            
    def get_tag_bkg_frac(self, tag):


        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        ntot = 0
        nsig = 0
        nbkg = 0
        self._yields[tag] = {} if tag not in self._yields else self._yields[tag]
        for key in self.mass_fit_results.keys():
            if 'sig_range_nb_' in key:
                if key.split('sig_range_nb_')[-1] in self._yields[tag].keys():
                    nbkg += self._yields[tag][key.split('sig_range_nb_')[-1]]
                else:
                    nbkg += self.get_bkg_num(tag, key)
            if 'sig_range_nsig' in key:
                if 'sig' in self._yields[tag].keys():
                    nsig += self._yields[tag]['sig']
                else:
                    nsig += self.get_sig_num(tag)
        ntot = nbkg + nsig
        #Looping over the pdfs making sure the frac matrix matchs pdfs
        ret = np.array([self._yields[tag][key] for key in self._pdf[tag].keys()])/ntot
        ret = ret.reshape(-1,1)
        self._bkg_frac[tag]['total'] = ret
        for key in self._pdf[tag].keys():
            self._bkg_frac[tag][key] = self._yields[tag][key]/ntot

    def reset_yield(self, tag, key, num=0):
        self._yields[tag][key] = num
        
        print(f"INFO:: Reset {key} yield for {tag} to {num}")

    def get_bkg_num(self, tag, key, default=0):
        """Get the number of the background
        Args:
            tag (str): the tag name
            key (str): the key name
            default (int): the default value if not found
            vary (bool): if True, return the value with the error
        """
            
        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        self.mass_fit_errors = self.yields.get(type='fit_result')['error']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        if key not in self.mass_fit_results.keys():
            print(f'INFO:: {key} not found in mass_fit_results of {tag}')
            return default
        else:
            val = self.re_sample_yields(self.mass_fit_results[key], self.mass_fit_errors[key]) if self._vary else self.mass_fit_results[key]
            self._yields[tag][key.split('sig_range_nb_')[-1]] = val
            return val
    
    def get_sig_num(self, tag):
        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        self.mass_fit_errors = self.yields.get(type='fit_result')['error']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        val = self.re_sample_yields(self.mass_fit_results[f'sig_range_nsig'],self.mass_fit_errors[f'sig_range_nsig']) if self._vary else self.mass_fit_results[f'sig_range_nsig']
        self._yields[tag]['sig'] = val
        return val
    
    
    def re_sample_yields(self, mean, error):
        """Resample the yields for the given tag
        Args:
            mean (float): the mean value for the resampling
            error (float): the error value for the resampling
        """
        #If error is 0.0, which is fixed, then do a possion
        if error == 0.0:
            return self.random.poisson(mean)
        return self.random.normal(mean, error)
        #new_yields = np.random.multivariate_normal(old_yields, covariance)
        #for i, key in enumerate(name_order):
        #    self.new_mass_fit_results[key] = new_yields[i]