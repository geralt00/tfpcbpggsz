import yaml
import uproot as up
import numpy as np
import time
from importlib.machinery import SourceFileLoader
from tfpcbpggsz.core import Normalisation
from tfpcbpggsz.variable import VarsManager
from .yields import yields, D02KsPiPi
from tfpcbpggsz.amp_up import D0ToKSpipi2018
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
        self.idx = {}
        self.data = load_data(self)
        self._data = {}
        self._mc = {}
        self._pdf = {}
        '''
        self._mc['phsp'] = {}
        self._qcmc = {}
        self._qcmc_oth = {}
        self._dpdm = {}
        self._qqbar = {}
        self._sig_um = {}
        '''
        #It will be easier to use MC as a big dictionary

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
        
        self._data, self._mc['phsp'], self._pdf, self._mc['qcmc'], self._mc['dpdm'], self._mc['qqbar'], self._mc['sigmc_um'], self._mc['qcmc_oth'] = [self.data.get_data(i) for i in datafile]
        return self._data, self._mc, self._pdf

    def get_data(self, type):
        self.get_order()
        self._data = self.data.get_data(type)
        return self._data

    #def get_data

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
        ret = np.array([self._pdf[tag][key] for key in self._pdf[tag].keys()])
        return ret
    
    def get_bkg_frac(self, tag, **kwargs):


        vary = False
        if 'vary' in kwargs.keys():
            vary = kwargs['vary']
        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        ntot = 0
        nsig = 0
        nbkg = 0
        for key in self.mass_fit_results.keys():
            if 'sig_range_nb_' in key:
                nbkg += self.get_bkg_num(tag, key, vary=vary) if vary else self.get_bkg_num(tag, key)
            if 'sig_range_nsig' in key:
                nsig += self.get_sig_num(tag, vary=vary) if vary else self.get_sig_num(tag)
        
        ntot = nbkg + nsig
        ret = np.array([self.get_bkg_num(tag, key, vary=vary) for key in self._pdf[tag].keys()])/ntot
        #print([f'sig_range_nb_{key}' for key in self._pdf[tag].keys()])
        #print(f"INFO:: {tag} bkg fraction: {ret}")
        return ret.reshape(-1,1)

    def get_bkg_num(self, tag, key, default=0, vary=False):
        """Get the number of the background
        Args:
            tag (str): the tag name
            key (str): the key name
            default (int): the default value if not found
            vary (bool): if True, return the value with the error
        """
            
        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        if f'sig_range_nb_{key}' not in self.mass_fit_results.keys():
            print(f'INFO:: {key} not found in mass_fit_results of {tag}')
            return default
        else:
            val = self.new_mass_fit_results[f'sig_range_nb_{key}'] if vary else self.mass_fit_results[f'sig_range_nb_{key}']
            return val
    
    def get_sig_num(self, tag, vary=False):
        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        val = self.new_mass_fit_results[f'sig_range_nsig'] if vary else self.mass_fit_results[f'sig_range_nsig']
        return val
    
    def re_sample_yields(self, tag):
        """Resample the yields for the given tag
        Args:
            tag (str): the tag name
            kwargs: the keyword arguments for the resampling
        """
        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        covariance = self.yields.get(type='fit_result')['cov']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        name_order = self.yields.get(type='fit_result')['cov_name']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        old_yields = np.array([self.mass_fit_results[key] for key in name_order])
        new_yields = np.random.multivariate_normal(old_yields, covariance)
        for i, key in enumerate(name_order):
            self.new_mass_fit_results[key] = new_yields[i]