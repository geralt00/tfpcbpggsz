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
        self._phsp = {}
        self._pdf = {}
        self.D02KsPiPi = D02KsPiPi()
        self.yields = yields(self.D02KsPiPi)
        self.mass_fit_results = {}


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
        datafile = ['data', 'phsp', 'pdf']
        self.get_order()

        self._data, self._phsp, self._pdf = [self.data.get_data(i) for i in datafile]
        return self._data, self._phsp, self._pdf

    def get_data(self, type):
        self.get_order()
        self._data = self.data.get_data(type)
        return self._data

    #def get_data

    def get_data_srd(self, tag):
        return self._data[tag]['srd']
    
    def get_data_mass(self, tag):
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
        
    def get_phsp_srd(self, tag):
        return self._phsp[tag]['srd']
    
    def get_phsp_mass(self, tag):
        return self._phsp[tag]['s12'], self._phsp[tag]['s13']
    
    def get_phsp_amp(self, tag, key=None):
        if isinstance(self._phsp[tag]['amp'], dict):
            return self._phsp[tag]['amp'][key]
        else:
            return self._phsp[tag]['amp']
        
    def get_phsp_ampbar(self, tag, key=None):
        if isinstance(self._phsp[tag]['ampbar'], dict):
            return self._phsp[tag]['ampbar'][key]
        else:
            return self._phsp[tag]['ampbar']
        
    def get_data_bkg(self, tag):
        """Get the probability of the background

        Args:
            ret (float64): the probability of the background in shape (n,)
        """
        ret = np.array([self._pdf[tag][key] for key in self._pdf[tag].keys()])
        return ret
    
    def get_bkg_frac(self, tag):

        self.yields.load(self._config_data['data'].get('mass_fit_results'))
        self.mass_fit_results = self.yields.get(type='fit_result')['mean']['all'][self.D02KsPiPi.catogery(tag=tag)][tag]
        ntot = 0
        for key in self.mass_fit_results.keys():
            if 'sig_range_n' in key:
                ntot += self.mass_fit_results[key]
        ret = np.array([self.mass_fit_results[f'sig_range_nb_{key}'] for key in self._pdf[tag].keys()])/ntot
        #shape as (n,1)
        return ret.reshape(-1,1)