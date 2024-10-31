import yaml
import uproot as up
import numpy as np
import time
from importlib.machinery import SourceFileLoader
from tfpcbpggsz.core import Normalisation, VarsManager
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
        self.vm = VarsManager
        self.load_config()
        self.idx = {}
        self.data = load_data(self)


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
        datafile = ['data', 'phsp', 'pdf_qcmc', 'pdf_dpdm', 'pdf_qqbar']
        self.get_order()
        data, phsp, pdf_qcmc, pdf_dpdm, pdf_qqbar = [self.data.get_data(i) for i in datafile]
        return data, phsp, pdf_qcmc, pdf_dpdm, pdf_qqbar

    #def get_data
