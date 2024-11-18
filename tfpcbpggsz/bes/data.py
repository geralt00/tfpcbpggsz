import numpy as np
import warnings
from tfpcbpggsz.bes.data_root import root_data
from tfpcbpggsz.data_io import data_io


class load_data:
    """The class to load the data from the root or npz file
    """
    def __init__(self, config_data):
        self.data ={}
        self.config = config_data

        self.data_path = {}


    def get_data_path(self, idx):
        """Get the data path from the configuration file

        Args:
            idx (int): index of the data in the configuration file

        Returns:
            dic: dictionary with the data path
        """

        if (idx != 'pdf' and len(self.config.idx) == len(self.config._config_data['data'].get(idx))) or idx == 'pdf':
            for tag in self.config.idx.keys():
                self.data_path[idx] = {} if idx not in self.data_path.keys() else self.data_path[idx]
                if idx != 'pdf':
                    #for i_file in self.config.idx.values():
                    self.data_path[idx][tag] = self.config._config_data['data'].get(idx)[self.config.idx.get(tag)]
                    #print(f'{tag}: {self.data_path[idx][tag]}')
                else:
                    self.data_path[idx][tag]={} #if tag not in self.data_path[idx].keys() else self.data_path[idx][tag]
                    for i_pdf in self.config._config_data['data'].get(idx).keys():
                        if isinstance(self.config._config_data['data'].get(idx).get(i_pdf), list):
                            self.data_path[idx][tag][i_pdf] = self.config._config_data['data'].get(idx)[i_pdf][self.config.idx.get(tag)]
                            #print(f'{tag}: {self.data_path[idx][tag][i_pdf]}')
                        elif tag in self.config._config_data['data'].get(idx).get(i_pdf).keys():
                            self.data_path[idx][tag][i_pdf] = self.config._config_data['data'].get(idx).get(i_pdf)[self.config.idx.get(tag)]
                            #print(f'{tag}: {self.data_path[idx][tag][i_pdf]}')
        else:
            warnings.warn("The number of tags is not equal to the number of files")


        return self.data_path

    def get_data(self, idx):
        """Get the data from the root file

        Args:
            idx (int): index of the data in the configuration file
            if file is root, we expect the data following the structure of the momentum
            in this case, we would like to get the invariant mass as well as the amplitude
            if file is npy, we expect is provided the probability values

        Returns:
            dic: dictionary with the data
        """
        self.get_data_path(idx)
        for tag in self.config.idx.keys():
            self.data[tag] = {} if tag not in self.data.keys() else self.data[tag]
            cuts=self.config.D02KsPiPi.cuts(tag)
            if idx != 'pdf':
                path = self.data_path[idx][tag]
                #if idx == 'qcmc':
                #    cuts = cuts + ' & ' + self.config.D02KsPiPi.topo_cut(tag)
                if path.endswith('.root'):
                    if tag in ['full', 'misspi0', 'misspi']:
                        branches = ['p4_Ks','p4_pim','p4_pip','p4_Ks2','p4_pim2','p4_pip2']
                        self.data[tag][idx] = data_io(root_data(path, self.config._config_data['data'].get('tree'), cut=cuts, branches=branches).load_tuple()).load_all()
                    else:
                        self.data[tag][idx] = data_io(root_data(path, self.config._config_data['data'].get('tree'),cut=self.config.D02KsPiPi.cuts(tag)).load_tuple()).load_all()
                elif path.endswith('.npy'):
                    #prob = 
                    self.data[tag][idx] = np.load(path)   
            else:
                self.data[tag][idx] = {}
                for i_pdf in self.data_path[idx][tag].keys():
                    self.data[tag][idx][i_pdf] = {}
                    if isinstance(self.data_path[idx][tag][i_pdf], str):
                        #for i_pdf_tag in range(len(self.data_path[idx][tag][i_pdf])):
                        path = self.data_path[idx][tag][i_pdf]
                        if path.endswith('.root'):
                            self.data[tag][idx][i_pdf] = data_io(root_data(path, self.config._config_data['data'].get('tree')).load_tuple()).load_all()
                        elif path.endswith('.npy'):
                            self.data[tag][idx][i_pdf] = np.load(path)
                    else:
                        for i_ext_pdf in self.data_path[idx][tag][i_pdf].keys():
                            path = self.data_path[idx][tag][i_pdf][i_ext_pdf]
                            if path.endswith('.root'):
                                self.data[tag][idx][i_pdf][i_ext_pdf] = data_io(root_data(path, self.config._config_data['data'].get('tree')).load_tuple()).load_all()
                            elif path.endswith('.npy'):
                                self.data[tag][idx][i_pdf][i_ext_pdf] = np.load(path)
        #reform the data as for returning the data
        data = {}
        for tag in self.data.keys():
            data[tag] = {}
            if idx != 'pdf':
                data[tag] = self.data[tag][idx]
            else:
                for i_pdf in self.data[tag][idx].keys():
                    data[tag][i_pdf] = {}
                    if isinstance(self.data[tag][idx][i_pdf], np.ndarray):
                        #for i_pdf_tag in range(len(self.data[tag][idx][i_pdf])):
                        data[tag][i_pdf] = self.data[tag][idx][i_pdf]
                    else:
                        for i_pdf_tag in self.data[tag][idx][i_pdf].keys():
                            data[tag][i_pdf][i_pdf_tag] = self.data[tag][idx][i_pdf][i_pdf_tag]


        return data
