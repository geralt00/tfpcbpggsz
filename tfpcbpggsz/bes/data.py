import numpy as np
import uproot as up
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
        #print(self.config.idx)
        #path = self.config._config_data['data'].get(idx)

        if len(self.config.idx) == len(self.config._config_data['data'].get(idx)):
            for tag in self.config.idx.keys():
                self.data_path[idx] = {}
                for i_file in self.config.idx.values():
                    self.data_path[idx][tag] = self.config._config_data['data'].get(idx)[i_file]
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
        
        #print(self.get_data_path(idx))
        for tag in self.config.idx.keys():
            self.data[tag] = {}
            path = self.data_path[idx][tag]
            if path.endswith('.root'):
                self.data[tag] = data_io(root_data(path, self.config._config_data['data'].get('tree')).load_tuple()).load_all()
            elif path.endswith('.npy'):
                prob = np.load(path)
                self.data[tag][idx] = prob
                
                #self.data[tag] = up.open(path).get(self.config._config_data['data'].get('tree')).arrays(library='np')
        
        return self.data
