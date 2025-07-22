import numpy as np
import uproot as up
import warnings
from tfpcbpggsz.data_io import data_io

class root_data:
    def __init__(self, data_path, tree_name, **kwargs):
        self.data_path = data_path
        self.tree_name = tree_name
        self.data = None
        #self.load_data()
        self.is_reorder = True
        self.data_arr = None
        self.branches = ['p4_Ks','p4_pim','p4_pip']
        self.cut = None
        if 'branches' in kwargs:
            self.branches = kwargs['branches']
        if 'is_reorder' in kwargs:
            self.is_reorder = kwargs['is_reorder']
        if 'cut' in kwargs:
            self.cut = kwargs['cut']

    def load_tuple(self):
        """Load the tuple with the data

        Returns:
            list: list with the data
        """
        #Double Kspipi
        #data_kspipi = up.open(f"{self.data_path}:{self.tree_name}")
        branches = self.branches
        data_arr_old = self.load_arr()#data_kspipi.arrays(branches, cut=self.cut) if self.cut else data_kspipi.arrays(branches)
        data_arr = {}
        #Since the BESIII data was stored in order of (px,py,pz, E), we need to reorder it to (px,py,pz,E)
        for key in branches:
            if self.is_reorder:
                data_arr[key] = self.reorder_p4(data_arr_old[key])
            else:
                data_arr[key] = data_arr_old[key]

        self.data_arr = data_arr

        if len(branches) == 3:
            return [data_arr[branches[0]], data_arr[branches[1]], data_arr[branches[2]]]
        else:
            return [data_arr[branches[0]], data_arr[branches[1]], data_arr[branches[2]]], [data_arr[branches[3]], data_arr[branches[4]], data_arr[branches[5]]]
        
    def reorder_p4(self, data):
        return np.array([data[:,3],data[:,0],data[:,1],data[:,2]]).T
    
    def read_momentum(self, data, idx):
        return data[idx]

    def load_arr(self):
        """Load the data as an array

        Returns:
            np.array: array with the data
        """
        import glob
        if '*' in self.data_path:
            files = glob.glob(self.data_path)
            data_arr = {}
            for file in files:
                data_kspipi = up.open(f"{file}:{self.tree_name}")
                branches = self.branches
                data_arr_old = data_kspipi.arrays(branches, cut=self.cut) if self.cut else data_kspipi.arrays(branches)
                for key in branches:
                    if key in data_arr:
                        data_arr[key] = np.concatenate((data_arr[key],data_arr_old[key]))
                    else:
                        data_arr[key] = data_arr_old[key]
            return data_arr
        else:
            data_kspipi = up.open(f"{self.data_path}:{self.tree_name}")
            branches = self.branches
            data_arr = data_kspipi.arrays(branches, cut=self.cut) if self.cut else data_kspipi.arrays(branches)
            return data_arr


class load_data:
    """The class to load the data from the root or npz file
    """
    def __init__(self, config_data):
        self.data ={}
        self.amp = None
        self.config = config_data
        self.data_path = {}


    def get_data_path(self, idx):
        """Get the data path from the configuration file

        Args:
            idx (int): index of the data in the configuration file

        Returns:
            dic: dictionary with the data path
        """

        if (idx != 'pdf' and len(self.config.idx) == len(self.config._config_data['data'].get(idx))) or idx == 'pdf' or idx == 'sigmc_um' or idx == 'qcmc_oth':
            for tag in self.config.idx.keys():
                self.data_path[idx] = {} if idx not in self.data_path.keys() else self.data_path[idx]
                
                if idx != 'pdf':
                    if idx in ['sigmc_um', 'qcmc_oth'] and (tag in self.config._config_data['data'].get('sigmc_um').keys() or tag in self.config._config_data['data'].get('qcmc_oth').keys()):
                        self.data_path[idx] = {} if idx not in self.data_path.keys() else self.data_path[idx]
                        if tag in self.config._config_data['data'].get(idx).keys():
                            self.data_path[idx][tag] = self.config._config_data['data'].get(idx).get(tag)
                        #self.data_path[idx][tag] = self.config._config_data['data'].get(idx).get(tag)
                        #print(f'{tag}: {self.data_path[idx][tag]}')
                        #print(f'{tag}: {self.data_path[idx][tag]}')
                    #for i_file in self.config.idx.values():
                    elif idx != 'sigmc_um' and idx != 'qcmc_oth':
                        self.data_path[idx][tag] = self.config._config_data['data'].get(idx)[self.config.idx.get(tag)]
                    #print(f'{tag}: {self.data_path[idx][tag]}')
                else:
                    self.data_path[idx][tag]={} #if tag not in self.data_path[idx].keys() else self.data_path[idx][tag]
                    for i_pdf in self.config._config_data['data'].get(idx).keys():
                        if isinstance(self.config._config_data['data'].get(idx).get(i_pdf), list):
                            self.data_path[idx][tag][i_pdf] = self.config._config_data['data'].get(idx)[i_pdf][self.config.idx.get(tag)]
                            #print(f'{tag}: {self.data_path[idx][tag][i_pdf]}')
                        elif tag in self.config._config_data['data'].get(idx).get(i_pdf).keys():
                            if tag in self.config._config_data['data'].get(idx).get(i_pdf).keys():
                                self.data_path[idx][tag][i_pdf] = self.config._config_data['data'].get(idx).get(i_pdf)[tag]
                            #self.data_path[idx][tag][i_pdf] = self.config._config_data['data'].get(idx).get(i_pdf)[self.config.idx.get(tag)]
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

        #Loading the data path
        self.get_data_path(idx)

        #It will be better to loop over the tags that the data is loaded
        for tag in self.data_path[idx].keys():
            #print(f"Loading data for {tag}, {idx}")
            self.data[tag] = {} if tag not in self.data.keys() else self.data[tag]
            cuts=self.config.D02KsPiPi.cuts(tag)
            if idx != 'pdf':
                path = self.data_path[idx][tag]
                #print(f"Path: {path}")
                if path.endswith('.root'):
                    if tag in ['full', 'misspi0', 'misspi']:
                        branches = ['p4_Ks','p4_pim','p4_pip','p4_Ks2','p4_pim2','p4_pip2']
                        self.data_io = data_io()
                        self.data_io.amp = self.amp
                        self.data[tag][idx] = self.data_io.load_all(root_data(path, self.config._config_data['data'].get('tree'), cut=cuts, branches=branches).load_tuple())
                        #print(f'Loading {tag} for {idx} alread: {self.data[tag][idx]}')
                    else:
                        self.data_io = data_io()
                        self.data_io.amp = self.amp
                        self.data[tag][idx] = self.data_io.load_all(root_data(path, self.config._config_data['data'].get('tree'),cut=self.config.D02KsPiPi.cuts(tag)).load_tuple())
                elif path.endswith('.npy'):
                    #prob = 
                    self.data[tag][idx] = np.load(path)   
            else :
                self.data[tag][idx] = {}
                for i_pdf in self.data_path[idx][tag].keys():
                    self.data[tag][idx][i_pdf] = {}
                    if isinstance(self.data_path[idx][tag][i_pdf], str):
                        #for i_pdf_tag in range(len(self.data_path[idx][tag][i_pdf])):
                        path = self.data_path[idx][tag][i_pdf]
                        if path.endswith('.root'):
                            self.data_io = data_io()
                            self.data_io.amp = self.amp
                            self.data[tag][idx][i_pdf] = self.data_io.load_all(root_data(path, self.config._config_data['data'].get('tree')).load_tuple())
                        elif path.endswith('.npy'):
                            self.data[tag][idx][i_pdf] = np.load(path)
                    else:
                        for i_ext_pdf in self.data_path[idx][tag][i_pdf].keys():
                            path = self.data_path[idx][tag][i_pdf][i_ext_pdf]
                            if path.endswith('.root'):
                                self.data_io = data_io()
                                self.data_io.amp = self.amp
                                self.data[tag][idx][i_pdf][i_ext_pdf] = self.data_io.load_all(root_data(path, self.config._config_data['data'].get('tree')).load_tuple())
                            elif path.endswith('.npy'):
                                self.data[tag][idx][i_pdf][i_ext_pdf] = np.load(path)




        #reform the data as for returning the data
        data = {}
        for tag in self.data.keys():
            data[tag] = {}
            for i_idx in self.data[tag].keys():
                if idx != i_idx: continue
                if i_idx != 'pdf':
                #print(self.data[tag])
                    data[tag] = self.data[tag][i_idx]
                else:
                    for i_pdf in self.data[tag][i_idx].keys():
                        data[tag][i_pdf] = {}
                        if isinstance(self.data[tag][i_idx][i_pdf], np.ndarray):
                            #for i_pdf_tag in range(len(self.data[tag][i_idx][i_pdf])):
                            data[tag][i_pdf] = self.data[tag][i_idx][i_pdf]
                        else:
                            for i_pdf_tag in self.data[tag][i_idx][i_pdf].keys():
                                data[tag][i_pdf][i_pdf_tag] = self.data[tag][i_idx][i_pdf][i_pdf_tag]
        '''
        for tag in self.data.keys():
            data[tag] = {}
            if idx != 'pdf':
                #print(self.data[tag])
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
        '''


        return data
