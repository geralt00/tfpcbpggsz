import numpy as np
import uproot as up


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
