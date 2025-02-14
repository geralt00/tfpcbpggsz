import uproot as up
import numpy as np
import time
from multiprocessing import Pool
import multiprocessing
import sys

from tfpcbpggsz.amp_up.D0ToKSpipi2018 import *


class GetCiSi:
    def __init__(self,D0ToKSpipi2018,**kwargs):
        self.D0ToKSpipi2018 = D0ToKSpipi2018
        self.kwargs = kwargs
        self.binning_file = kwargs.get('binning_file',None)



    def read_binning(self):
        """
        Reads data from a txt file using the map function.

        Args:
          filename: The name of the txt file.

        Returns:
          A NumPy array containing the data.
        """

        with open(self.binning_file, 'r') as f:
          data = list(map(lambda line: np.array(line.strip().split(' ')), f))
        return np.array(data)


    
    def find_bin_numpy(self, event):
        """
        Finds the bin index on the Dalitz plane using NumPy.

        Args:
            event: The event object with s(i, j) method for calculating invariant mass.
            s01_list: NumPy array of s01 values for the binning scheme.
            s02_list: NumPy array of s02 values for the binning scheme.
            binList: List of bin indices corresponding to the binning scheme.

        Returns:
            The bin index.
        """

        s01 = event['s12']
        s02 = event['s13']
        s01_list, s02_list, binList = self.read_binning()[:,0], self.read_binning()[:,1], self.read_binning()[:,2]
        binList = binList.astype(float)

        # Calculate squared distances using NumPy broadcasting
        s01_list = np.array(s01_list)
        s02_list = np.array(s02_list)

        # Broadcasting to get distances:
        s01_expanded = s01[:, np.newaxis]  # Add a new axis to s01
        s02_expanded = s02[:, np.newaxis] 
        chunk_size = 1000  # Adjust this based on your memory constraints
        min_index = []

        for i in range(0, len(s01), chunk_size):
            chunk_s01 = s01_expanded[i:i + chunk_size]
            chunk_s02 = s02_expanded[i:i + chunk_size]

            distances = (s01_list - chunk_s01)**2 + (s02_list - chunk_s02)**2
            chunk_min_index = np.argmin(distances, axis=1)
            min_index.append(chunk_min_index)

        min_index = np.concatenate(min_index)

        bin = binList[min_index]
        bin = np.where(s01 < s02, bin, np.negative(bin))
        return bin
    
    def get_cisi(self, event):
        """
        Gets the Ci and Si values for an event.

        Args:
            event: The event object with s(i, j) method for calculating invariant mass.

        Returns:
            A tuple containing the Ci and Si values.
        """

        bin_list = self.find_bin_numpy(event)
        phase = event['model_phase']
        absAmp = np.abs(event['amp'])
        absAmpbar = np.abs(event['ampbar'])

        unique_bins = np.unique(bin_list)
        #Ci = np.zeros_like(unique_bins, dtype=float)
        #Si = np.zeros_like(unique_bins, dtype=float)
        Ci = {}
        Si = {}

        for i, bin_index in enumerate(unique_bins):
            mask = (bin_list == bin_index)  # Boolean mask for events in the current bin
            Ci[f'{bin_index}'] = np.sum(absAmp[mask] * absAmpbar[mask] * np.cos(phase[mask]))/np.sqrt(np.sum(absAmp[mask]**2) * np.sum(absAmpbar[mask]**2))
            Si[f'{bin_index}'] = np.sum(absAmp[mask] * absAmpbar[mask] * np.sin(phase[mask]))/np.sqrt(np.sum(absAmp[mask]**2) * np.sum(absAmpbar[mask]**2))

        return Ci, Si, unique_bins



#Kspipi = PyD0ToKSpipi2018()
#Kspipi.init()


