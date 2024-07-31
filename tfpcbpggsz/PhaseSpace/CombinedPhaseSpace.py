# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tfpcbpggsz.Interface import *
import tensorflow as tf
import numpy as np
import math

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


class CombinedPhaseSpace:
    """
      Combination (direct product) of two phase spaces
    """

    def __init__(self, phsp1, phsp2):
        self.phsp1 = phsp1
        self.phsp2 = phsp2
        self.data_placeholder = self.Placeholder("data")
        self.norm_placeholder = self.Placeholder("norm")

    def Dimensionality(self):
        return self.phsp1.Dimensionality() + self.phsp2.Dimensionality()

    def Data1(self, x):
        return tf.slice(x, [0, 0], [-1, self.phsp1.Dimensionality()])

    def Data2(self, x):
        return tf.slice(x, [0, self.phsp1.Dimensionality()], [-1, self.phsp2.Dimensionality()])

    def Inside(self, x):
        return tf.logical_and(self.phsp1.Inside(self.Data1(x)), self.phsp2.Inside(self.Data2(x)))

    def Filter(self, x):
        return tf.boolean_mask(x, self.Inside(x))

    def UnfilteredSample(self, size, majorant=-1):
        """
          Generate uniform sample of point within phase space. 
            size     : number of _initial_ points to generate. Not all of them will fall into phase space, 
                       so the number of points in the output will be <size. 
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is 
                       uniform number from 0 to majorant. Useful for accept-reject toy MC. 
        """
        sample1 = self.phsp1.UnfilteredSample(size)
        sample2 = self.phsp2.UnfilteredSample(size, majorant)
        return np.concatenate((sample1, sample2), axis=1)

    def UnfilteredSampleGraph(self, size, majorant=-1):
        """
          Return TF graph for uniform sample of point within phase space. 
            size     : number of _initial_ points to generate. Not all of them will fall into phase space, 
                       so the number of points in the output will be <size. 
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is 
                       uniform number from 0 to majorant. Useful for accept-reject toy MC. 
        """
        sample1 = self.phsp1.UnfilteredSampleGraph(size)
        sample2 = self.phsp2.UnfilteredSampleGraph(size, majorant)
        return tf.concat((sample1, sample2), axis=1)

    def UniformSample(self, size, majorant=-1):
        """
          Generate uniform sample of point within phase space. 
            size     : number of _initial_ points to generate. Not all of them will fall into phase space, 
                       so the number of points in the output will be <size. 
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is 
                       uniform number from 0 to majorant. Useful for accept-reject toy MC. 
          Note it does not actually generate the sample, but returns the data flow graph for generation, 
          which has to be run within TF session. 
        """
        return self.Filter(self.UnfilteredSample(size, majorant))

    def Placeholder(self, name=None):
        """
          Create a placeholder for a dataset in this phase space 
        """
        return tf.placeholder(FPType(), shape=(None, None), name=name)

    def Bounds(self):
        return self.phsp1.Bounds() + self.phsp2.Bounds()
