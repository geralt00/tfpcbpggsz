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


class FourBodyAngularPhaseSpace:
    """
    Class for angular phase space of 4-body X->(AB)(CD) decay (3D).
    """

    def __init__(self):
        """
        Constructor
        """
        self.data_placeholder = self.Placeholder("data")
        self.norm_placeholder = self.Placeholder("norm")

    def Inside(self, x):
        """
          Check if the point x=(cos_theta_1, cos_theta_2, phi) is inside the phase space
        """
        cos1 = self.CosTheta1(x)
        cos2 = self.CosTheta2(x)
        phi = self.Phi(x)

        inside = tf.logical_and(tf.logical_and(tf.greater(cos1, -1.), tf.less(cos1, 1.)),
                                tf.logical_and(tf.greater(cos2, -1.), tf.less(cos2, 1.)))
        inside = tf.logical_and(inside,
                                tf.logical_and(tf.greater(
                                    phi, 0.), tf.less(phi, 2.*math.pi))
                                )
        return inside

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
        v = [
            np.random.uniform(-1., 1., size).astype('d'),
            np.random.uniform(-1., 1., size).astype('d'),
            np.random.uniform(0., 2.*math.pi, size).astype('d')
        ]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.stack(v, axis = 1)

    def UnfilteredSampleGraph(self, size, majorant=-1):
        """
          Return TF graph for uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [
            tf.random.uniform([size], -1., 1., dtype = FPType()),
            tf.random.uniform([size], -1., 1., dtype = FPType()),
            tf.random.uniform([size], 0., 2.*math.pi, dtype = FPType())
        ]
        if majorant > 0:
            v += [tf.random.uniform([size], 0., majorant, dtype = FPType()) ]
        return tf.stack(v, axis = 1)

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

    def RectangularGridSample(self, size_cos_1, size_cos_2, size_phi):
        """
          Create a data sample in the form of rectangular grid of points within the phase space.
          Useful for normalisation.
        """
        size = size_cos_1*size_cos_2*size_phi
        mgrid = np.lib.index_tricks.nd_grid()
        v1 = mgrid[0:size_cos_1, 0:size_cos_2,
                   0:size_phi][0]*2./float(size_cos_1) - 1.
        v2 = mgrid[0:size_cos_1, 0:size_cos_2,
                   0:size_phi][1]*2./float(size_cos_2) - 1.
        v3 = mgrid[0:size_cos_1, 0:size_cos_2,
                   0:size_phi][2]*2.*math.pi/float(size_phi)
        v = [v1.reshape(size).astype('d'), v2.reshape(
            size).astype('d'), v3.reshape(size).astype('d')]
        x = tf.stack(v, axis=1)
        return tf.boolean_mask(x, self.Inside(x))

    def CosTheta1(self, sample):
        """
          Return CosTheta1 variable (vector) for the input sample
        """
        return sample[:, 0]

    def CosTheta2(self, sample):
        """
          Return CosTheta2 variable (vector) for the input sample
        """
        return sample[:, 1]

    def Phi(self, sample):
        """
          Return Phi variable (vector) for the input sample
        """
        return sample[:, 2]

    def Placeholder(self, name=None):
        return tf.placeholder(FPType(), shape=(None, None), name=name)

    def Dimensionality(self):
        return 3
