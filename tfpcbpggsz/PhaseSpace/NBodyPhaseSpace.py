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

from tfpcbpggsz.Kinematics import PHSPGenerator
from tfpcbpggsz.Interface import *
import tensorflow as tf
import numpy as np
import math

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


class NBodyPhaseSpace:
    """
    Class for N-body decay expressed as:
      m_mother   : mass of the mother
      m_daughs   : list of daughter masses
    """

    def __init__(self, m_mother, m_daughs):
        """
          Constructor
        """
        self.ndaughters = len(m_daughs)

        self.PHSPGenerator = PHSPGenerator(m_mother, m_daughs)
        self.nev_ph = tf.placeholder(tf.int32)
        self.majorant_ph = tf.placeholder(tf.float64)
        self.phsp_model = self.PHSPGenerator.GenerateModel(self.nev_ph)
        self.phsp_model_majorant = tf.concat([self.phsp_model, Scalar(tf.random_uniform(
            [self.nev_ph], minval=0., maxval=self.majorant_ph, dtype=FPType()))], axis=1)

        self.data_placeholder = self.Placeholder("data")
        self.norm_placeholder = self.Placeholder("norm")

    def Filter(self, x):
        return x

    def Density(self, x):
        return tf.transpose(x)[4*self.ndaughters]

    def UnfilteredSample(self, size, majorant=-1):
        """
          Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
          Note it does not actually generate the sample, but returns the data flow graph for generation,
          which has to be run within TF session.
        """
        s = tf.Session()
        feed_dict = {self.nev_ph: size}
        if majorant > 0:
            feed_dict.update({self.majorant_ph: majorant})
        uniform_sample = s.run(self.phsp_model_majorant if majorant > 0 else self.phsp_model,
                               feed_dict=feed_dict)
        s.close()
        return uniform_sample

    def FinalStateMomenta(self, x):
        """
           Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
           defined by the phase space vector x. The momenta are calculated in the
           D rest frame.
        """
        p3s = [Vector(tf.transpose(x)[4*i], tf.transpose(x)[4*i+1],
                      tf.transpose(x)[4*i+2]) for i in range(self.ndaughters)]
        pLs = [LorentzVector(p3, tf.transpose(x)[4*i+3])
               for p3, i in zip(p3s, range(self.ndaughters))]
        return tuple(pL for pL in pLs)

    def Placeholder(self, name=None):
        return tf.placeholder(FPType(), shape=(None, None), name=name)
