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

from tfpcbpggsz.PhaseSpace.DalitzPhaseSpace import *
from tfpcbpggsz.Kinematics import *
from tfpcbpggsz.Interface import *
import tensorflow as tf
import numpy as np
import math

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


class Baryonic3BodyPhaseSpace(DalitzPhaseSpace):
    """
      Derived class for baryonic 3-body decay, baryon -> scalar scalar baryon
      Include 2D phase-space + 3 decay plane orientation angular variables, for full polarization treatment
    """

    def CosThetaA(self, sample):
        """
          Return thetaa variable (vector) for the input sample
        """
        return sample[:, 2]

    def PhiA(self, sample):
        """
          Return phia variable (vector) for the input sample
        """
        return sample[:, 3]

    def PhiBC(self, sample):
        """
          Return phibc variable (vector) for the input sample
        """
        return sample[:, 4]

    def Inside(self, x):
        """
          Check if the point x=(M2ab, M2bc, CosThetaA, PhiA, PhiBC) is inside the phase space
        """
        m2ab = self.M2ab(x)
        m2bc = self.M2bc(x)
        mab = Sqrt(m2ab)
        costhetaa = self.CosThetaA(x)
        phia = self.PhiA(x)
        phibc = self.PhiBC(x)

        inside = tf.logical_and(tf.logical_and(tf.greater(m2ab, self.minab), tf.less(m2ab, self.maxab)),
                                tf.logical_and(tf.greater(m2bc, self.minbc), tf.less(m2bc, self.maxbc)))

        if self.macrange:
            m2ac = self.msqsum - m2ab - m2bc
            inside = tf.logical_and(inside, tf.logical_and(tf.greater(
                m2ac, self.macrange[0]**2), tf.less(m2ac, self.macrange[1]**2)))

        if self.symmetric:
            inside = tf.logical_and(inside, tf.greater(m2bc, m2ab))

        eb = (m2ab - self.ma2 + self.mb2)/2./mab
        ec = (self.md2 - m2ab - self.mc2)/2./mab
        p2b = eb**2 - self.mb2
        p2c = ec**2 - self.mc2
        inside = tf.logical_and(inside, tf.logical_and(
            tf.greater(p2c, 0), tf.greater(p2b, 0)))
        pb = Sqrt(p2b)
        pc = Sqrt(p2c)
        e2bc = (eb+ec)**2
        m2bc_max = e2bc - (pb - pc)**2
        m2bc_min = e2bc - (pb + pc)**2

        inside_phsp = tf.logical_and(inside, tf.logical_and(
            tf.greater(m2bc, m2bc_min), tf.less(m2bc, m2bc_max)))

        inside_theta = tf.logical_and(tf.greater(
            costhetaa, -1.), tf.less(costhetaa, 1.))
        inside_phi = tf.logical_and(tf.logical_and(tf.greater(phia, -1.*math.pi), tf.less(phia, math.pi)),
                                    tf.logical_and(tf.greater(phibc, -1.*math.pi), tf.less(phibc, math.pi)))
        inside_ang = tf.logical_and(inside_theta, inside_phi)

        return tf.logical_and(inside_phsp, inside_ang)

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
        v = [np.random.uniform(self.minab, self.maxab, size).astype('d'),
             np.random.uniform(self.minbc, self.maxbc, size).astype('d'),
             np.random.uniform(-1., 1., size).astype('d'),
             np.random.uniform(-1.*math.pi, math.pi, size).astype('d'),
             np.random.uniform(-1.*math.pi, math.pi, size).astype('d'), ]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.transpose(np.array(v))

    def Placeholder(self, name=None):
        """
          Create a placeholder for a dataset in this phase space 
        """
        return tf.placeholder(fptype, shape=(None, None), name=name)

    def FinalStateMomenta(self, m2ab, m2bc, costhetaa, phia, phibc):
        """
          Calculate 4-momenta of final state tracks in the 5D phase space
            m2ab, m2bc : invariant masses of AB and BC combinations
            (cos)thetaa, phia : direction angles of the particle A in the D reference frame
            phibc : angle of BC plane wrt. polarisation plane z x p_a
        """

        thetaa = Acos(costhetaa)

        m2ac = self.msqsum - m2ab - m2bc

        # Magnitude of the momenta
        p_a = TwoBodyMomentum(self.md, self.ma, Sqrt(m2bc))
        p_b = TwoBodyMomentum(self.md, self.mb, Sqrt(m2ac))
        p_c = TwoBodyMomentum(self.md, self.mc, Sqrt(m2ab))

        cos_theta_b = (p_a*p_a + p_b*p_b - p_c*p_c)/(2.*p_a*p_b)
        cos_theta_c = (p_a*p_a + p_c*p_c - p_b*p_b)/(2.*p_a*p_c)

        # Fix momenta with p3a oriented in z (quantisation axis) direction
        p3a = Vector(Zeros(p_a), Zeros(p_a), p_a)
        p3b = Vector(p_b*Sqrt(1. - cos_theta_b**2),
                     Zeros(p_b), -p_b*cos_theta_b)
        p3c = Vector(-p_c*Sqrt(1. - cos_theta_c**2),
                     Zeros(p_c), -p_c*cos_theta_c)

        # Rotate vectors to have p3a with thetaa as polar helicity angle
        p3a = RotateVector(p3a, Const(0.), thetaa, Const(0.))
        p3b = RotateVector(p3b, Const(0.), thetaa, Const(0.))
        p3c = RotateVector(p3c, Const(0.), thetaa, Const(0.))

        # Rotate vectors to have p3a with phia as azimuthal helicity angle
        p3a = RotateVector(p3a, phia, Const(0.), Const(0.))
        p3b = RotateVector(p3b, phia, Const(0.), Const(0.))
        p3c = RotateVector(p3c, phia, Const(0.), Const(0.))

        # Rotate BC plane to have phibc as angle with the polarization plane
        p3b = Rotate(p3b, phibc, p3a)
        p3c = Rotate(p3c, phibc, p3a)

        # Define 4-vectors
        p4a = LorentzVector(p3a, Sqrt(p_a**2 + self.ma2))
        p4b = LorentzVector(p3b, Sqrt(p_b**2 + self.mb2))
        p4c = LorentzVector(p3c, Sqrt(p_c**2 + self.mc2))

        return (p4a, p4b, p4c)

    def Dimensionality(self):
        return 5
