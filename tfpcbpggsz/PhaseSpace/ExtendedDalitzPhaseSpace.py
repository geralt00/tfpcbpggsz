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

from tfpcbpggsz.Kinematics import *
from tfpcbpggsz.Interface import *
import tensorflow as tf
import numpy as np
import math

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


class ExtendedDalitzPhaseSpace:
    """
    Class for Dalitz plot phase space for the 3-body decay D->ABC, where the mass of the D changes
    """

    def __init__(self, ma, mb, mc, mdmin, mdmax, mabrange=None, mbcrange=None, macrange=None):
        """
        Constructor
          ma - A mass
          mb - B mass
          mc - C mass
          mdmin - minimum D (mother) mass
          mdmax - maximum D mass
        """
        self.ma = ma
        self.mb = mb
        self.mc = mc
        self.mdmin = mdmin
        self.mdmax = mdmax
        self.ma2 = ma*ma
        self.mb2 = mb*mb
        self.mc2 = mc*mc
        self.md2min = mdmin*mdmin
        self.md2max = mdmax*mdmax
        self.msqsum = self.ma2 + self.mb2 + self.mc2
        self.minab = (ma + mb)**2
        self.maxab = (mdmax - mc)**2
        self.minbc = (mb + mc)**2
        self.maxbc = (mdmax - ma)**2
        self.minac = (ma + mc)**2
        self.maxac = (mdmax - mb)**2
        self.macrange = macrange
        if mabrange:
            if mabrange[1]**2 < self.maxab:
                self.maxab = mabrange[1]**2
            if mabrange[0]**2 > self.minab:
                self.minab = mabrange[0]**2
        if mbcrange:
            if mbcrange[1]**2 < self.maxbc:
                self.maxbc = mbcrange[1]**2
            if mbcrange[0]**2 > self.minbc:
                self.minbc = mbcrange[0]**2
        self.data_placeholder = self.Placeholder("data")
        self.norm_placeholder = self.Placeholder("norm")

    def Inside(self, x):
        """
          Check if the point x=(M2ab, M2bc) is inside the phase space
        """
        m2ab = self.M2ab(x)
        m2bc = self.M2bc(x)
        md = self.Md(x)
        md2 = md*md
        mab = Sqrt(m2ab)
        msqsum = md2 + self.msqsum

        inside = tf.logical_and(tf.logical_and(tf.greater(m2ab, self.minab), tf.less(m2ab, self.maxab)),
                                tf.logical_and(tf.greater(m2bc, self.minbc), tf.less(m2bc, self.maxbc)))

        inside = tf.logical_and(inside, tf.logical_and(tf.greater(md, self.mdmin), tf.less(md, self.mdmax)))

        if self.macrange:
            m2ac = msqsum - m2ab - m2bc
            inside = tf.logical_and(inside, tf.logical_and(tf.greater(
                m2ac, self.macrange[0]**2), tf.less(m2ac, self.macrange[1]**2)))

        eb = (m2ab - self.ma2 + self.mb2)/2./mab
        ec = (md2 - m2ab - self.mc2)/2./mab
        p2b = eb**2 - self.mb2
        p2c = ec**2 - self.mc2
        inside = tf.logical_and(inside, tf.logical_and(
            tf.greater(p2c, 0), tf.greater(p2b, 0)))
        pb = Sqrt(p2b)
        pc = Sqrt(p2c)
        e2bc = (eb+ec)**2
        m2bc_max = e2bc - (pb - pc)**2
        m2bc_min = e2bc - (pb + pc)**2
        return tf.logical_and(inside, tf.logical_and(tf.greater(m2bc, m2bc_min), tf.less(m2bc, m2bc_max)))

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
             np.random.uniform(self.mdmin, self.mdmax, size).astype('d')]

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
        v = [tf.random.uniform([size], self.minab, self.maxab, dtype = FPType()),
             tf.random.uniform([size], self.minbc, self.maxbc, dtype = FPType()), 
             tf.random.uniform([size], self.mdmin, self.mdmax, dtype = FPType()) ]

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

    def M2ab(self, sample):
        """
          Return M2ab variable (vector) for the input sample
        """
        return sample[:, 0]

    def M2bc(self, sample):
        """
           Return M2bc variable (vector) for the input sample
        """
        return sample[:, 1]

    def Md(self, sample):
        """
           Return Md variable (vector) for the input sample
        """
        return sample[:, 2]

    def M2ac(self, sample):
        """
          Return M2ac variable (vector) for the input sample.
          It is calculated from M2ab and M2bc
        """
        md2 = self.Md(sample)**2
        msqsum = md2 + self.msqsum
        return msqsum - self.M2ab(sample) - self.M2bc(sample)

    def CosHelicityAB(self, sample):
        """
          Calculate cos(helicity angle) of the AB resonance
        """
        return CosHelicityAngleDalitz(self.M2ab(sample), self.M2bc(sample), self.Md(sample), self.ma, self.mb, self.mc)

    def CosHelicityBC(self, sample):
        """
           Calculate cos(helicity angle) of the BC resonance
        """
        return CosHelicityAngleDalitz(self.M2bc(sample), self.M2ac(sample), self.Md(sample), self.mb, self.mc, self.ma)

    def CosHelicityAC(self, sample):
        """
           Calculate cos(helicity angle) of the AC resonance
        """
        return CosHelicityAngleDalitz(self.M2ac(sample), self.M2ab(sample), self.Md(sample), self.mc, self.ma, self.mb)

    def Placeholder(self, name=None):
        """
          Create a placeholder for a dataset in this phase space 
        """
        return tf.placeholder(FPType(), shape=(None, None), name=name)

    def FromVectors(self, m2ab, m2bc, md):
        """
          Create Dalitz plot tensor from two vectors of variables, m2ab and m2bc
        """
        return tf.stack([m2ab, m2bc, md], axis=1)

    def FinalStateMomenta(self, m2ab, m2bc, md):
        """
          Calculate 4-momenta of final state tracks in a certain reference frame
          (decay is in x-z plane, particle A moves along z axis)
            m2ab, m2bc : invariant masses of AB and BC combinations
        """

        md2 = md*md
        msqsum = md2 + self.msqsum
        m2ac = msqsum - m2ab - m2bc

        p_a = TwoBodyMomentum(md, self.ma, Sqrt(m2bc))
        p_b = TwoBodyMomentum(md, self.mb, Sqrt(m2ac))
        p_c = TwoBodyMomentum(md, self.mc, Sqrt(m2ab))

        cos_theta_b = (p_a*p_a + p_b*p_b - p_c*p_c)/(2.*p_a*p_b)
        cos_theta_c = (p_a*p_a + p_c*p_c - p_b*p_b)/(2.*p_a*p_c)

        p4a = LorentzVector(Vector(Zeros(p_a), Zeros(p_a),
                                   p_a), Sqrt(p_a**2 + self.ma2))
        p4b = LorentzVector(Vector(p_b*Sqrt(1. - cos_theta_b**2),
                                   Zeros(p_b), -p_b*cos_theta_b), Sqrt(p_b**2 + self.mb2))
        p4c = LorentzVector(Vector(-p_c*Sqrt(1. - cos_theta_c**2),
                                   Zeros(p_c), -p_c*cos_theta_c), Sqrt(p_c**2 + self.mc2))

        return (p4a, p4b, p4c)

    def Dimensionality(self):
        return 3
