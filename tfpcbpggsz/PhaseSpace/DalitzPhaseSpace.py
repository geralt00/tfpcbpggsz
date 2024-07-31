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
tf.compat.v1.disable_eager_execution()


import numpy as np
import math

import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


class DalitzPhaseSpace:
    """
    Class for Dalitz plot (2D) phase space for the 3-body decay D->ABC
    """

    def __init__(self, ma, mb, mc, md, mabrange=None, mbcrange=None, macrange=None, symmetric=False):
        """
        Constructor
          ma - A mass
          mb - B mass
          mc - C mass
          md - D (mother) mass
        """
        self.ma = ma
        self.mb = mb
        self.mc = mc
        self.md = md
        self.ma2 = ma*ma
        self.mb2 = mb*mb
        self.mc2 = mc*mc
        self.md2 = md*md
        self.msqsum = self.md2 + self.ma2 + self.mb2 + self.mc2
        self.minab = (ma + mb)**2
        self.maxab = (md - mc)**2
        self.minbc = (mb + mc)**2
        self.maxbc = (md - ma)**2
        self.minac = (ma + mc)**2
        self.maxac = (md - mb)**2
        self.macrange = macrange
        self.symmetric = symmetric
        self.min_mprimeac = 0.0
        self.max_mprimeac = 1.0
        self.min_thprimeac = 0.0
        self.max_thprimeac = 1.0
        if self.symmetric:
            self.max_thprimeac = 0.5
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
        mab = Sqrt(m2ab)

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
             np.random.uniform(self.minbc, self.maxbc, size).astype('d')]

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
             tf.random.uniform([size], self.minbc, self.maxbc, dtype = FPType()) ]

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

    def RectangularGridSample(self, sizeab, sizebc, spaceTosample="DP"):
        """
          Create a data sample in the form of rectangular grid of points within the phase space.
          Useful for normalisation.
            sizeab : number of grid nodes in M2ab range
            sizebc : number of grid nodes in M2bc range
            spaceTosample: Sampling is done according to cases below but all of them return DP vars (m^2_{ab}, m^2_{bc}).
                -if 'DP': Unifrom sampling is in (m^2_{ab}, m^2_{bc})
                -if 'linDP': Samples in (m_{ab}, m_{bc})
                -if 'sqDP': Samples in (mPrimeAC, thPrimeAC).
        """
        size = sizeab*sizebc
        mgrid = np.lib.index_tricks.nd_grid()
        if spaceTosample == "linDP":
            vab = (mgrid[0:sizeab, 0:sizebc][0]*(math.sqrt(self.maxab) -
                                                 math.sqrt(self.minab))/float(sizeab) + math.sqrt(self.minab))**2.
            vbc = (mgrid[0:sizeab, 0:sizebc][1]*(math.sqrt(self.maxbc) -
                                                 math.sqrt(self.minbc))/float(sizebc) + math.sqrt(self.minbc))**2.
            v = [vab.reshape(size).astype('d'), vbc.reshape(size).astype('d')]
            dlz = tf.stack(v, axis=1)
        elif spaceTosample == "sqDP":
            x = np.linspace(self.min_mprimeac,  self.max_mprimeac,  sizeab)
            y = np.linspace(self.min_thprimeac, self.max_thprimeac, sizebc)
            # Remove corners of sqDP as they lie outside phsp
            xnew = x[(x > self.min_mprimeac) & (x < self.max_mprimeac) & (
                y > self.min_thprimeac) & (y < self.max_thprimeac)]
            ynew = y[(x > self.min_mprimeac) & (x < self.max_mprimeac) & (
                y > self.min_thprimeac) & (y < self.max_thprimeac)]
            mprimeac, thprimeac = np.meshgrid(xnew, ynew)
            dlz = self.FromSquareDalitzPlot(
                mprimeac.flatten().astype('d'), thprimeac.flatten().astype('d'))
        else:
            vab = mgrid[0:sizeab, 0:sizebc][0] * \
                (self.maxab-self.minab)/float(sizeab) + self.minab
            vbc = mgrid[0:sizeab, 0:sizebc][1] * \
                (self.maxbc-self.minbc)/float(sizebc) + self.minbc
            v = [vab.reshape(size).astype('d'), vbc.reshape(size).astype('d')]
            dlz = tf.stack(v, axis=1)

        return self.Filter(dlz)

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

    def M2ac(self, sample):
        """
          Return M2ac variable (vector) for the input sample.
          It is calculated from M2ab and M2bc
        """
        return self.msqsum - self.M2ab(sample) - self.M2bc(sample)

    def CosHelicityAB(self, sample):
        """
          Calculate cos(helicity angle) of the AB resonance
        """
        return CosHelicityAngleDalitz(self.M2ab(sample), self.M2bc(sample), self.md, self.ma, self.mb, self.mc)

    def CosHelicityBC(self, sample):
        """
           Calculate cos(helicity angle) of the BC resonance
        """
        return CosHelicityAngleDalitz(self.M2bc(sample), self.M2ac(sample), self.md, self.mb, self.mc, self.ma)

    def CosHelicityAC(self, sample):
        """
           Calculate cos(helicity angle) of the AC resonance
        """
        return CosHelicityAngleDalitz(self.M2ac(sample), self.M2ab(sample), self.md, self.mc, self.ma, self.mb)

    def MPrimeAC(self, sample):
        """
          Square Dalitz plot variable m'
        """
        mac = Sqrt(self.M2ac(sample))
        return Acos(2*(mac - math.sqrt(self.minac))/(math.sqrt(self.maxac) - math.sqrt(self.minac)) - 1.)/math.pi

    def ThetaPrimeAC(self, sample):
        """
          Square Dalitz plot variable theta'
        """
        return Acos(self.CosHelicityAC(sample))/math.pi

    def MPrimeAB(self, sample):
        """
          Square Dalitz plot variable m'
        """
        mab = Sqrt(self.M2ab(sample))
        return Acos(2*(mab - math.sqrt(self.minab))/(math.sqrt(self.maxab) - math.sqrt(self.minab)) - 1.)/math.pi

    def MAB(self,sample):
        return Sqrt(self.M2ab(sample))

    def FromSquareDalitzPlot(self, mprimeac, thprimeac):
        """
          sample: Given mprimeac and thprimeac, returns 2D tensor for (m2ab, m2bc). 
          Make sure you don't pass in sqDP corner points as they lie outside phsp.
        """
        m2AC = 0.25*(self.maxac**0.5*Cos(math.pi*mprimeac) + self.maxac **
                     0.5 - self.minac**0.5*Cos(math.pi*mprimeac) + self.minac**0.5)**2
        m2AB = 0.5*(-m2AC**2 + m2AC*self.ma**2 + m2AC*self.mb**2 + m2AC*self.mc**2 + m2AC*self.md**2 -
                    m2AC*Sqrt((m2AC*(m2AC - 2.0*self.ma**2 - 2.0*self.mc**2) +
                               self.ma**4 - 2.0*self.ma**2*self.mc**2 + self.mc**4)/m2AC)*Sqrt((m2AC*(m2AC - 2.0*self.mb**2 - 2.0*self.md**2) +
                                                                                                self.mb**4 - 2.0*self.mb**2*self.md**2 + self.md**4)/m2AC)*Cos(math.pi*thprimeac) -
                    self.ma**2*self.mb**2 + self.ma**2*self.md**2 + self.mb**2*self.mc**2 - self.mc**2*self.md**2)/m2AC
        m2BC = self.msqsum - m2AC - m2AB
        return tf.stack([m2AB, m2BC], axis=1)

    def SquareDalitzPlotJacobian(self, sample):
        """
          sample: [mAB^2, mBC^2]
          Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dMpr*dThpr where Mpr, Thpr are defined in (AC) frame.
        """
        mPrime = self.MPrimeAC(sample)
        thPrime = self.ThetaPrimeAC(sample)

        diff_AC = tf.cast(Sqrt(self.maxac) - Sqrt(self.minac), FPType())
        mAC = Const(0.5) * diff_AC * (Const(1.) + Cos(Pi()*mPrime)
                                      ) + tf.cast(Sqrt(self.minac), FPType())
        mACSq = mAC*mAC

        eAcmsAC = Const(0.5) * (mACSq - tf.cast(self.mc2,
                                                FPType()) + tf.cast(self.ma2, FPType()))/mAC
        eBcmsAC = Const(0.5) * (tf.cast(self.md, FPType()) **
                                2. - mACSq - tf.cast(self.mb2, FPType()))/mAC

        pAcmsAC = Sqrt(eAcmsAC**2. - tf.cast(self.ma2, FPType()))
        pBcmsAC = Sqrt(eBcmsAC**2. - tf.cast(self.mb2, FPType()))

        deriv1 = Pi() * Const(0.5) * diff_AC * Sin(Pi()*mPrime)
        deriv2 = Pi() * Sin(Pi() * thPrime)

        return Const(4.) * pAcmsAC * pBcmsAC * mAC * deriv1 * deriv2

    def InvariantMassJacobian(self, sample):
        """
          sample: [mAB^2, mBC^2]
          Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dmAB*dmBC. |J| = 4*mAB*mBC
        """
        return Const(4.) * Sqrt(self.M2ab(sample)) * Sqrt(self.M2bc(sample))

    def ThetaPrimeAB(self, sample):
        """
          Square Dalitz plot variable theta'
        """
        return Acos(-self.CosHelicityAB(sample))/math.pi

    def MPrimeBC(self, sample):
        """
          Square Dalitz plot variable m'
        """
        mbc = Sqrt(self.M2bc(sample))
        return Acos(2*(mbc - math.sqrt(self.minbc))/(math.sqrt(self.maxbc) - math.sqrt(self.minbc)) - 1.)/math.pi

    def ThetaPrimeBC(self, sample):
        """
          Square Dalitz plot variable theta'
        """
        return Acos(-self.CosHelicityBC(sample))/math.pi

    def Placeholder(self, name=None):
        """
          Create a placeholder for a dataset in this phase space 
        """
        return tf.compat.v1.placeholder(FPType(), shape=(None, None), name=name)

    def FromVectors(self, m2ab, m2bc):
        """
          Create Dalitz plot tensor from two vectors of variables, m2ab and m2bc
        """
        return tf.stack([m2ab, m2bc], axis=1)

    def FinalStateMomenta(self, m2ab, m2bc):
        """
          Calculate 4-momenta of final state tracks in a certain reference frame
          (decay is in x-z plane, particle A moves along z axis)
            m2ab, m2bc : invariant masses of AB and BC combinations
        """

        m2ac = self.msqsum - m2ab - m2bc

        p_a = TwoBodyMomentum(self.md, self.ma, Sqrt(m2bc))
        p_b = TwoBodyMomentum(self.md, self.mb, Sqrt(m2ac))
        p_c = TwoBodyMomentum(self.md, self.mc, Sqrt(m2ab))

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
        return 2
