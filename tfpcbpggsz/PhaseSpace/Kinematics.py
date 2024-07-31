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

import sys
import operator
import tensorflow as tf
import numpy as np
import math
import itertools

from tfpcbpggsz.Interface import *
#from tfpcbpggsz.Optimisation import *


def SpatialComponents(vector):
    """Return spatial components of the input Lorentz vector

    :param vector: input Lorentz vector
    :returns: tensor of spatial components

    """
    return vector[:, 0:3]


def TimeComponent(vector):
    """Return time component of the input Lorentz vector

    :param vector: input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    :returns: vector of time components

    """
    return vector[:, 3]


def XComponent(vector):
    """Return spatial X component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of X-components

    """
    return vector[:, 0]


def YComponent(vector):
    """Return spatial Y component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of Y-components

    """
    return vector[:, 1]


def ZComponent(vector):
    """Return spatial Z component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of Z-components

    """
    return vector[:, 2]


def Pt(vector):
    """Return transverse (X-Y) component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of transverse components

    """
    return Sqrt(XComponent(vector)**2 + YComponent(vector)**2)


def Eta(vector):
    """Return pseudorapidity component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of pseudorapidity components

    """
    return -Log(Pt(vector)/2./ZComponent(vector))


def Vector(x, y, z):
    """Make a 3-vector from components
      x, y, z : vector components

    :param x: 
    :param y: 
    :param z: 

    """
    return tf.stack([x, y, z], axis=1)


def Scalar(x):
    """Create a scalar (e.g. tensor with only one component) which can be used to e.g. scale a vector
    One cannot do e.g. Const(2.)*Vector(x, y, z), needs to do Scalar(Const(2))*Vector(x, y, z)

    :param x: 

    """
    return tf.stack([x], axis=1)


def LorentzVector(space, time):
    """Make a Lorentz vector from spatial and time components
      space : 3-vector of spatial components
      time  : time component

    :param space: 
    :param time: 

    """
    return tf.concat([space, tf.stack([time], axis=1)], axis=1)


def MetricTensor():
    """Metric tensor for Lorentz space (constant)"""
    return tf.constant([-1., -1., -1., 1.], dtype=FPType())


def Mass(vector):
    """Calculate mass scalar for Lorentz 4-momentum
      vector : input Lorentz momentum vector

    :param vector: 

    """
    return Sqrt(tf.reduce_sum(vector*vector*MetricTensor(), 1))


def ScalarProduct(vec1, vec2):
    """Calculate scalar product of two 3-vectors

    :param vec1: 
    :param vec2: 

    """
    return tf.reduce_sum(vec1*vec2, 1)


def VectorProduct(vec1, vec2):
    """Calculate vector product of two 3-vectors

    :param vec1: 
    :param vec2: 

    """
    return tf.cross(vec1, vec2)


def CrossProduct(vec1, vec2):
    """Calculate cross product of two 3-vectors

    :param vec1: 
    :param vec2: 

    """
    return tf.cross(vec1, vec2)


def Norm(vec):
    """Calculate norm of 3-vector

    :param vec: 

    """
    return Sqrt(tf.reduce_sum(vec*vec, 1))


def P(vector):
    """Calculate absolute value of the 4-momentum

    :param vector: 

    """
    return Norm(SpatialComponents(vector))


def UnitVector(vec):
    """Unit vector in the direction of vec

    :param vec: 

    """
    return vec/Scalar(Norm(vec))


def PerpendicularUnitVector(vec1, vec2):
    """Unit vector perpendicular to the plane formed by vec1 and vec2

    :param vec1: 
    :param vec2: 

    """
    v = VectorProduct(vec1, vec2)
    return v/Scalar(Norm(v))


def LorentzBoost(vector, boostvector):
    """Perform Lorentz boost
      vector :     4-vector to be boosted
      boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components are used)

    :param vector: 
    :param boostvector: 

    """
    boost = SpatialComponents(boostvector)
    b2 = ScalarProduct(boost, boost)
    gamma = 1./Sqrt(1.-b2)
    gamma2 = (gamma-1.0)/b2
    ve = TimeComponent(vector)
    vp = SpatialComponents(vector)
    bp = ScalarProduct(vp, boost)
    vp2 = vp + Scalar(gamma2*bp + gamma*ve)*boost
    ve2 = gamma*(ve + bp)
    return LorentzVector(vp2, ve2)


def BoostToRest(vector, boostvector):
    """Perform Lorentz boost to the rest frame of the 4-vector boostvector.

    :param vector: 
    :param boostvector: 

    """
    boost = -SpatialComponents(boostvector)/Scalar(TimeComponent(boostvector))
    return LorentzBoost(vector, boost)


def BoostFromRest(vector, boostvector):
    """Perform Lorentz boost from the rest frame of the 4-vector boostvector.

    :param vector: 
    :param boostvector: 

    """
    boost = SpatialComponents(boostvector)/Scalar(TimeComponent(boostvector))
    return LorentzBoost(vector, boost)


def Rotate(v, angle, axis):
    """Rotate vector around an arbitrary axis, from ROOT implementation

    :param v: 
    :param angle: 
    :param axis: 

    """
    if (angle != Zeros(angle)):
        ll = Norm(axis)
        if (ll == Zeros(ll)):
            sys.exit('ERROR in Rotate: rotation axis is zero')
        else:
            sa = Sin(angle)
            ca = Cos(angle)
            dx = XComponent(axis)/ll
            dy = YComponent(axis)/ll
            dz = ZComponent(axis)/ll
            vx = XComponent(v)
            vy = YComponent(v)
            vz = ZComponent(v)
            _vx = (ca+(1-ca)*dx*dx)*vx + ((1-ca)*dx*dy-sa*dz) * \
                vy + ((1-ca)*dx*dz+sa*dy)*vz
            _vy = ((1-ca)*dy*dx+sa*dz)*vx + (ca+(1-ca)*dy*dy) * \
                vy + ((1-ca)*dy*dz-sa*dx)*vz
            _vz = ((1-ca)*dz*dx-sa*dy)*vx + \
                ((1-ca)*dz*dy+sa*dx)*vy + (ca+(1-ca)*dz*dz)*vz

            return Vector(_vx, _vy, _vz)

    else:
        return v


def RotateVector(v, phi, theta, psi):
    """Perform 3D rotation of the 3-vector
      v : vector to be rotated
      phi, theta, psi : Euler angles in Z-Y-Z convention

    :param v: 
    :param phi: 
    :param theta: 
    :param psi: 

    """

    # Rotate Z (phi)
    c1 = Cos(phi)
    s1 = Sin(phi)
    c2 = Cos(theta)
    s2 = Sin(theta)
    c3 = Cos(psi)
    s3 = Sin(psi)

    # Rotate Y (theta)
    fzx2 = -s2*c1
    fzy2 = s2*s1
    fzz2 = c2

    # Rotate Z (psi)
    fxx3 = c3*c2*c1 - s3*s1
    fxy3 = -c3*c2*s1 - s3*c1
    fxz3 = c3*s2
    fyx3 = s3*c2*c1 + c3*s1
    fyy3 = -s3*c2*s1 + c3*c1
    fyz3 = s3*s2

    # Transform v
    vx = XComponent(v)
    vy = YComponent(v)
    vz = ZComponent(v)

    _vx = fxx3*vx + fxy3*vy + fxz3*vz
    _vy = fyx3*vx + fyy3*vy + fyz3*vz
    _vz = fzx2*vx + fzy2*vy + fzz2*vz

    return Vector(_vx, _vy, _vz)


def RotateLorentzVector(v, phi, theta, psi):
    """Perform 3D rotation of the 4-vector
      v : vector to be rotated
      phi, theta, psi : Euler angles in Z-Y-Z convention

    :param v: 
    :param phi: 
    :param theta: 
    :param psi: 

    """
    return LorentzVector(RotateVector(SpatialComponents(v), phi, theta, psi), TimeComponent(v))


def ProjectLorentzVector(p, axes):
    """

    :param p: 
    :param axes: 

    """
    (x1, y1, z1) = axes
    p0 = SpatialComponents(p)
    p1 = LorentzVector(Vector(ScalarProduct(x1, p0), ScalarProduct(
        y1, p0), ScalarProduct(z1, p0)), TimeComponent(p))
    return p1


def CosHelicityAngleDalitz(m2ab, m2bc, md, ma, mb, mc):
    """Calculate cos(helicity angle) for set of two Dalitz plot variables
      m2ab, m2bc : Dalitz plot variables (inv. masses squared of AB and BC combinations)
      md : mass of the decaying particle
      ma, mb, mc : masses of final state particles

    :param m2ab: 
    :param m2bc: 
    :param md: 
    :param ma: 
    :param mb: 
    :param mc: 

    """
    md2 = md**2
    ma2 = ma**2
    mb2 = mb**2
    mc2 = mc**2
    m2ac = md2 + ma2 + mb2 + mc2 - m2ab - m2bc
    mab = Sqrt(m2ab)
    mac = Sqrt(m2ac)
    mbc = Sqrt(m2bc)
    p2a = 0.25/md2*(md2-(mbc+ma)**2)*(md2-(mbc-ma)**2)
    p2b = 0.25/md2*(md2-(mac+mb)**2)*(md2-(mac-mb)**2)
    p2c = 0.25/md2*(md2-(mab+mc)**2)*(md2-(mab-mc)**2)
    eb = (m2ab-ma2+mb2)/2./mab
    ec = (md2-m2ab-mc2)/2./mab
    pb = Sqrt(eb**2-mb2)
    pc = Sqrt(ec**2-mc2)
    e2sum = (eb+ec)**2
    m2bc_max = e2sum-(pb-pc)**2
    m2bc_min = e2sum-(pb+pc)**2
    return (m2bc_max + m2bc_min - 2.*m2bc)/(m2bc_max-m2bc_min)


def SphericalAngles(pb):
    """theta, phi : polar and azimuthal angles of the vector pb

    :param pb: 

    """
    z1 = UnitVector(SpatialComponents(
        pb))       # New z-axis is in the direction of pb
    theta = Acos(ZComponent(z1))                 # Helicity angle
    phi = Atan2(YComponent(pb), XComponent(pb))  # Phi angle
    return (theta, phi)


def HelicityAngles(pb):
    """theta, phi : polar and azimuthal angles of the vector pb

    :param pb: 

    """
    return SphericalAngles(pb)


def FourMomentaFromHelicityAngles(md, ma, mb, theta, phi):
    """Calculate the four-momenta of the decay products in D->AB in the rest frame of D
        md:    mass of D
        ma:    mass of A
        mb:    mass of B
        theta: angle between A momentum in D rest frame and D momentum in its helicity frame
        phi:   angle of plane formed by A & B in D helicity frame

    :param md: 
    :param ma: 
    :param mb: 
    :param theta: 
    :param phi: 

    """
    # Calculate magnitude of momentum in D rest frame
    p = TwoBodyMomentum(md, ma, mb)
    # Calculate energy in D rest frame
    Ea = Sqrt(p**2 + ma**2)
    Eb = Sqrt(p**2 + mb**2)
    # Construct four-momenta with A aligned with D in D helicity frame
    Pa = LorentzVector(Vector(Zeros(p), Zeros(p),  p), Ea)
    Pb = LorentzVector(Vector(Zeros(p), Zeros(p), -p), Eb)
    # Rotate four-momenta
    Pa = RotateLorentzVector(Pa, Zeros(phi), -theta, -phi)
    Pb = RotateLorentzVector(Pb, Zeros(phi), -theta, -phi)
    return Pa, Pb


def RecursiveSum(vectors):
    """Helper function fro CalculateHelicityAngles. It sums all the vectors in
      a list or nested list

    :param vectors: 

    """
    return sum([RecursiveSum(vector) if isinstance(vector, list) else vector for vector in vectors])


def CalculateHelicityAngles(pdecays):
    """Calculate the Helicity Angles for every decay topology specified with brackets []
    examples:
       - input:
         A -> B (-> C D) E (-> F G) ==> CalculateHelicityAngles([[C,D],[F,G]])
         A -> B (-> C (-> D E) F) G ==> CalculateHelicityAngles([ [ [ D, E] , F ] , G ])
       - output:
         A -> B (-> C D) E (-> F G) ==> (thetaB,phiB,thetaC,phiC,thetaF,phiF)
         A -> B (-> C (-> D E) F) G ==> (thetaB,phiB,thetaC,phiC,thetaD,phiD)
         where thetaX,phiX are the polar and azimuthal angles of X in the mother rest frame

    :param pdecays: 

    """
    angles = ()
    if len(pdecays) != 2:
        sys.exit(
            'ERROR in CalculateHelicityAngles: lenght of the input list is different from 2')

    for i, pdau in enumerate(pdecays):
        if i == 0:
            angles += HelicityAngles(RecursiveSum(pdau)
                                     if isinstance(pdau, list) else pdau)
        # the particle is not basic but decay, rotate and boost to its new rest frame
        if isinstance(pdau, list):
            pmother = RecursiveSum(pdau)
            pdau_newframe = RotationAndBoost(pdau, pmother)
            angles += CalculateHelicityAngles(pdau_newframe)
    return angles


def ChangeAxes(ps, newaxes):
    """List of LorentzVector with the component described by the
      new axes (x,y,z).

    :param ps: 
    :param newaxes: 

    """
    (xnew, ynew, znew) = newaxes
    pout = []
    for p in ps:
        px = XComponent(p)
        py = YComponent(p)
        pz = ZComponent(p)
        pout.append(LorentzVector(Vector(px*XComponent(xnew)+py*YComponent(xnew)+pz*ZComponent(xnew),
                                         px*XComponent(ynew)+py*YComponent(ynew)+pz*ZComponent(ynew),
                                         px*XComponent(znew)+py*YComponent(znew)+pz*ZComponent(znew)), TimeComponent(p)))
    return pout


def RotatedAxes(pb, oldaxes=None):
    """Calculate new (rotated) axes aligned with the momentum vector pb

    :param pb: 
    :param oldaxes:  (Default value = None)

    """
    z1 = UnitVector(SpatialComponents(
        pb))       # New z-axis is in the direction of pb
    eb = TimeComponent(pb)
    zeros = Zeros(eb)
    ones = Ones(eb)
    # Old z-axis vector
    z0 = Vector(zeros, zeros, ones) if oldaxes == None else oldaxes[2]
    # Old x-axis vector
    x0 = Vector(ones, zeros, zeros) if oldaxes == None else oldaxes[0]
    sp = ScalarProduct(z1, z0)
    a0 = z0 - z1*Scalar(sp)   # Vector in z-pb plane perpendicular to z0
    x1 = tf.where(tf.equal(sp, 1.0), x0, -UnitVector(a0))
    y1 = VectorProduct(z1, x1)                   # New y-axis
    return (x1, y1, z1)


def OldAxes(pb):
    """Calculate old (before rotation) axes in the frame aligned with the momentum vector pb

    :param pb: 

    """
    z1 = UnitVector(SpatialComponents(
        pb))       # New z-axis is in the direction of pb
    eb = TimeComponent(pb)
    z0 = Vector(Zeros(eb), Zeros(eb), Ones(eb))  # Old z-axis vector
    x0 = Vector(Ones(eb), Zeros(eb), Zeros(eb))  # Old x-axis vector
    sp = ScalarProduct(z1, z0)
    a0 = z0 - z1*Scalar(sp)   # Vector in z-pb plane perpendicular to z0
    x1 = tf.where(tf.equal(sp, 1.0), x0, -UnitVector(a0))
    y1 = VectorProduct(z1, x1)                   # New y-axis
    x = Vector(XComponent(x1), XComponent(y1), XComponent(z1))
    y = Vector(YComponent(x1), YComponent(y1), YComponent(z1))
    z = Vector(ZComponent(x1), ZComponent(y1), ZComponent(z1))
    return (x, y, z)


def RotationAndBoost(ps, pb):
    """Rotate and boost all momenta from the list ps to the rest frame of pb
      After the rotation, the coordinate system is defined as:
        z axis: direction of pb
        y axis: perpendicular to the plane formed by the old z and pb
        x axis: [y,z]
    
      ps : list of Lorentz vectors to rotate and boost
      pb : Lorentz vector defining the new frame

    :param ps: 
    :param pb: 
    :returns: list of transformed Lorentz vectors
    :rtype: ps1

    """
    newaxes = RotatedAxes(pb)
    eb = TimeComponent(pb)
    zeros = Zeros(eb)
    # Boost vector in the rotated coordinates along z axis
    boost = Vector(zeros, zeros, -Norm(SpatialComponents(pb))/eb)

    return ApplyRotationAndBoost(ps, newaxes, boost)


def ApplyRotationAndBoost(ps, axes, boost):
    """Helper function for RotationAndBoost. It applies RotationAndBoost iteratively on nested lists

    :param ps: 
    :param axes: 
    :param boost: 

    """
    (x, y, z) = axes
    ps1 = []
    for p in ps:
        if isinstance(p, list):
            p2 = ApplyRotationAndBoost(p, (x, y, z), boost)
        else:
            p1 = ProjectLorentzVector(p, (x, y, z))
            p2 = LorentzBoost(p1, boost)
        ps1 += [p2]
    return ps1


def EulerAngles(x1, y1, z1, x2, y2, z2):
    """Calculate Euler angles (phi, theta, psi in the ZYZ convention) which transform the coordinate basis (x1, y1, z1)
      to the basis (x2, y2, z2). Both x1,y1,z1 and x2,y2,z2 are assumed to be orthonormal and right-handed.

    :param x1: 
    :param y1: 
    :param z1: 
    :param x2: 
    :param y2: 
    :param z2: 

    """
    theta = Acos(ScalarProduct(z1, z2))
    phi = Atan2(ScalarProduct(z1, y2), ScalarProduct(z1, x2))
    psi = Atan2(ScalarProduct(y1, z2), ScalarProduct(x1, z2))
    return (phi, theta, psi)


def HelicityAngles3Body(pa, pb, pc):
    """Calculate 4 helicity angles for the 3-body D->ABC decay defined as:
      theta_r, phi_r : polar and azimuthal angles of the AB resonance in the D rest frame
      theta_a, phi_a : polar and azimuthal angles of the A in AB rest frame

    :param pa: 
    :param pb: 
    :param pc: 

    """
    theta_r = Acos(-ZComponent(pc) / Norm(SpatialComponents(pc)))
    phi_r = Atan2(-YComponent(pc), -XComponent(pc))

    pa_prime = LorentzVector(RotateVector(SpatialComponents(
        pa), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pa))
    pb_prime = LorentzVector(RotateVector(SpatialComponents(
        pb), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pb))

    w = TimeComponent(pa) + TimeComponent(pb)

    pab = LorentzVector(-(pa_prime + pb_prime)/Scalar(w), w)
    pa_prime2 = LorentzBoost(pa_prime, pab)

    theta_a = Acos(ZComponent(pa_prime2) / Norm(SpatialComponents(pa_prime2)))
    phi_a = Atan2(YComponent(pa_prime2), XComponent(pa_prime2))

    return (theta_r, phi_r, theta_a, phi_a)


def CosHelicityAngle(p1, p2):
    """The helicity angle is defined as the angle between one of the two momenta in the p1+p2 rest frame
      with respect to the momentum of the p1+p2 system in the decaying particle rest frame (ptot)

    :param p1: 
    :param p2: 

    """
    p12 = LorentzVector(SpatialComponents(p1)+SpatialComponents(p2),
                        TimeComponent(p1)+TimeComponent(p2))
    pcm1 = BoostToRest(p1, p12)
    cosHel = ScalarProduct(UnitVector(SpatialComponents(
        pcm1)), UnitVector(SpatialComponents(p12)))
    return cosHel


def Azimuthal4Body(p1, p2, p3, p4):
    """Calculates the angle between the plane defined by (p1,p2) and (p3,p4)

    :param p1: 
    :param p2: 
    :param p3: 
    :param p4: 

    """
    v1 = SpatialComponents(p1)
    v2 = SpatialComponents(p2)
    v3 = SpatialComponents(p3)
    v4 = SpatialComponents(p4)
    n12 = UnitVector(VectorProduct(v1, v2))
    n34 = UnitVector(VectorProduct(v3, v4))
    z = UnitVector(v1+v2)
    cosPhi = ScalarProduct(n12, n34)
    sinPhi = ScalarProduct(VectorProduct(n12, n34), z)
    phi = Atan2(sinPhi, cosPhi)  # defined in [-pi,pi]
    return phi


def HelicityAngles4Body(pa, pb, pc, pd):
    """Calculate 4 helicity angles for the 4-body E->ABCD decay defined as:
      theta_ab, phi_ab : polar and azimuthal angles of the AB resonance in the E rest frame
      theta_cd, phi_cd : polar and azimuthal angles of the CD resonance in the E rest frame
      theta_ac, phi_ac : polar and azimuthal angles of the AC resonance in the E rest frame
      theta_bd, phi_bd : polar and azimuthal angles of the BD resonance in the E rest frame
      theta_ad, phi_ad : polar and azimuthal angles of the AD resonance in the E rest frame
      theta_bc, phi_bc : polar and azimuthal angles of the BC resonance in the E rest frame
      phi_ab_cd : azimuthal angle between AB and CD
      phi_ac_bd : azimuthal angle between AC and BD
      phi_ad_bc : azimuthal angle between AD and BC

    :param pa: 
    :param pb: 
    :param pc: 
    :param pd: 

    """
    theta_r = Acos(-ZComponent(pc) / Norm(SpatialComponents(pc)))
    phi_r = Atan2(-YComponent(pc), -XComponent(pc))

    pa_prime = LorentzVector(RotateVector(SpatialComponents(
        pa), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pa))
    pb_prime = LorentzVector(RotateVector(SpatialComponents(
        pb), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pb))

    w = TimeComponent(pa) + TimeComponent(pb)

    pab = LorentzVector(-(pa_prime + pb_prime)/Scalar(w), w)
    pa_prime2 = LorentzBoost(pa_prime, pab)

    theta_a = Acos(ZComponent(pa_prime2) / Norm(SpatialComponents(pa_prime2)))
    phi_a = Atan2(YComponent(pa_prime2), XComponent(pa_prime2))

    return (theta_r, phi_r, theta_a, phi_a)


def WignerD(phi, theta, psi, j, m1, m2):
    """Calculate Wigner capital-D function.
      phi,
      theta,
      psi  : Rotation angles
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2, e.g. 1 for projection 1/2)

    :param phi: 
    :param theta: 
    :param psi: 
    :param j2: 
    :param m2_1: 
    :param m2_2: 

    """
    i = Complex(Const(0), Const(1))
    return Exp(-i*CastComplex(m1/2.*phi))*CastComplex(Wignerd(theta, j, m1, m2))*Exp(-i*CastComplex(m2/2.*psi))


def Wignerd(theta, j, m1, m2):
    """Calculate Wigner small-d function. Needs sympy.
      theta : angle
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2)

    :param theta: 
    :param j: 
    :param m1: 
    :param m2: 

    """
    from sympy import Rational
    from sympy.abc import x
    from sympy.utilities.lambdify import lambdify
    from sympy.physics.quantum.spin import Rotation as Wigner
    d = Wigner.d(Rational(j, 2), Rational(m1, 2),
                 Rational(m2, 2), x).doit().evalf()
    return lambdify(x, d, "tensorflow")(theta)


def Legendre(n, var):
    """Calculate Legendre_n(var)
      theta : angle

    :param n: 
    :param var: 

    """
    from sympy import Rational
    from sympy.abc import x
    from sympy.utilities.lambdify import lambdify
    from sympy import legendre
    l = legendre(Rational(n), x)
    return lambdify(x, l, "tensorflow")(var)


def SpinRotationAngle(pa, pb, pc, bachelor=2):
    """Calculate the angle between two spin-quantisation axes for the 3-body D->ABC decay
      aligned along the particle B and particle A.
        pa, pb, pc : 4-momenta of the final-state particles
        bachelor : index of the "bachelor" particle (0=A, 1=B, or 2=C)

    :param pa: 
    :param pb: 
    :param pc: 
    :param bachelor:  (Default value = 2)

    """
    if bachelor == 2:
        return Const(0.)
    pboost = LorentzVector(-SpatialComponents(pb) /
                           Scalar(TimeComponent(pb)), TimeComponent(pb))
    if bachelor == 0:
        pa1 = SpatialComponents(LorentzBoost(pa, pboost))
        pc1 = SpatialComponents(LorentzBoost(pc, pboost))
        return Acos(ScalarProduct(pa1, pc1)/Norm(pa1)/Norm(pc1))
    if bachelor == 1:
        pac = pa + pc
        pac1 = SpatialComponents(LorentzBoost(pac, pboost))
        pa1 = SpatialComponents(LorentzBoost(pa, pboost))
        return Acos(ScalarProduct(pac1, pa1)/Norm(pac1)/Norm(pa1))
    return None


def HelicityAmplitude3Body(thetaR, phiR, thetaA, phiA, spinD, spinR, mu, lambdaR, lambdaA, lambdaB, lambdaC, cache=False):
    """Calculate complex helicity amplitude for the 3-body decay D->ABC
      thetaR, phiR : polar and azimuthal angles of AB resonance in D rest frame
      thetaA, phiA : polar and azimuthal angles of A in AB rest frame
      spinD : D spin
      spinR : spin of the intermediate R resonance
      mu : D spin projection onto z axis
      lambdaR : R resonance helicity
      lambdaA : A helicity
      lambdaB : B helicity
      lambdaC : C helicity

    :param thetaR: 
    :param phiR: 
    :param thetaA: 
    :param phiA: 
    :param spinD: 
    :param spinR: 
    :param mu: 
    :param lambdaR: 
    :param lambdaA: 
    :param lambdaB: 
    :param lambdaC: 
    :param cache:  (Default value = False)

    """

    lambda1 = lambdaR - lambdaC
    lambda2 = lambdaA - lambdaB
    ph = (mu-lambda1)/2.*phiR + (lambdaR-lambda2)/2.*phiA
    d_terms = Wignerd(thetaR, spinD, mu, lambda1) * \
        Wignerd(thetaA, spinR, lambdaR, lambda2)
    h = Complex(d_terms*Cos(ph), d_terms*Sin(ph))

    if cache:
        AddCacheableTensor(h)

    return h


def HelicityCouplingsFromLS(ja, jb, jc, lb, lc, bls):
    """Helicity couplings from a list of LS couplings.
        ja : spin of A (decaying) particle
        jb : spin of B (1st decay product)
        jc : spin of C (2nd decay product)
        lb : B helicity
        lc : C helicity
        bls : dictionary of LS couplings, where:
          keys are tuples corresponding to (L,S) pairs
          values are values of LS couplings
      Note that ALL j,l,s should be doubled, e.g. S=1 for spin-1/2, L=2 for P-wave etc.

    :param ja: 
    :param jb: 
    :param jc: 
    :param lb: 
    :param lc: 
    :param bls: 

    """
    a = 0.
    # print("%d %d %d %d %d" % (ja, jb, jc, lb, lc))
    for ls, b in bls.items():
        l = ls[0]
        s = ls[1]
        coeff = math.sqrt((l+1)/(ja+1))*Clebsch(jb, lb, jc,   -lc,  s, lb-lc)*\
                                        Clebsch( l,  0, s,  lb-lc, ja, lb-lc)

        # print ("   %d %d %f %f %f" % (l, s, coeff, Clebsch(jb, lb, jc, -lc, s, lb-lc), Clebsch(l, 0, s, lb-lc, ja, lb-lc) ) )

        if coeff : a += Complex(Const(float(coeff)), Const(0.))*b
    return a


def ZemachTensor(m2ab, m2ac, m2bc, m2d, m2a, m2b, m2c, spin, cache=False):
    """Zemach tensor for 3-body D->ABC decay

    :param m2ab: 
    :param m2ac: 
    :param m2bc: 
    :param m2d: 
    :param m2a: 
    :param m2b: 
    :param m2c: 
    :param spin: 
    :param cache:  (Default value = False)

    """
    z = None
    if spin == 0:
        z = Complex(Const(1.), Const(0.))
    if spin == 1:
        z = Complex(m2ac-m2bc+(m2d-m2c)*(m2b-m2a)/m2ab, Const(0.))
    if spin == 2:
        z = Complex((m2bc-m2ac+(m2d-m2c)*(m2a-m2b)/m2ab)**2-1./3.*(m2ab-2.*(m2d+m2c) +
                                                                   (m2d-m2c)**2/m2ab)*(m2ab-2.*(m2a+m2b)+(m2a-m2b)**2/m2ab), Const(0.))
    if cache:
        AddCacheableTensor(z)

    return z


def TwoBodyMomentum(md, ma, mb):
    """Momentum of two-body decay products D->AB in the D rest frame

    :param md: 
    :param ma: 
    :param mb: 

    """
    return Sqrt((md**2-(ma+mb)**2)*(md**2-(ma-mb)**2)/(4*md**2))


def ComplexTwoBodyMomentum(md, ma, mb):
    """Momentum of two-body decay products D->AB in the D rest frame.
      Output value is a complex number, analytic continuation for the
      region below threshold.

    :param md: 
    :param ma: 
    :param mb: 

    """
    return Sqrt(Complex((md**2-(ma+mb)**2)*(md**2-(ma-mb)**2)/(4*md**2), Const(0.)))


def FindBasicParticles(particle):
    """

    :param particle: 

    """
    if particle.GetNDaughters() == 0:
        return [particle]
    basic_particles = []
    for dau in particle.GetDaughters():
        basic_particles += FindBasicParticles(dau)
    return basic_particles


def AllowedHelicities(particle):
    """

    :param particle: 

    """
    return range(-particle.GetSpin2(), particle.GetSpin2(), 2)+[particle.GetSpin2()]


def HelicityMatrixDecayChain(parent, helAmps):
    """

    :param parent: 
    :param helAmps: 

    """
    matrix_parent = HelicityMatrixElement(parent, helAmps)
    daughters = parent.GetDaughters()
    if all(dau.GetNDaughters() == 0 for dau in daughters):
        return matrix_parent

    heldaug = list(itertools.product(
        *[AllowedHelicities(dau) for dau in daughters]))

    d1basics = FindBasicParticles(daughters[0])
    d2basics = FindBasicParticles(daughters[1])
    d1helbasics = list(itertools.product(
        *[AllowedHelicities(bas) for bas in d1basics]))
    d2helbasics = list(itertools.product(
        *[AllowedHelicities(bas) for bas in d2basics]))

    # matrix_dau = [HelicityMatrixDecayChain(dau,helAmps) for dau in daughters if dau.GetNDaughters()!=0 else {(d2hel,)+d2helbasic: c for d2hel,d2helbasic in itertools.product(AllowedHelicites(dau),d2helbasics)}]
    # matrix_dau=[]
    # for dau in daughters:
    #  if dau.GetNDaughters()!=0:
    #    matrix_dau.append( HelicityMatrixDecayChain(dau,helAmps) )
    matrix_dau = [HelicityMatrixDecayChain(
        dau, helAmps) for dau in daughters if dau.GetNDaughters() != 0]

    matrix = {}
    for phel, d1helbasic, d2helbasic in itertools.product(AllowedHelicities(parent), d1helbasics, d2helbasics):
        if len(matrix_dau) == 2:
            matrix[(phel,)+d1helbasic+d2helbasic] = sum([matrix_parent[(phel, d1hel, d2hel)]*matrix_dau[0][(d1hel,)+d1helbasic]
                                                         * matrix_dau[1][(d2hel,)+d2helbasic] for d1hel, d2hel in heldaug if abs(parent.GetSpin2()) >= abs(d1hel-d2hel)])
        elif daughters[0].GetNDaughters() != 0:
            matrix[(phel,)+d1helbasic+d2helbasic] = sum([matrix_parent[(phel, d1hel, d2hel)]*matrix_dau[0]
                                                         [(d1hel,)+d1helbasic] for d1hel, d2hel in heldaug if abs(parent.GetSpin2()) >= abs(d1hel-d2hel)])
        else:
            matrix[(phel,)+d1helbasic+d2helbasic] = sum([matrix_parent[(phel, d1hel, d2hel)]*matrix_dau[0]
                                                         [(d2hel,)+d2helbasic] for d1hel, d2hel in heldaug if abs(parent.GetSpin2()) >= abs(d1hel-d2hel)])
    return matrix


def HelicityMatrixElement(parent, helAmps):
    """

    :param parent: 
    :param helAmps: 

    """
    if parent.GetNDaughters() != 2:
        sys.exit('ERROR in HelicityMatrixElement, the parent ' +
                 parent.GetName()+' has no 2 daughters')

    matrixelement = {}
    [d1, d2] = parent.GetDaughters()
    parent_helicities = AllowedHelicities(parent)
    d1_helicities = AllowedHelicities(d1)
    d2_helicities = AllowedHelicities(d2)

    if parent.IsParityConserving():
        if not all(part.GetParity() in [-1, +1] for part in [parent, d1, d2]):
            sys.exit('ERROR in HelicityMatrixElement for the decay of particle '+parent.GetName() +
                     ', the parities have to be correctly defined (-1 or +1) for the particle and its daughters')

        parity_factor = parent.GetParity()*d1.GetParity()*d2.GetParity() * \
            (-1)**((d1.GetSpin2()+d2.GetSpin2()-parent.GetSpin2())/2)
        if 0 in d1_helicities and 0 in d2_helicities and parity_factor == -1 \
           and helAmps[parent.GetName()+'_'+d1.GetName()+'_0_'+d2.GetName()+'_0'] != 0:
            sys.exit('ERROR in HelicityMatrixElement, the helicity amplitude '
                     + parent.GetName()+'_'+d1.GetName()+'_0_'+d2.GetName()+'_0 should be set to 0 for parity conservation reason')

    theta = d1.Theta()
    phi = d1.Phi()
    for phel, d1hel, d2hel in itertools.product(parent_helicities, d1_helicities, d2_helicities):
        if parent.GetSpin2() < abs(d1hel-d2hel):
            continue
        d1hel_str = ('+' if d1hel > 0 else '')+str(d1hel)
        d2hel_str = ('+' if d2hel > 0 else '')+str(d2hel)
        flipped = False
        if parent.IsParityConserving() and (d1hel != 0 or d2hel != 0):
            d1hel_str_flip = ('+' if -d1hel > 0 else '')+str(-d1hel)
            d2hel_str_flip = ('+' if -d2hel > 0 else '')+str(-d2hel)
            helAmp_str = parent.GetName()+'_'+d1.GetName()+'_'+d1hel_str + \
                '_'+d2.GetName()+'_'+d2hel_str
            helAmp_str_flip = parent.GetName()+'_'+d1.GetName()+'_'+d1hel_str_flip + \
                '_'+d2.GetName()+'_'+d2hel_str_flip
            if helAmp_str in helAmps.keys() and helAmp_str_flip in helAmps.keys():
                sys.exit('ERROR in HelicityMatrixElement: particle '+parent.GetName() +
                         ' conserves parity in decay but both '+helAmp_str+' and '+helAmp_str_flip +
                         ' are declared. Only one has to be declared, the other is calculated from parity conservation')
            if helAmp_str_flip in helAmps.keys():
                d1hel_str = d1hel_str_flip
                d2hel_str = d2hel_str_flip
                flipped = True
        helAmp = (parity_factor if parent.IsParityConserving() and flipped else 1.) * \
            helAmps[parent.GetName()+'_'+d1.GetName()+'_'+d1hel_str +
                    '_'+d2.GetName()+'_'+d2hel_str]
        matrixelement[(phel, d1hel, d2hel)] = parent.GetShape()*helAmp\
            * Conjugate(WignerD(phi, theta, 0, parent.GetSpin2(), phel, d1hel-d2hel))
    return matrixelement


def RotateFinalStateHelicity(matrixin, particlesfrom, particlesto):
    """

    :param matrixin: 
    :param particlesfrom: 
    :param particlesto: 

    """
    if not all(part1.GetSpin2() == part2.GetSpin2() for part1, part2 in zip(particlesfrom, particlesto)):
        sys.exit(
            'ERROR in RotateFinalStateHelicity, found a mismatch between the spins given to RotateFinalStateHelicity')
    matrixout = {}
    for hels in matrixin.keys():
        matrixout[hels] = 0
    heldaugs = []
    axesfrom = [RotatedAxes(part.GetMomentum(), oldaxes=part.GetAxes())
                for part in particlesfrom]
    axesto = [RotatedAxes(part.GetMomentum(), oldaxes=part.GetAxes())
              for part in particlesto]
    thetas = [Acos(ScalarProduct(axisfrom[2], axisto[2]))
              for axisfrom, axisto in zip(axesfrom, axesto)]
    phis = [Atan2(ScalarProduct(axisfrom[1], axisto[0]), ScalarProduct(
        axisfrom[0], axisto[0])) for axisfrom, axisto in zip(axesfrom, axesto)]

    rot = []
    for part, theta, phi in zip(particlesfrom, thetas, phis):
        allhels = AllowedHelicities(part)
        rot.append({})
        for helfrom, helto in itertools.product(allhels, allhels):
            rot[-1][(helfrom, helto)] = Conjugate(WignerD(phi,
                                                          theta, 0, part.GetSpin2(), helfrom, helto))

    for helsfrom in matrixin.keys():
        daughelsfrom = helsfrom[1:]
        for helsto in matrixout.keys():
            daughelsto = helsto[1:]
            prod = reduce(operator.mul, [rot[i][(
                hpfrom, hpto)] for i, (hpfrom, hpto) in enumerate(zip(daughelsfrom, daughelsto))])
            matrixout[helsto] += prod*matrixin[helsfrom]

    return matrixout


class Particle:
    """Class to describe a Particle"""

    def __init__(self, name='default', shape=None, spin2=0, momentum=None, daughters=[], parityConserving=False, parity=None):
        self._name = name
        self._spin2 = spin2
        self._daughters = daughters
        self._shape = shape if shape != None else CastComplex(Const(1.))
        if momentum != None and daughters != []:
            sys.exit('ERROR in Particle '+name +
                     ' definition: do not define the momentum, it is taken from the sum of the daughters momenta!')
        self._momentum = momentum if momentum != None and daughters == [
        ] else sum([dau.GetMomentum() for dau in daughters])
        self._parityConserving = parityConserving
        self._parity = parity
        emom = TimeComponent(self._momentum)
        zeros = Zeros(emom)
        ones = Ones(emom)
        self._axes = (Vector(ones, zeros, zeros),
                      Vector(zeros, ones, zeros),
                      Vector(zeros, zeros, ones))

    def GetName(self): return self._name
    def GetSpin2(self): return self._spin2
    def GetDaughters(self): return self._daughters
    def GetNDaughters(self): return len(self._daughters)
    def GetShape(self): return self._shape
    def GetMomentum(self): return self._momentum
    def GetAxes(self): return self._axes
    def IsParityConserving(self): return self._parityConserving
    def GetParity(self): return self._parity
    def SetName(self, newname): self._name = newname
    def SetSpin(self, newspin): self._spin2 = newspin
    def SetShape(self, newshape): self._shape = newshape
    def SetMomentum(self, momentum): self._momentum = momentum
    def SetParity(self, parity): self._parity = parity

    def Theta(self):
        """

        """
        return Acos(ScalarProduct(UnitVector(SpatialComponents(self._momentum)), self._axes[2]))

    def Phi(self):
        """ """
        x = self._axes[0]
        y = self._axes[1]
        return Atan2(ScalarProduct(UnitVector(SpatialComponents(self._momentum)), y), ScalarProduct(UnitVector(SpatialComponents(self._momentum)), x))

    def ApplyRotationAndBoost(self, newaxes, boost):
        """

        :param newaxes: 
        :param boost: 

        """
        self._axes = newaxes
        self._momentum = LorentzBoost(self._momentum, boost)
        for dau in self._daughters:
            dau.ApplyRotationAndBoost(newaxes, boost)

    def RotateAndBoostDaughters(self, isAtRest=True):
        """

        :param isAtRest:  (Default value = True)

        """
        if not isAtRest:
            newaxes = RotatedAxes(self._momentum, oldaxes=self._axes)
            eb = TimeComponent(self._momentum)
            zeros = Zeros(eb)
            boost = -SpatialComponents(self._momentum)/Scalar(eb)
            #boost = newaxes[2]*(-Norm(SpatialComponents(self._momentum))/eb)
            #boost = Vector(zeros, zeros, -Norm(SpatialComponents(self._momentum))/eb)
            for dau in self._daughters:
                dau.ApplyRotationAndBoost(newaxes, boost)
        for dau in self._daughters:
            dau.RotateAndBoostDaughters(isAtRest=False)

    def __eq__(self, other):
        eq = (self._name == other._name)
        eq &= (self._spin2 == other._spin2)
        eq &= (self._shape == other._shape)
        eq &= (self._momentum == other._momentum)
        eq &= (self._daughters == other._daughters)
        return eq


class PHSPGenerator:
    def __init__(self, m_mother,  m_daughters):
        """Constructor"""
        self.ndaughters = len(m_daughters)
        self.m_daughters = m_daughters
        self.m_mother = m_mother

    def RandomOrdered(self, nev):
        """

        :param nev: 

        """
        return (-1)*tf.nn.top_k(-tf.random_uniform([nev, self.ndaughters-2], dtype=tf.float64), k=self.ndaughters-2).values

    def GenerateFlatAngles(self, nev):
        """

        :param nev: 

        """
        return (tf.random_uniform([nev], minval=-1., maxval=1., dtype=tf.float64),
                tf.random_uniform([nev], minval=-math.pi, maxval=math.pi, dtype=tf.float64))

    def RotateLorentzVector(self, p, costheta, phi):
        """

        :param p: 
        :param costheta: 
        :param phi: 

        """
        pvec = SpatialComponents(p)
        energy = TimeComponent(p)
        pvecrot = self.RotateVector(pvec, costheta, phi)
        return LorentzVector(pvecrot, energy)

    def RotateVector(self, vec, costheta, phi):
        """

        :param vec: 
        :param costheta: 
        :param phi: 

        """
        cZ = costheta
        sZ = Sqrt(1-cZ**2)
        cY = Cos(phi)
        sY = Sin(phi)
        x = XComponent(vec)
        y = YComponent(vec)
        z = ZComponent(vec)
        xnew = cZ*x-sZ*y
        ynew = sZ*x+cZ*y
        xnew2 = cY*xnew-sY*z
        znew = sY*xnew+cY*z
        return Vector(xnew2, ynew, znew)

    def GenerateModel(self, nev):
        """

        :param nev: 

        """
        rands = self.RandomOrdered(nev)
        delta = self.m_mother-sum(self.m_daughters)

        sumsubmasses = []
        for i in range(self.ndaughters-2):
            sumsubmasses.append(sum(self.m_daughters[:(i+2)]))
        SubMasses = rands*delta + sumsubmasses
        SubMasses = tf.concat(
            [SubMasses, Scalar(self.m_mother*tf.ones([nev], dtype=tf.float64))], axis=1)
        pout = []
        weights = tf.ones([nev], dtype=tf.float64)
        for i in range(self.ndaughters-1):
            submass = tf.unstack(SubMasses, axis=1)[i]
            zeros = Zeros(submass)
            ones = Ones(submass)
            if i == 0:
                MassDaughterA = self.m_daughters[i]*ones
                MassDaughterB = self.m_daughters[i+1]*ones
            else:
                MassDaughterA = tf.unstack(SubMasses, axis=1)[i-1]
                MassDaughterB = self.m_daughters[i+1]*Ones(MassDaughterA)
            pMag = TwoBodyMomentum(submass, MassDaughterA, MassDaughterB)
            (costheta, phi) = self.GenerateFlatAngles(nev)
            vecArot = self.RotateVector(
                Vector(zeros, pMag, zeros), costheta, phi)
            pArot = LorentzVector(vecArot, Sqrt(MassDaughterA**2+pMag**2))
            pBrot = LorentzVector(-vecArot, Sqrt(MassDaughterB**2+pMag**2))
            pout = [LorentzBoost(p, SpatialComponents(
                pArot)/Scalar(TimeComponent(pArot))) for p in pout]
            if i == 0:
                pout.append(pArot)
                pout.append(pBrot)
            else:
                pout.append(pBrot)
            weights = tf.multiply(weights, pMag)
        moms = tf.concat(pout, axis=1)
        phsp_model = tf.concat([moms, Scalar(weights)], axis=1)
        return phsp_model
