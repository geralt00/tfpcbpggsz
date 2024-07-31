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

import tensorflow as tf
import numpy as np
import itertools

fptype = tf.float64
ctype = tf.complex128


def SetSinglePrecision():
    global fptype, ctype
    fptype = tf.float32
    ctype = tf.complex64


def SetDoublePrecision():
    global fptype, ctype
    fptype = tf.float64
    ctype = tf.complex128


def FPType():
    global fptype
    return fptype


def CType():
    global ctype
    return ctype


def Sum(ampls):
    """ Sum of the list components """
    return tf.add_n(ampls)


def Density(ampl):
    """ Density for a complex amplitude """
    return tf.abs(ampl)**2


def Abs(a):
    """ Absolute value """
    return tf.abs(a)


def Max(x, y):
    return tf.maximum(x, y)


def Min(x, y):
    return tf.minimum(x, y)


def Complex(re, im):
    """ Create a complex number from real and imaginary parts """
    return tf.complex(re, im)


def Polar(a, ph):
    """ Create a complex number from a magnitude and a phase """
    return tf.complex(a*tf.cos(ph), a*tf.sin(ph))


def CastComplex(re):
    """ Cast a real number to complex """
    return tf.cast(re, dtype=CType())


def CastReal(re):
    """ Cast a number to real """
    return tf.cast(re, dtype=FPType())


def Conjugate(c):
    """ Return complex conjugate """
    return tf.conj(c)


def Real(c):
    """ Return the real part of a complex number """
    return tf.real(c)


def Imaginary(c):
    """ Return the imaginary part of a complex number """
    return tf.imag(c)


def Const(c):
    """ Declare constant """
    return tf.constant(c, dtype=FPType())


def Invariant(c):
    """ Declare invariant """
    return tf.constant([c], dtype=FPType())


def Sqrt(x):
    return tf.sqrt(x)


def Exp(x):
    return tf.exp(x)


def Log(x):
    return tf.log(x)


def Sin(x):
    return tf.sin(x)


def Cos(x):
    return tf.cos(x)


def Asin(x):
    return tf.asin(x)


def Atan(x):
    return tf.atan(x)


def Acos(x):
    return tf.acos(x)


def Tanh(x):
    return tf.tanh(x)


def Pow(x, p):
    return tf.pow(x, p)


def Pi():
    return Const(np.pi)


def Zeros(x):
    """ Create a tensor with zeros of the same shape as input tensor """
    return tf.zeros_like(x)


def Ones(x):
    """ Create a tensor with ones of the same shape as input tensor """
    return tf.ones_like(x)


def Atan2(y, x):
    return tf.atan2(y, x)


def Argument(c):
    """ Return argument (phase) of a complex number """
    return Atan2(tf.imag(c), tf.real(c))


def Clebsch(j1, m1, j2, m2, J, M):
    """
      Return Clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
      (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
    """
    from sympy.physics.quantum.cg import CG
    from sympy import Rational
    return CG(Rational(j1, 2), Rational(m1, 2), Rational(j2, 2), Rational(m2, 2), Rational(J, 2), Rational(M, 2)).doit().evalf()


def Interpolate(t, c):
    """
      Multilinear interpolation on a rectangular grid of arbitrary number of dimensions
        t : TF tensor representing the grid (of rank N)
        c : Tensor of coordinates for which the interpolation is performed
        return: 1D tensor of interpolated values
    """
    rank = len(t.get_shape())
    ind = tf.cast(tf.floor(c), tf.int32)
    t2 = tf.pad(t, rank*[[1, 1]], 'SYMMETRIC')
    wts = []
    for vertex in itertools.product([0, 1], repeat=rank):
        ind2 = ind + tf.constant(vertex, dtype=tf.int32)
        weight = tf.reduce_prod(
            1. - tf.abs(c - tf.cast(ind2, dtype=FPType())), 1)
        wt = tf.gather_nd(t2, ind2+1)
        wts += [weight*wt]
    interp = tf.reduce_sum(tf.stack(wts), 0)
    return interp


def SetSeed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


def Random(size):
    """
      Generate array of random numbers, uniform from 0 to 1
    """
    return np.random.uniform(0., 1., size).astype('d')
