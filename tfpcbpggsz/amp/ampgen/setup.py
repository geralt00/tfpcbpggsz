from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import os
#Get the path of the installed package
path = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension("D0ToKSpipi2018",
              include_dirs=[np.get_include()],
              extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
              extra_link_args=["-fopenmp"],
              sources=[f"{path}/D0ToKSpipi2018.pyx"], language="c++") ]

setup(
    name="D0ToKSpipi2018",
    ext_modules=cythonize(extensions),
)
