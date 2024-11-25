from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import os
#Get the path of the installed package
path = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension("D0ToKspipi2018",
              sources=[f"{path}/D0ToKspipi2018.pyx"], language="c++") ]

setup(
    name="D0ToKspipi2018",
    ext_modules=cythonize(extensions),
)
