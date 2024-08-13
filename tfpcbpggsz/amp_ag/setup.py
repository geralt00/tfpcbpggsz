from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("D0ToKSpipi2018",
              sources=["D0ToKSpipi2018.pyx"], language="c++") ]

setup(
    name="D0ToKSpipi2018",
    ext_modules=cythonize(extensions),
)
