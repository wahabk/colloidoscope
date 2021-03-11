import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11']
ext_modules = [
    Extension(
    'wrap',
        ['mainHScrusherPoly.cpp', 'wrap.cpp'],
        include_dirs=['../extern/pybind11/include'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='wrap',
    version='0.0.1',
    author='Wahab Kawafi',
    author_email='nan',
    description='paddy wrapper',
    ext_modules=ext_modules,
)