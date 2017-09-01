#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

depend_libs = [
    'kaldi-base', 
    'kaldi-util', 
    'kaldi-matrix',
    'kaldi-cudamatrix', 
    'kaldi-nnet3', 
    'kaldi-ivector',
    'kaldi-decoder', 
    'kaldi-lat', 
    'kaldi-fstext', 
    'kaldi-hmm', 
    'kaldi-feat', 
    'kaldi-tree', 
    'kaldi-gmm',
    'kaldi-transform', 
    'kaldi-online2',
    'fst'
]

include_dirs = [
    'kaldi/src/', 
    'openfst/include', 
    'openblas/install/include', 
    np.get_include()
]

library_dirs = [
    'kaldi/src/lib', 
    'openfst/lib'
]

kaldi_complie_args = [
    '-std=c++11', 
    '-DHAVE_OPENBLAS', 
    '-Wno-deprecated-declarations',
    '-Wno-sign-compare',
    '-Wno-unused-local-typedefs',
    '-Winit-self',
]

cxx_module = Extension('pydecoder',
                        language='c++',
                        extra_compile_args=kaldi_complie_args,
                        include_dirs=include_dirs,
                        library_dirs=library_dirs,
                        libraries=depend_libs,
                        sources=['../cc/py-online-nnet3-decoder.cc', 'pydecoder.pyx'],
                        )

setup(
    name='python-decoder-wrappers',
    ext_modules = cythonize([cxx_module]),
    description = 'python interface in kaldi-nnet3 decoding setup',
    author='wujian',
)