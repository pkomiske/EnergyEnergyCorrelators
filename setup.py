# EnergyEnergyCorrelators - Evaluates EECs on particle physics events
# Copyright (C) 2020 Patrick T. Komiske III
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import platform
import re
import subprocess
import sys

from setuptools import setup
from setuptools.extension import Extension

import numpy as np

serialization = (platform.system() != 'Windows')

with open(os.path.join('eec', '__init__.py'), 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

cxxflags = ['-fopenmp', '-std=c++14', '-ffast-math', '-g0', '-DEEC_SERIALIZATION', '-DEEC_COMPRESSION'][:(6 if serialization else 4)]
ldflags = ['-fopenmp']
libs = ['boost_serialization', 'boost_iostreams'][:(2 if serialization else 0)]
include_dirs = [np.get_include(), os.path.join('eec', 'include')]
if platform.system() == 'Darwin':
    cxxflags.insert(0, '-Xpreprocessor')
    del ldflags[0]
    libs.append('omp')
elif platform.system() == 'Windows':
    ldflags[0] = '/openmp'
    cxxflags = ['/openmp', '/std:c++14', '/fp:fast' '/DEEC_SERIALIZATION', '/DEEC_COMPRESSION'][:(5 if serialization else 3)]
    include_dirs.append('.')

if sys.argv[1] == 'swig':
    swig_opts = ['-fastproxy', '-w511', '-keyword', '-Ieec/include']
    if len(sys.argv) >= 3 and sys.argv[2] == '-py3':
        swig_opts.append('-py3')
    command = 'swig -python -c++ {} -o eec/eec.cpp eec/swig/eec.i'.format(' '.join(swig_opts))
    print(command)
    subprocess.run(command.split())

else:
    eec = Extension('eec._eec',
                    sources=[os.path.join('eec', 'eec.cpp')],
                    include_dirs=include_dirs,
                    extra_compile_args=cxxflags,
                    extra_link_args=ldflags,
                    libraries=libs)

    setup(
        ext_modules=[eec],
        version=__version__
    )