# EnergyEnergyCorrelators - Evaluates EECs on particle physics events
# Copyright (C) 2020-2021 Patrick T. Komiske III
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

use_fastjet = False

with open(os.path.join('eec', '__init__.py'), 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

# function to query a config binary and get the result
fastjet_config = os.environ.get('FASTJET_CONFIG', 'fastjet-config')
def query_config(query):
    return subprocess.check_output([fastjet_config, query]).decode('utf-8').strip()

# get fastjet info
fj_prefix = query_config('--prefix') if use_fastjet else ''
fj_cxxflags = query_config('--cxxflags') if use_fastjet else ''
fj_ldflags = query_config('--libs') if use_fastjet else ''

# run swig to generate eec.py and eec.cpp from eec.i
if sys.argv[1] == 'swig':
    swig_opts = ['-fastproxy', '-w509,511', '-keyword', '-Ieec/include']
    if len(sys.argv) >= 3 and sys.argv[2] == '-py3':
        swig_opts.append('-py3')
    if use_fastjet:
        swig_opts += ['-DSWIG_FASTJET', '-DFASTJET_PREFIX={}'.format(fj_prefix)] + fj_cxxflags.split()
    command = 'swig -python -c++ {} -o eec/eec.cpp eec/swig/eec.i'.format(' '.join(swig_opts))
    print(command)
    subprocess.run(command.split())

# build extension
else:

    # compiler flags, libraries, etc
    cxxflags = ['-fopenmp', '-std=c++14', '-ffast-math', '-g0'] + fj_cxxflags.split()
    ldflags = ['-fopenmp']
    libs = []
    include_dirs = [np.get_include(), os.path.join('eec', 'include')]
    if platform.system() == 'Darwin':
        cxxflags.insert(0, '-Xpreprocessor')
        del ldflags[0]
        libs.append('omp')
    elif platform.system() == 'Windows':
        assert not use_fastjet, 'fastjet not yet supported on windows'
        ldflags[0] = '/openmp'
        cxxflags = ['/openmp', '/std:c++14', '/fp:fast']
        include_dirs.append('.')

    # we only serialize on non-windows platforms
    serialization = (platform.system() != 'Windows')
    if serialization:
        cxxflags.extend(['-DEEC_SERIALIZATION', '-DEEC_COMPRESSION'])
        libs.extend(['boost_serialization', 'boost_iostreams', 'z'])

    # determine fastjet library paths and names for Python
    fj_libdirs = []
    for x in fj_ldflags.split():
        if x.startswith('-L'):
            fj_libdirs.append(x[2:])
        elif x.startswith('-l'):
            libs.append(x[2:])
        else:
            ldflags.append(x)

    if use_fastjet:
        include_dirs.append(os.path.join(fj_prefix, 'share', 'fastjet', 'pyinterface'))
        cxxflags += ['-DSWIG_FASTJET', '-DSWIG_TYPE_TABLE=fastjet']
    else:
        cxxflags.append('-DSWIG_TYPE_TABLE=eec')

    eec = Extension('eec._eec',
                    sources=[os.path.join('eec', 'eec.cpp')],
                    include_dirs=include_dirs,
                    library_dirs=fj_libdirs + ['/usr/local/lib'],
                    extra_compile_args=cxxflags,
                    extra_link_args=ldflags,
                    libraries=libs)

    setup(
        ext_modules=[eec],
        version=__version__
    )