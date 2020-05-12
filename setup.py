from setuptools import setup
import os
import re

from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension

# determine version
with open('eec/__init__.py', 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

path = os.path.abspath(os.path.dirname(__file__))
extensions = [
    Extension(
        'eec.core',
        sources=['eec/core.pyx'],
        language='c++',
        include_dirs=[np.get_include(), os.path.join(path, 'eec/include')],
        libraries=[],
        extra_compile_args=['-std=c++14']
    )
]

extensions = cythonize(extensions, 
                       compiler_directives={'language_level': 3, 
                                            'boundscheck': False, 
                                            'wraparound': False,
                                            'cdivision': True},
                       annotate=False)

# other options specified in setup.cfg
setup(
    version=__version__,
    ext_modules=extensions
)