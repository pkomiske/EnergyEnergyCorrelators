import os
import re

from Cython.Build import cythonize
import numpy as np
from setuptools import setup
from setuptools.extension import Extension

# determine version
with open('eec/__init__.py', 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

###############################################################################
# Helper functions
###############################################################################

# this function is used if we ever want to distribute pre-cythonized files
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

###############################################################################
# Specify extension
###############################################################################

path = os.path.abspath(os.path.dirname(__file__))
extensions = [
    Extension(
        'eec.core',
        sources=[os.path.join('eec', 'core.pyx')],
        language='c++',
        include_dirs=[np.get_include(), os.path.join(path, 'eec', 'include')],
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
    ext_modules=extensions,
    package_data={'': ['*.pyx', '*.pxd'] + package_files(os.path.join('eec', 'include'))}
)
