r"""

$$$$$$$$\ $$$$$$$$\  $$$$$$\
$$  _____|$$  _____|$$  __$$\
$$ |      $$ |      $$ /  \__|
$$$$$\    $$$$$\    $$ |
$$  __|   $$  __|   $$ |
$$ |      $$ |      $$ |  $$\
$$$$$$$$\ $$$$$$$$\ \$$$$$$  |
\________|\________| \______/

EnergyEnergyCorrelators - Evaluates EECs on particle physics events
Copyright (C) 2020-2021 Patrick T. Komiske III

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import

import numpy as _np

from . import eec
from .eec import *

# basic package info
__author__ = 'Patrick T. Komiske III'
__email__ = 'pkomiske@mit.edu'
__license__ = 'GPLv3'
__version__ = '1.1.0'

# Note that axis/axes is supported as well as axis_range/axes_range.
# The "axes" version is for compatibility with EECTriangleOPE and
# the values there are expected to be a one item list/tuple
def EECLongestSide(*args, axis='log', **kwargs):

    axis_range = kwargs.pop('axis_range', None)
    axes_range = kwargs.pop('axes_range', None)
    if axis_range is not None and axes_range is not None:
        raise ValueError('`axis_range` and `axes_range` cannot both be given')

    if axes_range is not None:
        assert len(axes_range) == 1, '`axes_range` must be length 1'
        axis_range = axes_range[0]

    if axis_range is not None:
        assert len(axis_range) == 2, '`axis_range` must be length 2'
        kwargs['axis_min'] = axis_range[0]
        kwargs['axis_max'] = axis_range[1]

    # validate axis options
    axes = kwargs.pop('axes', None)
    if axes is not None:
        assert len(axes) == 1, '`axes` must be length 1'
        axis = axes[0]

    if axis.lower() == 'log':
        return EECLongestSideLog(*args, **kwargs)
    elif axis.lower() == 'id':
        return EECLongestSideId(*args, **kwargs)
    else:
        raise ValueError('axis `{}` not understood'.format(axis))

# this accepts `axes` as a tuple/list of three strings and `axes_range`
# as a tuple/list of 3 pairs of values
def EECTriangleOPE(*args, axes=('log', 'log', 'id'), **kwargs):

    nbins = kwargs.pop('nbins', None)
    if nbins is not None:
        assert len(nbins) == 3, '`nbins` must be length 3'
        kwargs['nbins0'], kwargs['nbins1'], kwargs['nbins2'] = nbins

    axes_range = kwargs.pop('axes_range', None)
    if axes_range is not None:
        assert len(axes_range) == 3, '`axes_range` must be length 3'
        for i,axis_range in enumerate(axes_range):
            assert len(axis_range) == 2, 'axis_range ' + str(axis_range) + ' not length 2'
            kwargs['axis{}_min'.format(i)] = axis_range[0]
            kwargs['axis{}_max'.format(i)] = axis_range[1]

    axes = tuple(map(lambda x: x.lower(), axes))
    if axes == ('log', 'log', 'id'):
        return EECTriangleOPELogLogId(*args, **kwargs)
    elif axes == ('id', 'log', 'id'):
        return EECTriangleOPEIdLogId(*args, **kwargs)
    elif axes == ('log', 'id', 'id'):
        return EECTriangleOPELogIdId(*args, **kwargs)
    elif axes == ('id', 'id', 'id'):
        return EECTriangleOPEIdIdId(*args, **kwargs)
    else:
        raise ValueError('axes `{}` not understood'.format(axes))

# selects between EECLongestSide and EECTriangleOPE using a string `comp_type`
def EEC(comp_type, *args, **kwargs):
    if comp_type == 'EECLongestSide':
        return EECLongestSide(*args, **kwargs)

    if comp_type == 'EECTriangleOPE':
        return EECTriangleOPE(*args, **kwargs)

    raise ValueError('invalid comp_type `{}`'.format(comp_type))
