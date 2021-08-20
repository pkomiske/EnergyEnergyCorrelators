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

from . import eec
from .eec import *

# basic package info
__author__ = 'Patrick T. Komiske III'
__email__ = 'pkomiske@mit.edu'
__license__ = 'GPLv3'
__version__ = '2.0.0a3'

# Note that axis/axes is supported as well as axis_range/axes_range.
# The "axes" version is for compatibility with EECTriangleOPE and
# the values there are expected to be a one item list/tuple
def EECLongestSide(N, nbins, axis='log', **kwargs):

    axis_range = kwargs.pop('axis_range', None)
    axes_range = kwargs.pop('axes_range', None)
    if axis_range is not None and axes_range is not None:
        raise ValueError('`axis_range` and `axes_range` cannot both be given')

    if axes_range is not None:
        assert len(axes_range) == 1, '`axes_range` must be length 1'
        kwargs['axis_range'] = axes_range[0]

    if axis_range is not None:
        assert len(axis_range) == 2, '`axis_range` must be length 2'
        kwargs['axis_range'] = axis_range

    # validate axis options
    axes = kwargs.pop('axes', None)
    if axes is not None:
        assert 'axis' not in kwargs, '`axis` and `axes` cannot both be given'
        assert len(axes) == 1, '`axes` must be length 1'
        axis = axes[0]

    # allow integers for _powers arguments
    for key in ['weight_powers', 'charge_powers']:
        if key in kwargs and isinstance(kwargs[key], (int, float)):
            kwargs[key] = (kwargs[key],)

    if axis.lower() == 'log':
        return EECLongestSideLog(N, nbins, **kwargs)
    elif axis.lower() == 'id':
        return EECLongestSideId(N, nbins, **kwargs)
    else:
        raise ValueError('axis `{}` not understood'.format(axis))

# this accepts `axes` as a tuple/list of three strings and `axes_range`
# as a tuple/list of 3 pairs of values
def EECTriangleOPE(nbins,
                   axes=('log', 'log', 'id'),
                   **kwargs):

    # allow integers for _powers arguments
    for key in ['weight_powers', 'charge_powers']:
        if key in kwargs and isinstance(kwargs[key], (int, float)):
            kwargs[key] = (kwargs[key],)

    axes = tuple(map(lambda x: x.lower(), axes))
    if axes == ('log', 'log', 'id'):
        return EECTriangleOPELogLogId(nbins, **kwargs)
    elif axes == ('id', 'log', 'id'):
        return EECTriangleOPEIdLogId(nbins, **kwargs)
    elif axes == ('log', 'id', 'id'):
        return EECTriangleOPELogIdId(nbins, **kwargs)
    elif axes == ('id', 'id', 'id'):
        return EECTriangleOPEIdIdId(nbins, **kwargs)
    else:
        raise ValueError('axes `{}` not understood'.format(axes))

# selects between EECLongestSide and EECTriangleOPE using a string `comp_type`
def EEC(comp_type, *args, **kwargs):
    if comp_type == 'EECLongestSide':
        return EECLongestSide(*args, **kwargs)

    if comp_type == 'EECTriangleOPE':
        return EECTriangleOPE(*args, **kwargs)

    raise ValueError('invalid comp_type `{}`'.format(comp_type))
