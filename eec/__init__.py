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

from __future__ import absolute_import

from . import core
from .core import *

__all__ = core.__all__ + ['eec']

__version__ = '0.0.0'

def eec(name, events, *args, njobs=None, weights=None, **kwargs):

    import multiprocessing
    import numpy as np

    # check that computation is allowed
    if name not in core.__all__:
        raise ValueError('not implemented'.format(name))

    # handle using all cores
    if njobs is None or njobs == -1:
        njobs = multiprocessing.cpu_count() or 1
    njobs = min(njobs, len(events))

    if kwargs.get('verbose', 0) > 0:
        print('Using {} jobs'.format(njobs))

    if njobs > 1:

        # compute index ranges that divide up the events
        num_per_job, remainder = divmod(len(events), njobs)
        index_ranges = [[i*num_per_job, (i+1)*num_per_job] for i in range(njobs)]
        index_ranges[-1][1] += remainder

        # make argument for iterable
        eec_args = [(events[start:stop], weights[start:stop] if weights is not None else None) for start,stop in index_ranges]

        # use a pool of worker processes to compute 
        with multiprocessing.Pool(processes=njobs, initializer=_init_eec, initargs=(name, args, kwargs)) as pool:
            #results = pool.map(_compute_eec_on_events, index_ranges, chunksize=1)
            results = pool.map(_compute_eec_on_events, eec_args, chunksize=1)

        # add histograms, add errors in quadrature
        hist, hist_errs, bin_centers, bin_edges = results[0]
        hist_errs2 = hist_errs**2
        for i in range(1, njobs):
            hist += results[i][0]
            hist_errs2 += results[i][1]**2
        hist_errs = np.sqrt(hist_errs2)

    else:
        eec = getattr(core, name)(*args, **kwargs)
        eec.compute(events, weights=weights)
        hist, hist_errs, bin_centers, bin_edges = eec.hist, eec.hist_errs, eec.bin_centers, eec.bin_edges

    #del global_weights, global_events

    return hist, hist_errs, bin_centers, bin_edges

def _init_eec(name, args, kwargs):
    global eec
    eeccomp = getattr(core, name)
    eec = eeccomp(*args, **kwargs)

def _compute_eec_on_events(arg):
    events, weights = arg
    eec.compute(events, weights=weights)

    return eec.hist, eec.hist_errs, eec.bin_centers, eec.bin_edges
