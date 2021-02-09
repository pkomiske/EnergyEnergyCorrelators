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
Copyright (C) 2020 Patrick T. Komiske III

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

from .eec import *

__version__ = '0.3.0'

def combine_bins(hist, nbins2combine, axes=None, overflows=True, keep_overflows=False,
                                      add_in_quadrature=False, bins=None):

    import numpy as np
    
    # process arguments
    nax = len(hist.shape)
    nbins2combine = _to_iterable(nbins2combine)
    axes = _to_iterable(list(range(nax)) if axes is None else axes)
    keep_overflows = _to_iterable(keep_overflows)
    
    # check arguments for consistency
    if len(nbins2combine) != len(axes):
        assert len(nbins2combine) == 1, 'nbins2compare must be length 1 or match number of axes'
        nbins2combine = len(axes)*nbins2combine
    if len(keep_overflows) != len(axes):
        assert len(keep_overflows) == 1, 'keep_overflows must be length 1 or match number of axes'
        keep_overflows = len(axes)*keep_overflows
    
    # iterate over axes
    for ax,nb2c,kov in zip(axes, nbins2combine, keep_overflows):
        nbins = hist.shape[ax]
        start, end = 0, nbins
        
        # adjust for overflows
        if overflows:
            start = 1
            end -= 1
            nbins -= 2
        assert nbins % nb2c == 0, 'cannot combine hist bins evenly for axis {}'.format(ax)

        # reshape
        axtrans = [i for i in range(nax) if i != ax] + [ax]
        h = hist.transpose(axtrans)
        h = h[...,start:end].reshape(h.shape[:-1] + (nbins//nb2c, nb2c))
        if add_in_quadrature:
            h = np.sqrt(np.sum(h**2, axis=-1).transpose(np.argsort(axtrans)))
        else:
            h = np.sum(h, axis=-1).transpose(np.argsort(axtrans))
        
        # add overflows back if requested
        if kov:
            inds = nax*[slice(None)]
            inds[ax] = slice(1)
            unders = hist[tuple(inds)]
            inds[ax] = slice(-1, None)
            overs = hist[tuple(inds)]
            h = np.concatenate((unders, h, overs), axis=ax)

        # set hist to result
        hist = h

    # resample bins as well
    if bins is not None:
        single = len(bins.shape) == 1
        bins = [x for x in np.atleast_2d(bins)]
        for ax,nb2c in zip(axes, nbins2combine):
            bins[ax] = bins[ax][::nb2c]

        new_bins = [len(b) for b in bins]
        dtype = np.double if min(new_bins) == max(new_bins) else object

        return hist, bins[0] if single else np.asarray(bins, dtype=dtype)
    else:
        return hist

def midbins(bins, axis='id'):
    if axis == 'log' or axis == True:
        import numpy as np
        return np.sqrt(bins[:-1]*bins[1:])
    else:
        return (bins[:-1] + bins[1:])/2

def _init_eec(l):
    global lock
    lock = l

def _compute_eec_on_events(arg):
    start, events, weights, name, args, kwargs = arg
    eec_obj = getattr(eeccore, name)(*args, **kwargs, lock=lock)

    try:
        eec_obj.compute(events, weights=weights)
    except RuntimeError as e:
        ind = e.args[1]
        raise RuntimeError(str(e), 'event ' + str(start + ind))

    return eec_obj.hists, eec_obj.hist_errs, eec_obj.bin_centers, eec_obj.bin_edges, str(eec_obj)

def _to_iterable(arg):
    import numpy as np
    return arg if isinstance(arg, (tuple, list, np.ndarray)) else [arg]
