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

from __future__ import absolute_import, division, print_function

import sys
import time

import numpy as np

from cpython.exc cimport PyErr_CheckSignals
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np

cimport eec.eeccomps as eeccomps
from eec.eeccomps cimport id_tr, log_tr

__all__ = ['EECTriangleOPE', 'EECLongestSide']

cdef print_atomic(s, lock):
    if lock is not None:
        lock.acquire()
    print(s, end='')
    if lock is not None:
        lock.release()

###############################################################################
# Helper function that runs the computation and accepts a fused type
#   - It appears that this cannot be part of the base class due to Cython
###############################################################################

cdef void compute_events(eeccomps.EECComputation_t * eec, size_t nev,
                         const vector[double*] & event_ptrs, const vector[unsigned] & event_mults, double[::1] weights_view,
                         size_t hist_size, const vector[double*] & hists_ptrs, const vector[double*] & hist_errs_ptrs,
                         int verbose, int print_every, bool overflows, object lock) nogil:

    with gil:
        start = time.time()

    # get size of histograms
    cdef size_t j = 0
    for j in range(nev):
        eec.compute(event_ptrs[j], event_mults[j], weights_view[j])

        if verbose > 0 and (j+1) % print_every == 0:
            with gil:
                print_atomic('  {} events done in {:.3f}s\n'.format(j+1, time.time() - start), lock)

        # check signals (handles interrupt)
        if (j % 100) == 0:
            with gil:
                PyErr_CheckSignals()

    if verbose > 0 and ((j+1) % print_every != 0 or nev == 0):
        with gil:
            print_atomic('  {} events done in {:.3f}s\n'.format(nev, time.time() - start), lock)

    # extract histograms
    for j in range(hists_ptrs.size()):
        eec.get_hist(hists_ptrs[j], hist_errs_ptrs[j], hist_size, overflows, j)

###############################################################################
# EECComputation base class
###############################################################################

cdef class EECComputation:
    cdef:
        readonly np.ndarray hists, hist_errs, bin_centers, bin_edges, weights
        readonly list events
        readonly int print_every, verbose
        readonly unsigned nfeatures, nhists, transform_i
        readonly bool norm, overflows
        readonly tuple hist_shape
        readonly vector[double] pt_powers
        readonly vector[unsigned] ch_powers

        size_t nev, hist_size
        vector[double*] event_ptrs, hists_ptrs, hist_errs_ptrs
        vector[unsigned] event_mults
        double[::1] weights_view

        object lock

    def __init__(self, pt_powers, ch_powers, N, norm, overflows, print_every, verbose, lock):

        self.norm = norm
        self.overflows = overflows
        self.print_every = print_every
        self.verbose = verbose
        self.nfeatures = 3

        if hasattr(pt_powers, '__getitem__'):
            if len(pt_powers) != N:
                raise ValueError('pt_powers should be length N or a number')
        else:
            pt_powers = N*[pt_powers]

        if hasattr(ch_powers, '__getitem__'):
            if len(ch_powers) != N:
                raise ValueError('ch_powers should be length N or an integer')
        else:
            ch_powers = N*[ch_powers]

        for i in range(N):
            self.pt_powers.push_back(pt_powers[i])
            self.ch_powers.push_back(ch_powers[i])
            if self.ch_powers.back() != 0:
                self.nfeatures = 4

    def _set_lock(self, lock):
        self.lock = lock

    cdef void print_atomic(self, s):
        print_atomic(s, self.lock)

    cdef void preprocess_events(self, events, weights):

        # clear vectors
        self.event_ptrs.clear()
        self.hists_ptrs.clear()
        self.hist_errs_ptrs.clear()
        self.event_mults.clear()

        # initialize fresh histograms
        self.init_hists()

        # preprocess the events
        self.events = []
        nf = self.nfeatures
        for i,event in enumerate(events):
            event = np.atleast_2d(event)
            if len(event) > 0 and event.shape[1] < nf:
                raise IndexError('event {} has too few columns for requested computation'.format(i))
            self.events.append(np.asarray(event[:,:nf], dtype=np.double, order='C'))

        # get event pointers
        self.nev = len(self.events)
        cdef double[:,::1] data_view_2d
        for event in self.events:
            data_view_2d = event
            self.event_ptrs.push_back(&data_view_2d[0,0])
            self.event_mults.push_back(len(event))

        if weights is None:
            self.weights = np.ones(self.nev, dtype=np.double, order='C')
        if isinstance(weights, (float, int)):
            self.weights = weights * np.ones(self.nev, dtype=np.double, order='C')
        self.weights_view = self.weights

    cdef void init_hists(self):

        # initialize histograms
        self.hists = np.zeros((self.nhists,) + self.hist_shape, dtype=np.double, order='C')
        self.hist_errs = np.zeros((self.nhists,) + self.hist_shape, dtype=np.double, order='C')
        self.hist_size = np.prod(self.hist_shape)

###############################################################################
# EECTriangleOPE
###############################################################################

cdef class EECTriangleOPE(EECComputation):

    allowed_axis_transforms = frozenset([('id', 'id', 'id'), ('log', 'id', 'id'), 
                                         ('id', 'log', 'id'), ('log', 'log', 'id')])

    cdef:
        readonly tuple axis_transforms
        readonly unsigned nbins0, nbins1, nbins2
        readonly double axis0_min, axis0_max, axis1_min, axis1_max, axis2_min, axis2_max

        eeccomps.EECTriangleOPE[id_tr, id_tr, id_tr] * eec_p_iii
        eeccomps.EECTriangleOPE[log_tr, id_tr, id_tr] * eec_p_lii
        eeccomps.EECTriangleOPE[id_tr, log_tr, id_tr] * eec_p_ili
        eeccomps.EECTriangleOPE[log_tr, log_tr, id_tr] * eec_p_lli

    def __cinit__(self):
        self.eec_p_iii = self.eec_p_lii = self.eec_p_ili = self.eec_p_lli = NULL

    def __init__(self, nbins, axis_ranges, axis_transforms, pt_powers=1, ch_powers=0,
                       norm=True, overflows=True, print_every=1000, verbose=0,
                       check_degen=False, average_verts=False):
        super(EECTriangleOPE, self).__init__(pt_powers, ch_powers, 3, norm, overflows, print_every, verbose, None)

        self.axis_transforms = tuple(axis_transforms)

        if len(nbins) != 3:
            raise ValueError('nbins should be length 3')

        if len(axis_ranges) != 3:
            raise ValueError('axis_ranges should be length 3')

        if len(self.axis_transforms) != 3:
            raise ValueError('axis_transforms should be length 3')

        if self.axis_transforms not in self.allowed_axis_transforms:
            raise ValueError('axis_transforms not in ' + str(self.allowed_axis_transforms))

        for r in axis_ranges:
            if len(r) != 2:
                raise ValueError('axis_range ' + str(r) + ' not length 2')

        self.axis0_min, self.axis0_max = axis_ranges[0]
        self.axis1_min, self.axis1_max = axis_ranges[1]
        self.axis2_min, self.axis2_max = axis_ranges[2]

        for nbin in nbins:
            if nbin <= 0:
                raise ValueError('nbins must be positive integers')
            
        self.nbins0, self.nbins1, self.nbins2 = nbins
        self.hist_shape = (self.nbins0 + 2, self.nbins1 + 2, self.nbins2 + 2) if self.overflows else tuple(nbins)
        self.init_hists()

        # set pointer to eec
        if self.axis_transforms == ('id', 'id', 'id'):
            self.transform_i = 0
            self.eec_p_iii = new eeccomps.EECTriangleOPE[id_tr, id_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                              self.nbins1, self.axis1_min, self.axis1_max,
                                                                              self.nbins2, self.axis2_min, self.axis2_max,
                                                                              self.norm, self.pt_powers, self.ch_powers,
                                                                              check_degen, average_verts)
            self.store_bins(self.eec_p_iii)
        elif self.axis_transforms == ('log', 'id', 'id'):
            self.transform_i = 1
            self.eec_p_lii = new eeccomps.EECTriangleOPE[log_tr, id_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                               self.nbins1, self.axis1_min, self.axis1_max,
                                                                               self.nbins2, self.axis2_min, self.axis2_max,
                                                                               self.norm, self.pt_powers, self.ch_powers,
                                                                               check_degen, average_verts)
            self.store_bins(self.eec_p_lii)
        elif self.axis_transforms == ('id', 'log', 'id'):
            self.transform_i = 2
            self.eec_p_ili = new eeccomps.EECTriangleOPE[id_tr, log_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                               self.nbins1, self.axis1_min, self.axis1_max,
                                                                               self.nbins2, self.axis2_min, self.axis2_max,
                                                                               self.norm, self.pt_powers, self.ch_powers,
                                                                               check_degen, average_verts)
            self.store_bins(self.eec_p_ili)
        elif self.axis_transforms == ('log', 'log', 'id'):
            self.transform_i = 3
            self.eec_p_lli = new eeccomps.EECTriangleOPE[log_tr, log_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                                self.nbins1, self.axis1_min, self.axis1_max,
                                                                                self.nbins2, self.axis2_min, self.axis2_max,
                                                                                self.norm, self.pt_powers, self.ch_powers,
                                                                                check_degen, average_verts)
            self.store_bins(self.eec_p_lli)
        else:
            raise ValueError('transform ' + str(self.axis_transforms) + ' not allowed')

    def __dealloc__(self):
        del self.eec_p_iii
        del self.eec_p_lii
        del self.eec_p_ili
        del self.eec_p_lli

    def __repr__(self):
        if self.transform_i == 0:
            return self.eec_p_iii.description().decode('UTF-8')
        elif self.transform_i == 1:
            return self.eec_p_lii.description().decode('UTF-8')
        elif self.transform_i == 2:
            return self.eec_p_ili.description().decode('UTF-8')
        elif self.transform_i == 3:
            return self.eec_p_lli.description().decode('UTF-8')

    cdef void store_bins(self, eeccomps.EECTriangleOPE_t * eec):
        self.nhists = eec.nhists()
        self.bin_centers = np.array([list(eec.bin_centers(0)), list(eec.bin_centers(1)), list(eec.bin_centers(2))])
        self.bin_edges = np.array([list(eec.bin_edges(0)), list(eec.bin_edges(1)), list(eec.bin_edges(2))])

    def compute(self, events, weights=None):

        if self.verbose > 0:
            self.print_atomic('  Computing {}, axes - {}, on {} events\n'.format(
                              self.__class__.__name__, self.axis_transforms, len(events)))

        self.preprocess_events(events, weights)

        # initialize histograms
        cdef double[:,:,:,::1] hists_view = self.hists
        cdef double[:,:,:,::1] hist_errs_view = self.hist_errs
        for i in range(self.nhists):
            self.hists_ptrs.push_back(&hists_view[i,0,0,0])
            self.hist_errs_ptrs.push_back(&hist_errs_view[i,0,0,0])

        # compute on specific eec
        if self.transform_i == 0:
            self._compute(self.eec_p_iii)
        elif self.transform_i == 1:
            self._compute(self.eec_p_lii)
        elif self.transform_i == 2:
            self._compute(self.eec_p_ili)
        elif self.transform_i == 3:
            self._compute(self.eec_p_lli)

    cdef void _compute(self, eeccomps.EECTriangleOPE_t * eec) nogil:
        compute_events(eec, self.nev, self.event_ptrs, self.event_mults, self.weights_view,
                            self.hist_size, self.hists_ptrs, self.hist_errs_ptrs,
                            self.verbose, self.print_every, self.overflows, self.lock)

###############################################################################
# EECLongestSide
###############################################################################

cdef class EECLongestSide(EECComputation):

    allowed_axis_transforms = set(['id', 'log'])

    cdef:
        readonly str axis_transform
        readonly unsigned N, nbins
        readonly double axis_min, axis_max

        eeccomps.EECLongestSide[id_tr] * eec_p_i
        eeccomps.EECLongestSide[log_tr] * eec_p_l

    def __cinit__(self):
        self.eec_p_i = self.eec_p_l = NULL

    def __init__(self, N, nbins, axis_range, axis_transform, pt_powers=1, ch_powers=0,
                       norm=True, overflows=True, print_every=1000, verbose=0,
                       check_degen=False, average_verts=False):
        super(EECLongestSide, self).__init__(pt_powers, ch_powers, N, norm, overflows, print_every, verbose, None)

        self.axis_transform = str(axis_transform)
        self.nbins = nbins
        self.N = N

        if self.N < 2 or self.N > 5:
            raise ValueError('N must be 2, 3, 4, or 5')

        if self.nbins <= 0:
            raise ValueError('nbins must be a positive integer')

        if len(axis_range) != 2:
            raise ValueError('axis_range should be length 2')

        if self.axis_transform not in self.allowed_axis_transforms:
            raise ValueError('axis_transform not in ' + str(self.allowed_axis_transforms))

        self.axis_min, self.axis_max = axis_range
        self.hist_shape = (self.nbins + 2 if self.overflows else self.nbins,)
        self.init_hists()

        # set pointer to eec
        if self.axis_transform == 'id':
            self.transform_i = 0
            self.eec_p_i = new eeccomps.EECLongestSide[id_tr](self.nbins, self.axis_min, self.axis_max, 
                                                              self.N, self.norm, self.pt_powers, self.ch_powers,
                                                              check_degen, average_verts)
            self.store_bins(self.eec_p_i)

        elif self.axis_transform == 'log':
            self.transform_i = 1
            self.eec_p_l = new eeccomps.EECLongestSide[log_tr](self.nbins, self.axis_min, self.axis_max, 
                                                               self.N, self.norm, self.pt_powers, self.ch_powers,
                                                               check_degen, average_verts)
            self.store_bins(self.eec_p_l)

    def __dealloc__(self):
        del self.eec_p_i
        del self.eec_p_l

    def __repr__(self):
        if self.transform_i == 0:
            return self.eec_p_i.description().decode('UTF-8')
        elif self.transform_i == 1:
            return self.eec_p_l.description().decode('UTF-8')

    cdef void store_bins(self, eeccomps.EECLongestSide_t * eec):
        self.nhists = eec.nhists()
        self.bin_centers = np.array(list(eec.bin_centers()))
        self.bin_edges = np.array(list(eec.bin_edges()))

    def compute(self, events, weights=None):

        if self.verbose > 0:
            self.print_atomic('  Computing {}, N = {}, axis - {}, on {} events\n'.format(
                              self.__class__.__name__, self.N, self.axis_transform, len(events)))

        self.preprocess_events(events, weights)

        # initialize histograms
        cdef double[:,::1] hists_view = self.hists
        cdef double[:,::1] hist_errs_view = self.hist_errs
        for i in range(self.nhists):
            self.hists_ptrs.push_back(&hists_view[i,0])
            self.hist_errs_ptrs.push_back(&hist_errs_view[i,0])

        # compute on specific eec
        if self.transform_i == 0:
            self._compute(self.eec_p_i)
        elif self.transform_i == 1:
            self._compute(self.eec_p_l)

    cdef void _compute(self, eeccomps.EECLongestSide_t * eec) nogil:
        compute_events(eec, self.nev, self.event_ptrs, self.event_mults, self.weights_view,
                            self.hist_size, self.hists_ptrs, self.hist_errs_ptrs,
                            self.verbose, self.print_every, self.overflows, self.lock)