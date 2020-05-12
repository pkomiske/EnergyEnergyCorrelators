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

import time

import numpy as np

from cpython.exc cimport PyErr_CheckSignals
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np

cimport eec.eeccomps as eeccomps
from eec.eeccomps cimport id_tr, log_tr

__all__ = ['EECTriangleOPE', 'EECLongestSide']

###############################################################################
# Function taking a fused type of all possible eec computations
###############################################################################

cdef void compute_events(eeccomps.EECComputation_t * eec, 
                         size_t nev, vector[double*] event_ptrs, vector[size_t] event_mults, double[::1] event_weights,
                         double * hist_ptr, double * hist_errs_ptr, size_t hist_size, 
                         bool overflows, int print_every, int verbose) nogil:

        with gil:
            start = time.time()

        cdef size_t j = 0
        for j in range(nev):
            eec.compute(event_ptrs[j], event_mults[j], event_weights[j])

            if verbose > 0 and (j+1) % print_every == 0:
                with gil:
                    print('  {} events done in {:.3f}s'.format(j+1, time.time() - start))

            # check signals (handles interrupt)
            if (j % 10) == 0:
                with gil:
                    PyErr_CheckSignals()

        if verbose > 0 and (j+1) % print_every != 0:
            with gil:
                print('  {} events done in {:.3f}s'.format(j+1, time.time() - start))

        eec.get_hist(hist_ptr, hist_errs_ptr, hist_size, overflows)

###############################################################################
# EECComputation base class
###############################################################################

cdef class EECComputation:
    cdef:
        np.ndarray _hist, _hist_errs, _bin_centers, _bin_edges
        int print_every, verbose
        bool norm, overflows

    def __init__(self, norm, overflows, print_every, verbose):

        self.norm = norm
        self.overflows = overflows
        self.print_every = print_every
        self.verbose = verbose

        self._hist = np.zeros(1, dtype=np.double, order='C')
        self._hist_errs = np.zeros(1, dtype=np.double, order='C')
        self._bin_centers = np.zeros(1, dtype=np.double, order='C')
        self._bin_edges = np.zeros(1, dtype=np.double, order='C')

    cdef void init_hist(self, shape):
        self._hist = np.zeros(shape, dtype=np.double, order='C')
        self._hist_errs = np.zeros(shape, dtype=np.double, order='C')

    cdef preprocess_events(self, events, vector[double*] & event_ptrs, vector[size_t] & event_mults):

        # preprocess the events
        events = [np.asarray(np.atleast_2d(event)[:,:3], dtype=np.double, order='C') for event in events]

        cdef double[:,::1] data_view_2d

        # norm events and get their pointers
        for i in range(len(events)):
            if self.norm:
                events[i][:,0] /= np.sum(events[i][:,0])

            data_view_2d = events[i]
            event_ptrs.push_back(&data_view_2d[0,0])
            event_mults.push_back(len(events[i]))

        return events

    @property
    def hist(self):
        return self._hist

    @property
    def hist_errs(self):
        return self._hist_errs

    @property
    def bin_centers(self):
        return self._bin_centers

    @property
    def bin_edges(self):
        return self._bin_edges

###############################################################################
# EECTriangleOPE
###############################################################################

cdef class EECTriangleOPE(EECComputation):

    allowed_axis_transforms = set([('id', 'id', 'id'), ('log', 'id', 'id'), 
                                   ('id', 'log', 'id'), ('log', 'log', 'id')])

    cdef:
        tuple axis_transforms, hist_shape
        unsigned int nbins0, nbins1, nbins2, transform_i
        double axis0_min, axis0_max, axis1_min, axis1_max, axis2_min, axis2_max

        eeccomps.EECTriangleOPE[id_tr, id_tr, id_tr] * eec_p_iii
        eeccomps.EECTriangleOPE[log_tr, id_tr, id_tr] * eec_p_lii
        eeccomps.EECTriangleOPE[id_tr, log_tr, id_tr] * eec_p_ili
        eeccomps.EECTriangleOPE[log_tr, log_tr, id_tr] * eec_p_lli

    def __cinit__(self):
        self.eec_p_iii = self.eec_p_lii = self.eec_p_ili = self.eec_p_lli = NULL

    def __init__(self, nbins, axis_ranges, axis_transforms, norm=True, overflows=True, print_every=100, verbose=0):
        super(EECTriangleOPE, self).__init__(norm, overflows, print_every, verbose)

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
        self.init_hist(self.hist_shape)

        # set pointer to eec
        if self.axis_transforms == ('id', 'id', 'id'):
            self.transform_i = 0
            self.eec_p_iii = new eeccomps.EECTriangleOPE[id_tr, id_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                              self.nbins1, self.axis1_min, self.axis1_max,
                                                                              self.nbins2, self.axis2_min, self.axis2_max)
            self.store_bins(self.eec_p_iii)

        elif self.axis_transforms == ('log', 'id', 'id'):
            self.transform_i = 1
            self.eec_p_lii = new eeccomps.EECTriangleOPE[log_tr, id_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                               self.nbins1, self.axis1_min, self.axis1_max,
                                                                               self.nbins2, self.axis2_min, self.axis2_max)
            self.store_bins(self.eec_p_lii)

        elif self.axis_transforms == ('id', 'log', 'id'):
            self.transform_i = 2
            self.eec_p_ili = new eeccomps.EECTriangleOPE[id_tr, log_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                               self.nbins1, self.axis1_min, self.axis1_max,
                                                                               self.nbins2, self.axis2_min, self.axis2_max)
            self.store_bins(self.eec_p_ili)

        elif self.axis_transforms == ('log', 'log', 'id'):
            self.transform_i = 3
            self.eec_p_lli = new eeccomps.EECTriangleOPE[log_tr, log_tr, id_tr](self.nbins0, self.axis0_min, self.axis0_max,
                                                                                self.nbins1, self.axis1_min, self.axis1_max,
                                                                                self.nbins2, self.axis2_min, self.axis2_max)
            self.store_bins(self.eec_p_lli)
        else:
            raise ValueError('transform ' + str(self.axis_transforms) + ' not allowed')

    def __dealloc__(self):
        del self.eec_p_iii
        del self.eec_p_lii
        del self.eec_p_ili
        del self.eec_p_lli

    cdef void store_bins(self, eeccomps.EECTriangleOPE_t * eec):
        self._bin_centers = np.array([list(eec.bin_centers(0)), list(eec.bin_centers(1)), list(eec.bin_centers(2))])
        self._bin_edges = np.array([list(eec.bin_edges(0)), list(eec.bin_edges(1)), list(eec.bin_edges(2))])

    def compute(self, events, weights=None):

        if self.verbose > 0:
            print('Computing {}, axes - {}, on {} events'.format(self.__class__.__name__, 
                  self.axis_transforms, len(events)))

        cdef vector[double*] event_ptrs
        cdef vector[size_t] event_mults
        events = self.preprocess_events(events, event_ptrs, event_mults)

        cdef double[::1] event_weights
        if weights is None:
            weights = np.ones(len(events), dtype=np.double, order='C')
        if isinstance(weights, (float, int)):
            weights = weights * np.ones(len(events), dtype=np.double, order='C')
        event_weights = np.asarray(weights, dtype=np.double, order='C')

        cdef double[:,:,::1] hist_view = self.hist
        cdef double[:,:,::1] hist_errs_view = self.hist_errs
        cdef double* hist_ptr = &hist_view[0,0,0]
        cdef double* hist_errs_ptr = &hist_errs_view[0,0,0]

        cdef size_t hist_size = self.hist.size
        cdef size_t nev = len(events)

        # compute on specific eec
        if self.transform_i == 0:
            compute_events(self.eec_p_iii, nev, event_ptrs, event_mults, event_weights, 
                           hist_ptr, hist_errs_ptr, hist_size, self.overflows, self.print_every, self.verbose)
        elif self.transform_i == 1:
            compute_events(self.eec_p_lii, nev, event_ptrs, event_mults, event_weights, 
                           hist_ptr, hist_errs_ptr, hist_size, self.overflows, self.print_every, self.verbose)
        elif self.transform_i == 2:
            compute_events(self.eec_p_ili, nev, event_ptrs, event_mults, event_weights, 
                           hist_ptr, hist_errs_ptr, hist_size, self.overflows, self.print_every, self.verbose)
        elif self.transform_i == 3:
            compute_events(self.eec_p_lli, nev, event_ptrs, event_mults, event_weights, 
                           hist_ptr, hist_errs_ptr, hist_size, self.overflows, self.print_every, self.verbose)

###############################################################################
# EECLongestSide
###############################################################################

cdef class EECLongestSide(EECComputation):

    allowed_axis_transforms = set(['id', 'log'])

    cdef:
        str axis_transform
        unsigned int N, nbins, transform_i, hist_shape
        double axis_min, axis_max

        eeccomps.EECLongestSide[id_tr] * eec_p_i
        eeccomps.EECLongestSide[log_tr] * eec_p_l

    def __cinit__(self):
        self.eec_p_i = self.eec_p_l = NULL

    def __init__(self, N, nbins, axis_range, axis_transform, norm=True, overflows=True, print_every=1000, verbose=0):
        super(EECLongestSide, self).__init__(norm, overflows, print_every, verbose)

        self.axis_transform = str(axis_transform)
        self.nbins = int(nbins)
        self.N = int(N)

        if self.N < 2 or self.N > 5:
            raise ValueError('N must be 2, 3, 4, or 5')

        if self.nbins <= 0:
            raise ValueError('nbins must be a positive integer')

        if len(axis_range) != 2:
            raise ValueError('axis_range should be length 2')

        if self.axis_transform not in self.allowed_axis_transforms:
            raise ValueError('axis_transform not in ' + str(self.allowed_axis_transforms))

        self.axis_min, self.axis_max = axis_range

        self.hist_shape = self.nbins + 2 if self.overflows else self.nbins
        self.init_hist(self.hist_shape)

        # set pointer to eec
        if self.axis_transform == 'id':
            self.transform_i = 0
            self.eec_p_i = new eeccomps.EECLongestSide[id_tr](self.N, self.nbins, self.axis_min, self.axis_max)
            self.store_bins(self.eec_p_i)

        elif self.axis_transform == 'log':
            self.transform_i = 1
            self.eec_p_l = new eeccomps.EECLongestSide[log_tr](self.N, self.nbins, self.axis_min, self.axis_max)
            self.store_bins(self.eec_p_l)

    def __dealloc__(self):
        del self.eec_p_i
        del self.eec_p_l

    cdef void store_bins(self, eeccomps.EECLongestSide_t * eec):
        self._bin_centers = np.array(list(eec.bin_centers()))
        self._bin_edges = np.array(list(eec.bin_edges()))

    def compute(self, events, weights=None):

        if self.verbose > 0:
            print('Computing {}, N = {}, axis - {}, on {} events'.format(self.__class__.__name__, 
                  self.N, self.axis_transform, len(events)))

        cdef vector[double*] event_ptrs
        cdef vector[size_t] event_mults
        events = self.preprocess_events(events, event_ptrs, event_mults)

        cdef double[::1] event_weights
        if weights is None:
            weights = np.ones(len(events), dtype=np.double)
        if isinstance(weights, (float, int)):
            weights = weights * np.ones(len(events), dtype=np.double, order='C')
        event_weights = np.asarray(weights, dtype=np.double, order='C')

        cdef double[::1] hist_view = self.hist
        cdef double[::1] hist_errs_view = self.hist_errs
        cdef double* hist_ptr = &hist_view[0]
        cdef double* hist_errs_ptr = &hist_errs_view[0]

        cdef size_t hist_size = self.hist.size
        cdef size_t nev = len(events)

        # compute on specific eec
        if self.transform_i == 0:
            compute_events(self.eec_p_i, nev, event_ptrs, event_mults, event_weights, 
                           hist_ptr, hist_errs_ptr, hist_size, self.overflows, self.print_every, self.verbose)
        elif self.transform_i == 1:
            compute_events(self.eec_p_l, nev, event_ptrs, event_mults, event_weights, 
                           hist_ptr, hist_errs_ptr, hist_size, self.overflows, self.print_every, self.verbose)
