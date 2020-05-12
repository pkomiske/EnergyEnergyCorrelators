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

from libcpp cimport bool
from libcpp.vector cimport vector

# external definitions from EECComputations C++ header 
cdef extern from 'EECComputations.hh' namespace 'eec' nogil:
    cdef cppclass id_tr 'eec::bh::axis::transform::id'
    cdef cppclass log_tr 'eec::bh::axis::transform::log'

    cdef cppclass EECTriangleOPE[Tr0, Tr1, Tr2]:
        EECTriangleOPE(unsigned int, double, double, unsigned int, double, double, unsigned int, double, double)
        void compute(const double *, size_t mult, double)
        void get_hist(double *, double *, size_t, bool)
        vector[double] bin_centers(int)
        vector[double] bin_edges(int)
        
    cdef cppclass EECLongestSide[Tr0]:
        EECLongestSide(unsigned int, unsigned int, double, double)
        void compute(const double *, size_t mult, double)
        void get_hist(double *, double *, size_t, bool)
        vector[double] bin_centers()
        vector[double] bin_edges()

# fused type for EEC TriangleOPE templates
ctypedef fused EECTriangleOPE_t:
    EECTriangleOPE[id_tr, id_tr, id_tr]
    EECTriangleOPE[log_tr, id_tr, id_tr]
    EECTriangleOPE[id_tr, log_tr, id_tr]
    EECTriangleOPE[log_tr, log_tr, id_tr]

# fused type for EEC LongestSide templates
ctypedef fused EECLongestSide_t:
    EECLongestSide[id_tr]
    EECLongestSide[log_tr]

# type that can hold any EECComputation template of relevance
ctypedef fused EECComputation_t:
    EECTriangleOPE[id_tr, id_tr, id_tr]
    EECTriangleOPE[log_tr, id_tr, id_tr]
    EECTriangleOPE[id_tr, log_tr, id_tr]
    EECTriangleOPE[log_tr, log_tr, id_tr]
    EECLongestSide[id_tr]
    EECLongestSide[log_tr]
