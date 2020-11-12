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
from libcpp.string cimport string

# external definitions from EEC C++ header 
cdef extern from 'EEC.hh' namespace 'eec' nogil:
    cdef cppclass id_tr 'eec::bh::axis::transform::id'
    cdef cppclass log_tr 'eec::bh::axis::transform::log'

    cdef cppclass EECTriangleOPE[Tr0, Tr1, Tr2]:
        EECTriangleOPE(unsigned, double, double,
                       unsigned, double, double,
                       unsigned, double, double,
                       bool,
                       const vector[double] &,
                       const vector[unsigned] &,
                       bool, bool) except +
        void compute(const double *, unsigned, double) except +
        void get_hist(double *, double *, size_t, bool, unsigned) except +
        vector[double] bin_centers(int) except +
        vector[double] bin_edges(int) except +
        unsigned nhists()
        string description() except +
        
    cdef cppclass EECLongestSide[Tr0]:
        EECLongestSide(unsigned, double, double,
                       unsigned, bool,
                       const vector[double] &,
                       const vector[unsigned] &,
                       bool, bool, bool) except +
        void compute(const double *, unsigned, double) except +
        void get_hist(double *, double *, size_t, bool, unsigned) except +
        vector[double] bin_centers()
        vector[double] bin_edges()
        unsigned nhists()
        string description() except +

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
