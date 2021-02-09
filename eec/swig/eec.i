// -*- C -*-
//
// EnergyEnergyCorrelators - Evaluates EECs on particle physics events
// Copyright (C) 2020 Patrick T. Komiske III
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

%module("threads"=1) eec
%nothreadallow;

#define EECNAMESPACE eec

// this can be used to ensure that swig parses classes correctly
#define SWIG_EEC

// use numpy
#define SWIG_NUMPY

// indicate swig is running
#define SWIG_PREPROCESSOR

%feature("autodoc", "1");

// C++ standard library wrappers
%include <exception.i>
%include <std_string.i>
%include <std_vector.i>

// vector templates
%template(vectorDouble) std::vector<double>;
%template(vectorUnsigned) std::vector<unsigned>;

%{
// include these to avoid needing to define them at compile time 
#ifndef SWIG
#define SWIG
#endif
#ifndef SWIG_EEC
#define SWIG_EEC
#endif

// needed by numpy.i, harmless otherwise
#define SWIG_FILE_WITH_INIT

// standard library headers we need
#include <cstdlib>
#include <cstring>

// EEC library headers
#include "EEC.hh"

// macros for exception handling
#define CATCH_STD_EXCEPTION catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
#define CATCH_STD_INVALID_ARGUMENT catch (std::invalid_argument & e) { SWIG_exception(SWIG_ValueError, e.what()); }
#define CATCH_STD_RUNTIME_ERROR catch (std::runtime_error & e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
#define CATCH_STD_LOGIC_ERROR catch (std::logic_error & e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
#define CATCH_STD_OUT_OF_RANGE catch (std::out_of_range & e) { SWIG_exception(SWIG_IndexError, e.what()); }

// using namespace
using namespace eec;
%}

// numpy wrapping and initialization
#ifdef SWIG_NUMPY

// include numpy typemaps
%include numpy.i
%init %{
import_array();
%}

// numpy typemaps
%apply (double* IN_ARRAY1, int DIM1) {(double* weights0, int n0), (double* weights1, int n1)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* coords0, int n00, int n01),
                                                (double* coords1, int n10, int n11),
                                                (double* external_dists, int d0, int d1)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int n0)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* coords, int n1, int d)}
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double** arr_out0, int* n0), (double** arr_out1, int* n1)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(double** arr_out, int* n0, int* n1)}

#endif // SWIG_NUMPY

// allow threads in PairwiseEMD computation
%threadallow EMDNAMESPACE::PairwiseEMD::compute;

// basic exception handling for all functions
%exception {
  try { $action }
  CATCH_STD_EXCEPTION
}

namespace EECNAMESPACE {

// ignore EECHist functions
%ignore get_bin_centers;
%ignore get_bin_edges;
%ignore combine_hists;
%ignore HistBase::get_hist_errs;
%ignore Hist1D::duplicate_hists;
%ignore Hist1D::axis;
%ignore Hist1D::hists;
%ignore Hist1D::combined_hist;
%ignore Hist1D::get_hist;
%ignore Hist3D::duplicate_hists;
%ignore Hist3D::axis;
%ignore Hist3D::hists;
%ignore Hist3D::combined_hist;
%ignore Hist3D::get_hist;

// ignore/rename Multinomial functions
%ignore multinomial;
%rename(multinomial) multinomial_vector;

// ignore EEC functions
%ignore EECBase::EECBase();
%ignore EECBase::compute;

} // namespace EECNAMESPACE

// include EECHist and declare templates
%include "EECHist.hh"
namespace EECNAMESPACE {
  %template(Hist1DId) Hist1D<axis::id>;
  %template(Hist1DLog) Hist1D<axis::log>;
  %template(Hist3DIdIdId) Hist3D<axis::id, axis::id, axis::id>;
  %template(Hist3DLogIdId) Hist3D<axis::log, axis::id, axis::id>;
  %template(Hist3DIdLogId) Hist3D<axis::id, axis::log, axis::id>;
  %template(Hist3DLogLogId) Hist3D<axis::log, axis::log, axis::id>;
}

// include EEC code and declare templates
%include "EECBase.hh"
%include "EECMultinomial.hh"
%include "EECLongestSide.hh"
%include "EECTriangleOPE.hh"
namespace EECNAMESPACE {
  %template(Multinomial2) Multinomial<2>;
  %template(Multinomial3) Multinomial<3>;
  %template(Multinomial4) Multinomial<4>;
  %template(Multinomial5) Multinomial<5>;
  %template(Multinomial6) Multinomial<6>;

  %template(EECLongestSideId) EECLongestSide<axis::id>;
  %template(EECLongestSideLog) EECLongestSide<axis::log>;
  %template(EECTriangleOPEIdIdId) EECTriangleOPE<axis::id, axis::id, axis::id>;
  %template(EECTriangleOPELogIdId) EECTriangleOPE<axis::log, axis::id, axis::id>;
  %template(EECTriangleOPEIdLogId) EECTriangleOPE<axis::id, axis::log, axis::id>;
  %template(EECTriangleOPELogLogId) EECTriangleOPE<axis::log, axis::log, axis::id>;
}
