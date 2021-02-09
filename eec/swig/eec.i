// -*- C++ -*-
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

// include numpy typemaps
%include numpy.i
%init %{
import_array();
%}

// numpy typemaps
//%apply (double* IN_ARRAY1, int DIM1) {(double* weights0, int n0), (double* weights1, int n1)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* particles, int mult, int nfeatures)}
//%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int n0)}
//%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* coords, int n1, int d)}
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double** arr_out0, int* n0),
                                                 (double** arr_out1, int* n1)}
%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {(double** arr_out0, int* n0, int* n1, int* n2),
                                                                       (double** arr_out1, int* m0, int* m1, int* m2)}

// makes python class printable from a description method
%define ADD_STR_FROM_DESCRIPTION
std::string __str__() const {
  return $self->description();
}
std::string __repr__() const {
  return $self->description();
}
%enddef

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_VALUE_ARRAY(arr_out, n, size, nbytes)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %zu bytes", nbytes);
    return;
  }
%enddef

// mallocs a 3D array of doubles of the specified size
%define MALLOC_3D_VALUE_ARRAY(arr_out, n0, n1, n2, size0, size1, size2, nbytes)
  *n0 = size0;
  *n1 = size1;
  *n2 = size2;
  size_t nbytes = size_t(*n0)*size_t(*n1)*size_t(*n2)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %zu bytes", nbytes);
    return;
  }
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR(pyname, cppname, size)
void pyname(double** arr_out0, int* n0) {
  MALLOC_1D_VALUE_ARRAY(arr_out0, n0, size, nbytes)
  memcpy(*arr_out0, $self->cppname().data(), nbytes);
}
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR_HIST3D(pyname, cppname)
void pyname(double** arr_out0, int* n0, int i) {
  MALLOC_1D_VALUE_ARRAY(arr_out0, n0, $self->nbins(i), nbytes)
  memcpy(*arr_out0, $self->cppname(i).data(), nbytes);
}
%enddef

// allow threads in PairwiseEMD computation
%threadallow EMDNAMESPACE::PairwiseEMD::compute;

// basic exception handling for all functions
%exception {
  try { $action }
  CATCH_STD_EXCEPTION
}

namespace EECNAMESPACE {

// ignore/rename EECHist functions
%ignore get_bin_centers;
%ignore get_bin_edges;
%ignore combine_hists;
%ignore HistBase::get_hist_errs;
%ignore Hist1D::duplicate_hists;
%ignore Hist3D::duplicate_hists;
%rename(bin_centers_vec) Hist1D::bin_centers;
%rename(bin_edges_vec) Hist1D::bin_edges;
%rename(bin_centers_vec) Hist3D::bin_centers;
%rename(bin_edges_vec) Hist3D::bin_edges;
%rename(bin_centers) Hist1D::npy_bin_centers;
%rename(bin_edges) Hist1D::npy_bin_edges;
%rename(bin_centers) Hist3D::npy_bin_centers;
%rename(bin_edges) Hist3D::npy_bin_edges;
%rename(get_hist_errs) Hist1D::npy_get_hist_errs;
%rename(get_hist_errs) Hist3D::npy_get_hist_errs;

// ignore/rename Multinomial functions
%ignore multinomial;
%rename(multinomial) multinomial_vector;

// ignore EEC functions
%ignore EECEvents::append;
//%ignore EECBase::EECBase();
%ignore EECBase::batch_compute;
%ignore EECBase::compute;
%rename(compute) EECBase::npy_compute;

} // namespace EECNAMESPACE

// include EECHist and declare templates
%include "EECHist.hh"

// extend Hist1D code
%extend EECNAMESPACE::Hist1D {
  RETURN_1DNUMPY_FROM_VECTOR(npy_bin_centers, bin_centers, $self->nbins())
  RETURN_1DNUMPY_FROM_VECTOR(npy_bin_edges, bin_edges, $self->nbins())

  void npy_get_hist_errs(double** arr_out0, int* n0,
                         double** arr_out1, int* n1,
                         bool include_overflows = true, unsigned hist_i = 0) {
    MALLOC_1D_VALUE_ARRAY(arr_out0, n0, $self->hist_size(include_overflows), nbytes0)
    MALLOC_1D_VALUE_ARRAY(arr_out1, n1, $self->hist_size(include_overflows), nbytes1)
    try {
      $self->get_hist_errs(*arr_out0, *arr_out1, include_overflows, hist_i);
    }
    catch (std::exception & e) {
      free(*arr_out0);
      free(*arr_out1);
      throw e;
    }
  }
}

// extend Hist3D code
%extend EECNAMESPACE::Hist3D {
  RETURN_1DNUMPY_FROM_VECTOR_HIST3D(npy_bin_centers, bin_centers)
  RETURN_1DNUMPY_FROM_VECTOR_HIST3D(npy_bin_edges, bin_edges)

  void npy_get_hist_errs(double** arr_out0, int* n0, int* n1, int* n2,
                         double** arr_out1, int* m0, int* m1, int* m2,
                         bool include_overflows = true, unsigned hist_i = 0) {
    MALLOC_3D_VALUE_ARRAY(arr_out0, n0, n1, n2, $self->hist_size(include_overflows, 0),
                                                $self->hist_size(include_overflows, 1),
                                                $self->hist_size(include_overflows, 2), nbytes0)
    MALLOC_3D_VALUE_ARRAY(arr_out1, m0, m1, m2, $self->hist_size(include_overflows, 0),
                                                $self->hist_size(include_overflows, 1),
                                                $self->hist_size(include_overflows, 2), nbytes1)
    try {
      $self->get_hist_errs(*arr_out0, *arr_out1, include_overflows, hist_i);
    }
    catch (std::exception & e) {
      free(*arr_out0);
      free(*arr_out1);
      throw e;
    }
  }
}

// declare histogram templates
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

// extend functionality to include numpy support
%extend EECNAMESPACE::EECEvents {
  void add_event(double* particles, int mult, int nfeatures, double weight = 1.0) {
    $self->append(particles, mult, weight);
  }
}

%extend EECNAMESPACE::EECBase {
  ADD_STR_FROM_DESCRIPTION

  void npy_compute(double* particles, int mult, int nfeatures, double weight = 1.0, int thread_i = 0) {
    if (nfeatures != (int) $self->nfeatures()) {
      std::ostringstream oss;
      oss << "Got array with " << nfeatures << " per particle, expected "
          << $self->nfeatures() << " per particle";
      throw std::runtime_error(oss.str());
      return;
    }
    $self->compute(particles, mult, weight, thread_i);
  }

  // this is needed because we've hidden batch_compute
  void operator()(const EECEvents & evs) {
    $self->batch_compute(evs);
  }

  %pythoncode %{

    def batch_compute(self, events, weights=None):

        if weights is None:
            weights = np.ones(len(events), order='C', dtype=np.double)
        elif len(weights) != len(events):
            raise ValueError('events and weights have different length')

        eecevents = EECEvents(len(events))
        for event,weight in zip(events, weights):
            eecevents.add_event(event, weight)

        self(eecevents)
  %}
}

// instantiate EEC templates
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
