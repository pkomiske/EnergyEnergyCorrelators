// -*- C++ -*-
//
// EnergyEnergyCorrelators - Evaluates EECs on particle physics events
// Copyright (C) 2020-2021 Patrick T. Komiske III
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
%include <std_array.i>
%include <std_string.i>
%include <std_vector.i>

// this makes SWIG aware of the types contained in the main fastjet library
// but does not generate new wrappers for them here
#ifdef SWIG_FASTJET
%import FASTJET_PREFIX/share/fastjet/pyinterface/fastjet.i

// turn off exception handling for now, since fastjet::Error is not thrown here
%exception;
#endif

%{
// include these to avoid needing to define them at compile time 
#ifndef SWIG
#define SWIG
#endif

// needed to ensure bytes are returned 
#define SWIG_PYTHON_STRICT_BYTE_CHAR

// EEC library headers
#include "EEC.hh"

typedef boost::histogram::algorithm::reduce_command reduce_command;

// using namespaces
using namespace eec;
using namespace eec::hist;
%}

// include numpy support
%include numpy_helpers.i

// additional numpy typemaps
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* particles, int mult, int nfeatures)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(const double* event_ptr, unsigned mult, unsigned nfeatures)}

// vector templates
%template(vectorDouble) std::vector<double>;
%template(vectorUnsigned) std::vector<unsigned>;

// array templates
%template(arrayUnsigned13) std::array<unsigned, 13>;

// allow threads in PairwiseEMD computation
%threadallow EECNAMESPACE::EECBase::_batch_compute;
%threadallow EECNAMESPACE::hist::EECHistBase::reduce;

// makes python class printable from a description method
%define ADD_REPR_FROM_DESCRIPTION
%pythoncode %{
  def __repr__(self):
      return self.description().decode('utf-8')
%}
%enddef

// for pickling
%define PYTHON_PICKLE_FUNCTIONS
  %pythoncode %{
    def __getstate__(self):
        return (self.__getstate_internal__(),)

    def __setstate__(self, state):
        self.__init__(*self._default_args)
        try:
            self.__setstate_internal__(state[0])
        except Exception as e:
            raise RuntimeError('issue loading eec - check `eec.get_archive_format()`'
                               ' and `eec.get_compression_mode()`',
                               repr(e))
  %}
%enddef

%define CPP_PICKLE_FUNCTIONS
  std::string __getstate_internal__() {
    std::ostringstream oss;
    $self->save(oss);
    return oss.str();
  }

  void __setstate_internal__(const std::string & state) {
    std::istringstream iss(state);
    $self->load(iss);
  }
%enddef

// for overloading operators in final derived class
%define DEFINE_OPERATORS(Class) 
  Class & operator*=(const double x) {
    $self->operator*=(x);
    return *$self;
  }
%enddef

// basic exception handling for all functions
%exception {
  try { $action }
  SWIG_CATCH_STDEXCEPT
}

namespace EECNAMESPACE {

// ignore EECUtils functions
%ignore get_coverage;
%ignore get_thread_num;

// ignore/rename EECHist functions
namespace hist {
  %ignore EECHistBase::combined_hist;
  %ignore EECHistBase::combined_covariance;
  %ignore EECHistBase::combined_variance_bound;
  %ignore EECHistBase::get_hist_vars;
  %ignore EECHistBase::get_covariance;
  %ignore EECHistBase::get_variance_bound;
  %ignore EECHistBase::operator+=;
  %ignore EECHistBase::operator*=;
  %rename(bin_centers_vec) EECHistBase::bin_centers;
  %rename(bin_edges_vec) EECHistBase::bin_edges;
  %rename(bin_centers) EECHistBase::npy_bin_centers;
  %rename(bin_edges) EECHistBase::npy_bin_edges;
  %rename(get_hist_vars) EECHist1D::npy_get_hist_vars;
  %rename(get_hist_vars) EECHist3D::npy_get_hist_vars;
  %rename(get_covariance) EECHist1D::npy_get_covariance;
  %rename(get_covariance) EECHist3D::npy_get_covariance;
  %rename(get_variance_bound) EECHist1D::npy_get_variance_bound;
  %rename(get_variance_bound) EECHist3D::npy_get_variance_bound;
}

// ignore/rename Multinomial functions
%ignore multinomial;
%rename(multinomial) multinomial_vector;

// ignore EEC functions
%ignore argsort3;
%ignore EECEvents::set_pseudojet_charge_func;
%ignore EECBase::batch_compute;
%ignore EECBase::compute;
%ignore EECBase::operator+=;
%ignore EECBase::set_pseudojet_charge_func;
%rename(compute) EECBase::npy_compute;

} // namespace EECNAMESPACE

// custom declaration of this struct because swig can't handle nested unions
namespace boost {
  namespace histogram {
    namespace algorithm {
      struct reduce_command {};

      reduce_command rebin(unsigned iaxis, unsigned merge);
      reduce_command rebin(unsigned merge);
      reduce_command shrink(unsigned iaxis, double lower, double upper);
      reduce_command shrink(double lower, double upper);
      reduce_command slice(unsigned iaxis, int begin, int end);
      reduce_command slice(int begin, int end);
      reduce_command shrink_and_rebin(unsigned iaxis, double lower, double upper, unsigned merge);
      reduce_command shrink_and_rebin(double lower, double upper, unsigned merge);
      reduce_command slice_and_rebin(unsigned iaxis, int begin, int end, unsigned merge);
      reduce_command slice_and_rebin(int begin, int end, unsigned merge);
    }
  }
}

%template(vectorReduceCommand) std::vector<boost::histogram::algorithm::reduce_command>;

// include EECHist and declare templates
%include "EECUtils.hh"
%include "EECHistBase.hh"
%include "EECHist1D.hh"
%include "EECHist3D.hh"

%define GET_HIST_TWO_QUANTITIES(cppfunc)
try {
  $self->cppfunc(*arr_out0, *arr_out1, hist_i, overflows);
}
catch (...) {
  free(*arr_out0);
  free(*arr_out1);
  throw;
}
%enddef

%define GET_HIST_ONE_QUANTITY(cppfunc)
try {
  $self->cppfunc(*arr_out0, hist_i, overflows);
}
catch (...) {
  free(*arr_out0);
  throw;
}
%enddef

namespace EECNAMESPACE {
  namespace hist {

    // extend EECHistBase
    %extend EECHistBase {
      void npy_bin_centers(double** arr_out0, int* n0, int i = 0) {
        COPY_1DARRAY_TO_NUMPY(arr_out0, n0, $self->nbins(i), nbytes, $self->bin_centers(i).data())
      }

      void npy_bin_edges(double** arr_out0, int* n0, int i = 0) {
        COPY_1DARRAY_TO_NUMPY(arr_out0, n0, $self->nbins(i)+1, nbytes, $self->bin_edges(i).data())
      }

      %pythoncode {
        def get_hist_errs(self, hist_i=0, overflows=True):
            hist, vars = self.get_hist_vars(hist_i, overflows)
            return hist, _np.sqrt(vars)

        def get_error_bound(self, hist_i=0, overflows=True):
            return _np.sqrt(self.get_variance_bound(hist_i, overflows))
      }
    }

    // extend EECHist1D code
    %extend EECHist1D {
      void npy_get_hist_vars(double** arr_out0, int* n0,
                             double** arr_out1, int* n1,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_1D_DOUBLE_ARRAY(arr_out0, n0, $self->hist_size(overflows), nbytes0)
        MALLOC_1D_DOUBLE_ARRAY(arr_out1, n1, $self->hist_size(overflows), nbytes1)
        GET_HIST_TWO_QUANTITIES(get_hist_vars)
      }

      void npy_get_covariance(double** arr_out0, int* n0, int* n1,
                              unsigned hist_i = 0, bool overflows = true) {
        std::size_t s($self->hist_size(overflows));
        MALLOC_2D_DOUBLE_ARRAY(arr_out0, n0, n1, s, s, nbytes0)
        GET_HIST_ONE_QUANTITY(get_covariance)
      }

      void npy_get_variance_bound(double** arr_out0, int* n0,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_1D_DOUBLE_ARRAY(arr_out0, n0, $self->hist_size(overflows), nbytes0)
        GET_HIST_ONE_QUANTITY(get_variance_bound)
      }
    }

    // extend EECHist3D code
    %extend EECHist3D {
      void npy_get_hist_vars(double** arr_out0, int* n0, int* n1, int* n2,
                             double** arr_out1, int* m0, int* m1, int* m2,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_3D_DOUBLE_ARRAY(arr_out0, n0, n1, n2, $self->hist_size(overflows, 0),
                                                     $self->hist_size(overflows, 1),
                                                     $self->hist_size(overflows, 2), nbytes0)
        MALLOC_3D_DOUBLE_ARRAY(arr_out1, m0, m1, m2, $self->hist_size(overflows, 0),
                                                     $self->hist_size(overflows, 1),
                                                     $self->hist_size(overflows, 2), nbytes1)
        GET_HIST_TWO_QUANTITIES(get_hist_vars)
      }

      void npy_get_covariance(double** arr_out0, int* n0, int* n1, int* n2, int* n3, int* n4, int* n5,
                          unsigned hist_i = 0, bool overflows = true) {
        MALLOC_6D_DOUBLE_ARRAY(arr_out0, n0, n1, n2, n3, n4, n5,
                                         $self->hist_size(overflows, 0),
                                         $self->hist_size(overflows, 1),
                                         $self->hist_size(overflows, 2),
                                         $self->hist_size(overflows, 0),
                                         $self->hist_size(overflows, 1),
                                         $self->hist_size(overflows, 2),
                               nbytes0)
        GET_HIST_ONE_QUANTITY(get_covariance)
      }

      void npy_get_variance_bound(double** arr_out0, int* n0, int* n1, int* n2,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_3D_DOUBLE_ARRAY(arr_out0, n0, n1, n2, $self->hist_size(overflows, 0),
                                                     $self->hist_size(overflows, 1),
                                                     $self->hist_size(overflows, 2), nbytes0)
        GET_HIST_ONE_QUANTITY(get_variance_bound)
      }
    }

    // declare histogram templates
    %template(EECHistBase1DId) EECHistBase<EECHist1D<axis::id>>;
    %template(EECHistBase1DLog) EECHistBase<EECHist1D<axis::log>>;
    %template(EECHistBaseIdIdId) EECHistBase<EECHist3D<axis::id, axis::id, axis::id>>;
    %template(EECHistBaseLogIdId) EECHistBase<EECHist3D<axis::log, axis::id, axis::id>>;
    %template(EECHistBaseIdLogId) EECHistBase<EECHist3D<axis::id, axis::log, axis::id>>;
    %template(EECHistBaseLogLogId) EECHistBase<EECHist3D<axis::log, axis::log, axis::id>>;
    %template(EECHist1DId) EECHist1D<axis::id>;
    %template(EECHist1DLog) EECHist1D<axis::log>;
    %template(EECHist3DIdIdId) EECHist3D<axis::id, axis::id, axis::id>;
    %template(EECHist3DLogIdId) EECHist3D<axis::log, axis::id, axis::id>;
    %template(EECHist3DIdLogId) EECHist3D<axis::id, axis::log, axis::id>;
    %template(EECHist3DLogLogId) EECHist3D<axis::log, axis::log, axis::id>;

  } // namespace hist
} // namespace EECNAMESPACE

// include EEC code and declare templates
%include "EECEvents.hh"
%include "EECBase.hh"
%include "EECMultinomial.hh"
%include "EECLongestSide.hh"
%include "EECTriangleOPE.hh"

namespace EECNAMESPACE {

  %extend Multinomial {

    template<unsigned i>
    void py_set_index(unsigned ind) {
      if (i == 0 || i >= $self->N() - 1)
        throw std::out_of_range("trying to set invalid index");
      $self->set_index<i>(ind);
    }
  }

  %extend EECBase {
    ADD_REPR_FROM_DESCRIPTION
    PYTHON_PICKLE_FUNCTIONS

    void npy_compute(double* particles, int mult, int nfeatures, double weight = 1.0, int thread_i = 0) {
      if (nfeatures != (int) $self->nfeatures()) {
        std::ostringstream oss;
        oss << "Got array with " << nfeatures << " features per particle, expected "
            << $self->nfeatures() << " features per particle";
        throw std::runtime_error(oss.str());
        return;
      }
      $self->compute(particles, mult, weight, thread_i);
    }

    // this is needed because we've hidden batch_compute
    void _batch_compute(const EECEvents & evs) {
      $self->batch_compute(evs);
    }

    %pythoncode %{

      def __call__(self, events, weights=None):

          if weights is None:
              weights = _np.ones(len(events), order='C', dtype=_np.double)
          elif len(weights) != len(events):
              raise ValueError('events and weights have different length')

          ncol = 4 if self.use_charges() else 3
          eecevents = EECEvents(len(events), self.nfeatures())
          events_arr = []
          for event,weight in zip(events, weights):
              event = _np.asarray(_np.atleast_2d(event)[:,:ncol], dtype=_np.double, order='C')
              eecevents.append(event, weight)
              events_arr.append(event)

          self._batch_compute(eecevents)
    %}
  }

  %extend EECLongestSide {
    CPP_PICKLE_FUNCTIONS
    ADD_REPR_FROM_DESCRIPTION
    DEFINE_OPERATORS(EECLongestSide)
    %pythoncode %{
      _default_args = (2, 1, 0.1, 1.0)
    %}
  }

  %extend EECTriangleOPE {
    CPP_PICKLE_FUNCTIONS
    ADD_REPR_FROM_DESCRIPTION
    DEFINE_OPERATORS(EECTriangleOPE)
    %pythoncode %{
      _default_args = (1, 0.1, 1.0, 1, 0.1, 1.0, 1, 0., 1.5)
    %}
  }

  // instantiate EEC templates
  %template(set_index_1) Multinomial::py_set_index<1>;
  %template(set_index_2) Multinomial::py_set_index<2>;
  %template(set_index_3) Multinomial::py_set_index<3>;
  %template(set_index_4) Multinomial::py_set_index<4>;
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

} // namespace EECNAMESPACE
