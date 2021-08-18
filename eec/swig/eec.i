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
%feature("autodoc", "1");
%nothreadallow;

// this can be used to ensure that swig parses classes correctly
#define SWIG_PREPROCESSOR

// C++ standard library wrappers
%include <exception.i>
%include <std_array.i>
%include <std_string.i>
%include <std_vector.i>

// ensure FASTJET_PREFIX is always defined even if not used
#ifndef FASTJET_PREFIX
# define FASTJET_PREFIX /usr/local
#endif

// import either pyfjcore or fastjet
#ifdef EEC_USE_PYFJCORE
  %import pyfjcore/swig/pyfjcore.i
  %pythoncode %{
    from pyfjcore import FastJetError
  %}
#else
  %import FASTJET_PREFIX/share/fastjet/pyinterface/fastjet.i
  %pythoncode %{
    from fastjet import FastJetError
  %}
#endif

// converts fastjet::Error into a FastJetError Python exception
FASTJET_ERRORS_AS_PYTHON_EXCEPTIONS(eec)

%{
// include these to avoid needing to define them at compile time 
#ifndef SWIG
# define SWIG
#endif

// needed to ensure bytes are returned 
#define SWIG_PYTHON_STRICT_BYTE_CHAR

// EEC library headers
#include "EEC.hh"

typedef boost::histogram::algorithm::reduce_command reduce_command;

// using namespaces
using namespace fastjet::contrib::eec;
using namespace fastjet::contrib::eec::hist;
%}

// this needs to be early so that the symbols are loaded from the dynamic library
%pythonbegin %{
  import pyfjcore
%}

// include numpy support
#define EEC_INT std::ptrdiff_t
%include numpy_helpers.i

// additional numpy typemaps
%apply (double* INPLACE_ARRAY1, EEC_INT DIM1) {
  (const double* raw_weights, unsigned weights_mult),
  (const double* charges, unsigned charges_mult)
}
%apply (double* INPLACE_ARRAY2, EEC_INT DIM1, EEC_INT DIM2) {
  (const double* event_ptr, unsigned mult, unsigned nfeatures),
  (const double* dists, unsigned d0, unsigned d1)
}

// makes python class printable from a description method
%define ADD_REPR_FROM_DECODED_DESCRIPTION
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

%define CPP_EECCOMP_FUNCTIONS(EECComp)
  std::string __getstate_internal__() {
    std::ostringstream oss;
    $self->save(oss);
    return oss.str();
  }

  void __setstate_internal__(const std::string & state) {
    std::istringstream iss(state);
    if (!iss.good())
      throw std::runtime_error("problem creating istringstream from provided state");

    $self->load(iss);
  }

  void add(const EECComp & rhs) {
    $self->operator+=(rhs);
  }
%enddef

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

// basic exception handling for all functions
%exception {
  try { $action }
  SWIG_CATCH_STDEXCEPT
  catch (...) {
    SWIG_exception_fail(SWIG_UnknownError, "unknown exception");
  }
}

// array templates
%template(arrayDouble2) std::array<double, 2>;
%template(arrayUnsigned3) std::array<unsigned, 3>;
%template(arrayUnsigned13) std::array<unsigned, 13>;
%template(arrayPairDoubleDouble) std::array<std::array<double, 2>, 3>;

// vector templates
%template(vectorDouble) std::vector<double>;
%template(vectorUnsigned) std::vector<unsigned>;
%template(vectorArrayDouble2) std::vector<std::array<double, 2>>;

// include EECUtils so that namespace is defined
%include "EECUtils.hh"

// allow threads in PairwiseEMD computation
%threadallow EEC_NAMESPACE::EECBase::batch_compute;
%threadallow EEC_NAMESPACE::hist::EECHistBase::reduce;

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

namespace EEC_NAMESPACE {

  // ignore/rename EECHist functions
  namespace hist {
    %ignore get_coverage;
    %ignore EECHistBase::EECHistBase;
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
} // namespace EEC_NAMESPACE

// include EECHist and declare templates
%include "EECHistBase.hh"
%include "EECHist1D.hh"
%include "EECHist3D.hh"

namespace EEC_NAMESPACE {
  namespace hist {

    // extend EECHistBase
    %extend EECHistBase {
      void npy_bin_centers(double** arr_out0, EEC_INT* n0, int i = 0) {
        COPY_1DARRAY_TO_NUMPY(arr_out0, n0, $self->nbins(i), nbytes, $self->bin_centers(i).data())
      }

      void npy_bin_edges(double** arr_out0, EEC_INT* n0, int i = 0) {
        COPY_1DARRAY_TO_NUMPY(arr_out0, n0, $self->nbins(i)+1, nbytes, $self->bin_edges(i).data())
      }

      void scale(double x) {
        $self->operator*=(x);
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
      void npy_get_hist_vars(double** arr_out0, EEC_INT* n0,
                             double** arr_out1, EEC_INT* n1,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_1D_DOUBLE_ARRAY(arr_out0, n0, $self->hist_size(overflows), nbytes0)
        MALLOC_1D_DOUBLE_ARRAY(arr_out1, n1, $self->hist_size(overflows), nbytes1)
        GET_HIST_TWO_QUANTITIES(get_hist_vars)
      }

      void npy_get_covariance(double** arr_out0, EEC_INT* n0, EEC_INT* n1,
                              unsigned hist_i = 0, bool overflows = true) {
        std::size_t s($self->hist_size(overflows));
        MALLOC_2D_DOUBLE_ARRAY(arr_out0, n0, n1, s, s, nbytes0)
        GET_HIST_ONE_QUANTITY(get_covariance)
      }

      void npy_get_variance_bound(double** arr_out0, EEC_INT* n0,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_1D_DOUBLE_ARRAY(arr_out0, n0, $self->hist_size(overflows), nbytes0)
        GET_HIST_ONE_QUANTITY(get_variance_bound)
      }
    }

    // extend EECHist3D code
    %extend EECHist3D {
      void npy_get_hist_vars(double** arr_out0, EEC_INT* n0, EEC_INT* n1, EEC_INT* n2,
                             double** arr_out1, EEC_INT* m0, EEC_INT* m1, EEC_INT* m2,
                             unsigned hist_i = 0, bool overflows = true) {
        MALLOC_3D_DOUBLE_ARRAY(arr_out0, n0, n1, n2, $self->hist_size(overflows, 0),
                                                     $self->hist_size(overflows, 1),
                                                     $self->hist_size(overflows, 2), nbytes0)
        MALLOC_3D_DOUBLE_ARRAY(arr_out1, m0, m1, m2, $self->hist_size(overflows, 0),
                                                     $self->hist_size(overflows, 1),
                                                     $self->hist_size(overflows, 2), nbytes1)
        GET_HIST_TWO_QUANTITIES(get_hist_vars)
      }

      void npy_get_covariance(double** arr_out0, EEC_INT* n0, EEC_INT* n1, EEC_INT* n2, EEC_INT* n3, EEC_INT* n4, EEC_INT* n5,
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

      void npy_get_variance_bound(double** arr_out0, EEC_INT* n0, EEC_INT* n1, EEC_INT* n2,
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
} // namespace EEC_NAMESPACE

namespace EEC_NAMESPACE {

  // ignore/rename Multinomial functions
  %ignore FACTORIALS_LONG;
  %ignore multinomial;
  %rename(multinomial) multinomial_vector;

  // ignore EEC functions
  %ignore argsort3;
  %ignore EECBase::operator+=;
  %ignore EECBase::operator();
  %ignore EECLongestSide::load;
  %ignore EECLongestSide::save;
  %ignore EECTriangleOPE::load;
  %ignore EECTriangleOPE::save;
  %rename(_compute) EECBase::compute;
  %rename(_push_back) EECBase::push_back;

} // namespace EEC_NAMESPACE

// include EEC code and declare templates
%include "EECEvent.hh"
%include "EECBase.hh"
%include "EECMultinomial.hh"
%include "EECLongestSide.hh"
%include "EECTriangleOPE.hh"

%inline %{
  EEC_NAMESPACE::EECEvent _event_from_pjc(const EEC_NAMESPACE::EECConfig & config,
                                          double event_weight,
                                          const fastjet::PseudoJetContainer & pjc,
                                          const std::vector<double> & charges) {
    return EEC_NAMESPACE::EECEvent(config, event_weight, pjc, charges);
  }
%}

namespace EEC_NAMESPACE {

  %extend Multinomial {

    template<unsigned i>
    void py_set_index(unsigned ind) {
      if (i == 0 || i >= $self->N() - 1)
        throw std::out_of_range("trying to set invalid index");
      $self->set_index<i>(ind);
    }
  }

  %extend EECBase {
    ADD_REPR_FROM_DECODED_DESCRIPTION
    PYTHON_PICKLE_FUNCTIONS

    %pythoncode %{

      def _create_eec_event(self, event, charges, dists, event_weight):
          if not len(event):
              return EECEvent()

          if charges is None:
              charges = []

          if isinstance(event, pyfjcore.PseudoJetContainer):
              eec_event = _event_from_pjc(self.config(), event_weight, event, charges)

          elif isinstance(event[0], pyfjcore.PseudoJet):
              eec_event = EECEvent(self.config(), event_weight, event, charges)

          elif dists is None:
              event = _np.asarray(_np.atleast_2d(event)[:,:self.nfeatures()], dtype=_np.double, order='C')

              eec_event = EECEvent(self.use_charges(), event_weight, event)
              eec_event._numpy_arrays = (event,)

          else:
              raw_weights = _np.asarray(event, dtype=_np.double, order='C')
              charges = _np.asarray(charges, dtype=_np.double, order='C')
              dists = _np.asarray(dists, dtype=_np.double, order='C')

              eec_event = EECEvent(self.use_charges(), event_weight, raw_weights, charges, dists)
              eec_event._numpy_arrays = (raw_weights, dists, charges)

          return eec_event

      def compute(self, event, event_weight=1.0, charges=None, dists=None, thread=0):
          eec_event = self._create_eec_event(event, charges, dists, event_weight)
          self._compute(eec_event, thread)

      def __call__(self, events, event_weights=None, charges=None, dists=None):

          if event_weights is None:
              event_weights = _np.ones(len(events), order='C', dtype=_np.double)
          elif len(event_weights) != len(events):
              raise ValueError('`events` and `event_weights` have different lengths')

          if charges is None:
              charges = len(events)*[None]
          elif len(charges) != len(events):
              raise ValueError('`events` and `charges` have different lengths')

          if dists is None:
              dists = len(events)*[None]
          elif len(dists) != len(events):
              raise ValueError('`events` and `dists` have different lengths')

          stored_events = []
          for event, chs, ds, event_weight in zip(events, charges, dists, event_weights):
              eec_event = self._create_eec_event(event, chs, ds, event_weight)
              self._push_back(eec_event)
              stored_events.append(eec_event)

          self.batch_compute()
          self.clear_events()
    %}
  }

  %extend EECLongestSide {
    CPP_EECCOMP_FUNCTIONS(EECLongestSide)
    ADD_REPR_FROM_DECODED_DESCRIPTION
    %pythoncode %{
      _default_args = (2, 1)
    %}
  }

  %extend EECTriangleOPE {
    CPP_EECCOMP_FUNCTIONS(EECTriangleOPE)
    ADD_REPR_FROM_DECODED_DESCRIPTION
    %pythoncode %{
      _default_args = ((1, 1, 1),)
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

} // namespace EEC_NAMESPACE
