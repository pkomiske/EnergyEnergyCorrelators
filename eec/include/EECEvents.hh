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

/*   ______ ______ _____ 
 *  |  ____|  ____/ ____|
 *  | |__  | |__ | |     
 *  |  __| |  __|| |     
 *  | |____| |___| |____ 
 *  |______|______\_____|
 *   ________      ________ _   _ _______ _____ 
 *  |  ____\ \    / /  ____| \ | |__   __/ ____|
 *  | |__   \ \  / /| |__  |  \| |  | | | (___  
 *  |  __|   \ \/ / |  __| | . ` |  | |  \___ \ 
 *  | |____   \  /  | |____| |\  |  | |  ____) |
 *  |______|   \/   |______|_| \_|  |_| |_____/ 
 */

#ifndef EEC_EVENTS_HH
#define EEC_EVENTS_HH

#include <algorithm>
#include <cstddef>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "EECUtils.hh"

#if defined(EEC_FASTJET_SUPPORT) && !defined(SWIG_PREPROCESSOR)
namespace fastjet {

class UserInfoPython : public fastjet::PseudoJet::UserInfoBase {
public:
  UserInfoPython(PyObject * pyobj) : _pyobj(pyobj) {
    Py_XINCREF(_pyobj);
  }

  PyObject * get_pyobj() const {
    // since there's going to be an extra reference to this object
    // one must increase the reference count; it seems that this
    // is _our_ responsibility
    Py_XINCREF(_pyobj);
    return _pyobj;
  }
  //const PyObject * get_pyobj() const {return _pyobj;}
  
  ~UserInfoPython() {
    Py_XDECREF(_pyobj);
  }
private:
  PyObject * _pyobj;
};

} // namespace fastjet
#endif // EEC_FASTJET_SUPPORT && !SWIG_PREPROCESSOR

#define EEC_IGNORED_WEIGHT -1

namespace eec {

//-----------------------------------------------------------------------------
// Class to help store multiple events for multithreaded computation
//-----------------------------------------------------------------------------

class EECEvents {
private:

  std::vector<const double *> events_;
  std::vector<unsigned> mults_;
  std::vector<double> weights_;

  unsigned nfeatures_;

#ifdef EEC_FASTJET_SUPPORT

  // vector to hold values extracted from fastjet PseudoJets
  std::vector<std::vector<double>> fj_events_;

public:
  typedef fastjet::PseudoJet pj_charge_t;

// no PseudoJet, so just declare a type that will compile
#else
public:
  typedef int pj_charge_t;

#endif // EEC_FASTJET_SUPPORT

private:

  // pointer to function to get charge from a PseudoJet
  std::function<double(const pj_charge_t &)> pj_charge_;

public:

  // constructor just from a size and number of features
  EECEvents(std::size_t nev, unsigned nfeatures, double (*pj_charge)(const pj_charge_t &) = nullptr) : 
    nfeatures_(nfeatures)
  {
    if (nfeatures_ != 3 && nfeatures_ != 4)
      throw std::invalid_argument("nfeatures should be 3 or 4");

    events_.reserve(nev);
    mults_.reserve(nev);
    weights_.reserve(nev);

  #ifdef EEC_FASTJET_SUPPORT
    fj_events_.reserve(nev);
  #endif

    set_pseudojet_charge_func(pj_charge);
  }

  // constructor taking a vector of events and weights as input
  template<class T>
  EECEvents(const std::vector<std::vector<T>> & events,
            unsigned nfeatures,
            const std::vector<double> & weights = {},
            double (*pj_charge)(const pj_charge_t &) = nullptr) :
    EECEvents(events.size(), nfeatures, pj_charge)
  {
    operator()(events, weights);
  }

  // access functions
  const std::vector<const double *> & events() const { return events_; }
  const std::vector<unsigned> & mults() const { return mults_; }
  const std::vector<double> & weights() const { return weights_; }

  // add vector of events (events must live for as long as the EECEvents object)
  template<class T>
  EECEvents & operator()(const std::vector<std::vector<T>> & events,
                         const std::vector<double> & weights = {}) {

    // empty weights means use weight 1 for all events
    if (weights.size() == 0) {
      weights_.resize(events.size());
      std::fill(weights_.begin(), weights_.end(), 1.0);
    }

    // check that events and weights are the same size
    else if (events.size() != weights.size())
      throw std::runtime_error("events and weights are different sizes");

    else weights_ = weights;

    for (const T & event : events)
      append(event, EEC_IGNORED_WEIGHT);

    return *this;
  }

  // add single event 
  void append(const double * event_ptr, unsigned mult, unsigned nfeatures, double weight = 1.0) {
    if (nfeatures_ > 0 && nfeatures != nfeatures_) {
      std::ostringstream oss;
      oss << "event has " << nfeatures << " features per particle, expected "
          << nfeatures_ << " features per particle";
      throw std::invalid_argument(oss.str());
    }

    events_.push_back(event_ptr);
    mults_.push_back(mult);
    if (weight != EEC_IGNORED_WEIGHT) weights_.push_back(weight);
  }

  void append(const std::vector<double> & event, double weight = 1.0) {
    if (event.size() % nfeatures_ != 0)
      throw std::runtime_error("nfeatures does not divide event length evenly");
    append(event.data(), event.size()/nfeatures_, nfeatures_, weight);
  }

  void set_pseudojet_charge_func(double (*pj_charge)(const pj_charge_t &)) {
    pj_charge_ = pj_charge;

  #ifdef EEC_FASTJET_SUPPORT
    #ifdef SWIG

      // provide default function in case of Python
      if (!pj_charge_) {

        // define default python function to get charge from pseudojet
        pj_charge_ = [](const fastjet::PseudoJet & pj) {

          // check if pj has python user info
          if (!pj.has_user_info<fastjet::UserInfoPython>())
            throw std::runtime_error("PseudoJet does not have any Python user info");

          // get python info 
          PyObject * py_info(pj.user_info<fastjet::UserInfoPython>().get_pyobj());
          bool first_pass(true);

        numeric_check:

          // if float, use value
          if (PyFloat_Check(py_info))
            return PyFloat_AS_DOUBLE(py_info);

          // if int, convert to double
          if (PyLong_Check(py_info))
            return PyLong_AsDouble(py_info);

          if (!first_pass) goto failure;
          first_pass = false;

          // if list, use element 0
          if (PyList_Check(py_info)) {
            if (PyList_GET_SIZE(py_info) > 0) {
              py_info = PyList_GET_ITEM(py_info, 0);
              goto numeric_check;
            }
          }

          // if tuple, use element 0
          else if (PyTuple_Check(py_info)) {
            if (PyTuple_GET_SIZE(py_info) > 0) {
              py_info = PyTuple_GET_ITEM(py_info, 0);
              goto numeric_check;
            }
          }

          // if dict, use key "charge"
          else if (PyDict_Check(py_info)) {
            py_info = PyDict_GetItemString(py_info, "charge");
            if (py_info != NULL) goto numeric_check;
          }

        failure:
          throw std::invalid_argument("cannot extract charge from PseudoJet UserPythonInfo");
          return 0.;
        };

        // exit since we don't want to check nfeatures
        return;
      }

    #endif // SWIG

    // some sanity check that we have a function if we expected one
    if (pj_charge_ && nfeatures_ != 4)
      throw std::runtime_error("nfeatures should be 4 if using charges");
    if (!pj_charge_ && nfeatures_ != 3)
      throw std::runtime_error("nfeatures should be 3 if not using charges");

  #endif // EEC_FASTJET_SUPPORT
  }

#ifdef EEC_FASTJET_SUPPORT

  // process a fastjet event into internal representation
  void append(const std::vector<fastjet::PseudoJet> & pjs, double weight = 1.0) {

    // add new vector of doubles
    fj_events_.emplace_back(pjs.size()*nfeatures_);

    events_.push_back(fj_events_.back().data());
    mults_.push_back(pjs.size());
    if (weight != EEC_IGNORED_WEIGHT) weights_.push_back(weight);

    // extract info from PseudoJet
    std::size_t i(0);
    for (const fastjet::PseudoJet & pj : pjs) {
      fj_events_.back()[i++] = pj.pt();
      fj_events_.back()[i++] = pj.rap();
      fj_events_.back()[i++] = pj.phi();
      if (nfeatures_ == 4) fj_events_.back()[i++] = pj_charge_(pj);
    }
  }

#endif // EEC_FASTJET_SUPPORT

}; // EECEvents

} // namespace eec

#endif // EEC_EVENTS_HH
