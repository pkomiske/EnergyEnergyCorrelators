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
 *   ____           _____ ______ 
 *  |  _ \   /\    / ____|  ____|
 *  | |_) | /  \  | (___ | |__   
 *  |  _ < / /\ \  \___ \|  __|  
 *  | |_) / ____ \ ____) | |____ 
 *  |____/_/    \_\_____/|______|
 */

#ifndef EEC_BASE_HH
#define EEC_BASE_HH

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "EECUtils.hh"

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

public:

  EECEvents(std::size_t nev = 0, unsigned nfeatures = 0) : nfeatures_(nfeatures) {
    events_.reserve(nev);
    mults_.reserve(nev);
    weights_.reserve(nev);
  }

  // access functions
  const std::vector<const double *> & events() const { return events_; }
  const std::vector<unsigned> & mults() const { return mults_; }
  const std::vector<double> & weights() const { return weights_; }

  // add event 
  void append(const double * event_ptr, unsigned mult, unsigned nfeatures,  double weight) {
    if (nfeatures_ > 0 && nfeatures != nfeatures_) {
      std::ostringstream oss;
      oss << "event has " << nfeatures << " features per particle, expected "
          << nfeatures_ << " features per particle";
      throw std::invalid_argument(oss.str());
    }

    events_.push_back(event_ptr);
    mults_.push_back(mult);
    weights_.push_back(weight);
  }
  void append(const double * event_ptr, unsigned mult) {
    events_.push_back(event_ptr);
    mults_.push_back(mult);
  }

}; // EECEvents

//-----------------------------------------------------------------------------
// Base class for all EEC computations
//-----------------------------------------------------------------------------

class EECBase {
private:

  // the pt and charge powers
  std::vector<double> orig_pt_powers_, pt_powers_;
  std::vector<unsigned> orig_ch_powers_, ch_powers_;

  // number of particles to correlate, features per particle, unique weights to compute
  // 3 by default, charge optional (pt, y, phi, [charge])
  unsigned N_, nsym_, nfeatures_, event_counter_;
  bool norm_, use_charges_, check_degen_, average_verts_;
  int num_threads_, print_every_, omp_chunksize_;
  std::ostream * print_stream_;
  std::ostringstream oss_;
  std::chrono::steady_clock::time_point start_time_;

  // name of method used for core computation
  std::string compname_;

#ifdef __FASTJET_PSEUDOJET_HH__
  double (*pj_charge_)(const fastjet::PseudoJet &);
#endif

  // vectors used by the computations (outer axis is the thread axis)
  std::vector<std::vector<std::vector<double>>> weights_;
  std::vector<std::vector<double>> dists_;
  std::vector<double> event_weights_;
  std::vector<unsigned> mults_;

#ifdef EEC_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & orig_pt_powers_ & pt_powers_ & orig_ch_powers_ & ch_powers_
       & N_ & nsym_ & nfeatures_ & event_counter_
       & norm_ & use_charges_ & check_degen_ & average_verts_
       & num_threads_ & print_every_ & omp_chunksize_;

    init();
  }
#endif

  void init() {
    set_print_stream(std::cout);

    oss_ = std::ostringstream(std::ios_base::ate);
    oss_.setf(std::ios_base::fixed, std::ios_base::floatfield);

    weights_.resize(num_threads(), std::vector<std::vector<double>>(N()));
    dists_.resize(num_threads());
    event_weights_.resize(num_threads());
    mults_.resize(num_threads());
  }

protected:

  // method that carries out the specific EEC computation
  virtual void compute_eec(int thread_i) = 0;

  // access to vector storage by derived classes
  const std::vector<std::vector<double>> & weights(int thread_i) const { return weights_[thread_i]; }
  const std::vector<double> & dists(int thread_i) const { return dists_[thread_i]; }
  double event_weight(int thread_i) const { return event_weights_[thread_i]; }
  unsigned mult(int thread_i) const { return mults_[thread_i]; }

public:

  EECBase(unsigned N, bool norm,
          const std::vector<double> & pt_powers, const std::vector<unsigned> & ch_powers,
          int num_threads, int print_every, bool check_degen, bool average_verts) : 
    orig_pt_powers_(pt_powers), orig_ch_powers_(ch_powers),
    N_(N), nsym_(N_), event_counter_(0),
    norm_(norm), use_charges_(false), check_degen_(check_degen), average_verts_(average_verts),
    num_threads_(determine_num_threads(num_threads)),
    print_every_(print_every)
  {

    // initialize data members
    init();

    // set default thread chunksize
    set_omp_chunksize(10);
    

    if (orig_pt_powers_.size() == 1)
      orig_pt_powers_ = std::vector<double>(N_, orig_pt_powers_[0]);
    if (orig_ch_powers_.size() == 1)
      orig_ch_powers_ = std::vector<unsigned>(N_, orig_ch_powers_[0]);

    if (orig_pt_powers_.size() != N_)
      throw std::invalid_argument("pt_powers must be a vector of size 1 or " + std::to_string(N_));
    if (orig_ch_powers_.size() != N_)
      throw std::invalid_argument("ch_powers must be a vector of size 1 or " + std::to_string(N_));

    // copy original powers, because these will be modified by how many symmetries there are
    pt_powers_ = orig_pt_powers_;
    ch_powers_ = orig_ch_powers_;

    // check for symmetries in the different cases
    switch (N_) {
      case 2: {
        if (pt_powers_[0] == pt_powers_[1] && ch_powers_[0] == ch_powers_[1]) {
          pt_powers_ = {pt_powers_[0]};
          ch_powers_ = {ch_powers_[0]};
          compname_ = "eec_ij_sym";
        }
        else {
          compname_ = "eec_no_sym";
          nsym_ = 0;
        }
        break;
      }

      case 3: {
        bool match01(pt_powers_[0] == pt_powers_[1] && ch_powers_[0] == ch_powers_[1]),
             match12(pt_powers_[1] == pt_powers_[2] && ch_powers_[1] == ch_powers_[2]),
             match02(pt_powers_[0] == pt_powers_[2] && ch_powers_[0] == ch_powers_[2]);
        if (match01 && match12) {
          pt_powers_ = {pt_powers_[0]};
          ch_powers_ = {ch_powers_[0]};
          compname_ = "eeec_ijk_sym";
        }
        else if (match01) {
          pt_powers_ = {pt_powers_[0], pt_powers_[2]};
          ch_powers_ = {ch_powers_[0], ch_powers_[2]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else if (match12) {
          pt_powers_ = {pt_powers_[1], pt_powers_[0]};
          ch_powers_ = {ch_powers_[1], ch_powers_[0]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else if (match02) {
          pt_powers_ = {pt_powers_[2], pt_powers_[1]};
          ch_powers_ = {ch_powers_[2], ch_powers_[1]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else {
          compname_ = "eeec_no_sym";
          nsym_ = 0;
        }
        break;
      }

      case 4: {
        for (int i = 1; i < 4; i++) {
          if (pt_powers_[i] != pt_powers_[0] || ch_powers_[i] != ch_powers_[0])
            throw std::invalid_argument("N = 4 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeec_ijkl_sym";
        break;
      }

      case 5: {
        for (int i = 1; i < 5; i++) {
          if (pt_powers_[i] != pt_powers_[0] || ch_powers_[i] != ch_powers_[0])
            throw std::invalid_argument("N = 5 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeeec_ijklm_sym";
        break;
      }

      case 6: {
        for (int i = 1; i < 6; i++) {
          if (pt_powers_[i] != pt_powers_[0] || ch_powers_[i] != ch_powers_[0])
            throw std::invalid_argument("N = 6 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeeeec_ijklm_sym";
        break;
      }

      default:
        for (unsigned i = 1; i < N_; i++) {
          if (pt_powers_[i] != pt_powers_[0] || ch_powers_[i] != ch_powers_[0])
            throw std::invalid_argument("this N only supports the fully symmetric correlator currently");
        }
        compname_ = "eNc_sym";
    }

    // check for using charges at all
    for (int ch_power : ch_powers_)
      if (ch_power != 0)
        use_charges_ = true;
    nfeatures_ = (use_charges_ ? 4 : 3);

    // initialize this to null in case we have FastJet support
    #ifdef __FASTJET_PSEUDOJET_HH__
    pj_charge_ = nullptr;
    #endif   
  }
  virtual ~EECBase() = default;

  virtual std::string description(int = 1) const {
    std::ostringstream oss;
    oss << std::boolalpha
        << compname_ << '\n'
        << "  N - " << N() << '\n'
        << "  norm - " << norm_ << '\n'
        << "  use_charges - " << use_charges_ << '\n'
        << "  nfeatures - " << nfeatures() << '\n'
        << "  check_for_degeneracy - " << check_degen_ << '\n'
        << "  average_verts - " << average_verts_;

    // record pt and charge powers
    oss << '\n'
        << "  pt_powers - (" << pt_powers_[0];
    for (unsigned i = 1; i < orig_pt_powers_.size(); i++)
      oss << ", " << orig_pt_powers_[i];
    oss << ")\n"
        << "  ch_powers - (" << ch_powers_[0];
    for (unsigned i = 1; i < orig_ch_powers_.size(); i++)
      oss << ", " << orig_ch_powers_[i];
    oss << ")\n";

    return oss.str();
  }

  // access the number of threads
  unsigned N() const { return N_; }
  unsigned nsym() const { return nsym_; }
  unsigned nfeatures() const { return nfeatures_; }
  unsigned event_counter() const { return event_counter_; }
  int num_threads() const { return num_threads_; }
  int print_every() const { return print_every_; }
  bool norm() const { return norm_; }
  bool use_charges() const { return use_charges_; }
  bool average_verts() const { return average_verts_; }

  // externally set the number of EEC evaluations that will be spooled to each OpenMP thread at a time
  void set_omp_chunksize(int chunksize) { omp_chunksize_ = std::abs(chunksize); }
  void set_print_every(int print_every) { print_every_ = print_every; }
  void set_print_stream(std::ostream & os) { print_stream_ = &os; }

  // compute on a vector of events (themselves vectors of particles)
  void batch_compute(const std::vector<std::vector<double>> & events,
                     const std::vector<double> & weights) {
    if (events.size() != weights.size())
      throw std::runtime_error("events and weights are different sizes");

    EECEvents evs(events.size());
    for (unsigned i = 0; i < events.size(); i++) {
      evs.append(events[i].data(), events[i].size()/nfeatures());
      if (events[i].size() % nfeatures() != 0)
        throw std::runtime_error("incorrect particles size");
    }

    batch_compute(evs.events(), evs.mults(), weights);
  }

  void batch_compute(const EECEvents & evs) {
    batch_compute(evs.events(), evs.mults(), evs.weights());
  }

  // compute on a vector of events (pointers to arrays of particles)
  void batch_compute(const std::vector<const double *> & events,
                     const std::vector<unsigned> & mults,
                     const std::vector<double> & weights) {
  
    // check number of events
    long long nevents(events.size());
    if (events.size() != mults.size())
      throw std::runtime_error("events and mults are different sizes");
    if (events.size() != weights.size())
      throw std::runtime_error("events and weights are different sizes");

    // handle print_every
    int print_every(print_every_ == 0 ? -1 : print_every_);
    if (print_every < 0) {
      print_every = nevents/std::abs(print_every);
      if (print_every == 0 || (print_every_ != 0 && nevents % std::abs(print_every_) != 0))
        print_every++;
    }

    long long start(0), counter(0);
    start_time_ = std::chrono::steady_clock::now();
    while (counter < nevents) {
      counter += print_every;
      if (counter > nevents) counter = nevents;

      #pragma omp parallel for num_threads(num_threads()) default(shared) schedule(dynamic, omp_chunksize_)
      for (long long i = start; i < counter; i++) {
        compute(events[i], mults[i], weights[i], get_thread_num());
      }

      // update and do printing
      start = counter;
      print_update(counter, nevents);
    }
  }

  // compute on a vector of particles
  void compute(const std::vector<double> & particles, double weight = 1.0, int thread_i = 0) {
    if (particles.size() % nfeatures() != 0)
      throw std::runtime_error("incorrect particles size");

    compute(particles.data(), particles.size()/nfeatures(), weight, thread_i);
  }

  // compute on a C array of particles
  void compute(const double * particles, unsigned mult, double weight = 1.0, int thread_i = 0) {

    // store event weight and multiplicity
    event_weights_[thread_i] = weight;
    mults_[thread_i] = mult;
    
    // compute pairwise distances and extract pts
    std::vector<double> & dists(dists_[thread_i]);
    dists.resize(mult*mult);
    std::vector<double> pts(mult);
    for (unsigned i = 0; i < mult; i++) {
      unsigned ixm(i*mult), ixnf(i*nfeatures());

      // store pt
      pts[i] = particles[ixnf];

      // zero out diagonal
      dists[ixm + i] = 0;

      double y_i(particles[ixnf + 1]), phi_i(particles[ixnf + 2]);
      for (unsigned j = 0; j < i; j++) {
        unsigned jxnf(j*nfeatures());
        double ydiff(y_i - particles[jxnf + 1]), 
               phidiffabs(std::fabs(phi_i - particles[jxnf + 2]));

        // ensure that the phi difference is properly handled
        if (phidiffabs > PI) phidiffabs -= TWOPI;
        dists[ixm + j] = dists[j * mult + i] = std::sqrt(ydiff*ydiff + phidiffabs*phidiffabs);
      }
    }

    // store charges
    std::vector<double> charges;
    if (use_charges_) {
      charges.resize(mult);
      for (unsigned i = 0; i < mult; i++)
        charges[i] = particles[i*nfeatures() + 3];
    }

    // check for degeneracy
    if (check_degen_) {
      std::unordered_set<double> dists_set;
      unsigned ndegen(0);
      for (unsigned i = 0; i < mult; i++) {
        unsigned ixm(i*mult);
        for (unsigned j = 0; j < i; j++) {
          auto x(dists_set.insert(dists[ixm + j]));
          if (!x.second) {
            if (ndegen++ == 0)
              std::cerr << "Begin Event\n";
            std::cerr << "  distance degeneracy encountered, particles " 
                      << i << " and " << j << ", distance is " << *x.first << std::endl;
          }
        }
      }
      if (ndegen > 0)
        std::cerr << "End Event\n";
    }

    // run actual computation
    else {

      // fill weights with powers of pts and charges
      set_weights(pts, charges, thread_i);

      // delegate EEC computation to subclass
      compute_eec(thread_i);
    }

    #pragma omp atomic
    event_counter_++;
  }

  // fastjet support
#ifdef __FASTJET_PSEUDOJET_HH__
  void set_pseudojet_charge_func(double (*pj_charge)(const fastjet::PseudoJet &)) {
    pj_charge_ = pj_charge;
  }

  void compute(const std::vector<fastjet::PseudoJet> & pjs, const double weight = 1.0) {

    event_weights_[0] = weight;
    mults_[0] = pjs.size();

    // compute pairwise distances and extract pts
    std::vector<double> dists(dists_[0]);
    dists.resize(mults_[0]*mults_[0]);
    std::vector<double> pts(mults_[0]);
    for (unsigned i = 0; i < mults_[0]; i++) {
      unsigned ixm(i*mults_[0]);
      pts[i] = pjs[i].pt();
      dists[ixm + i] = 0;

      for (unsigned j = 0; j < i; j++)
        dists[ixm + j] = dists[j * mults_[0] + i] = pjs[i].delta_R(pjs[j]);
    }

    // store charges if provided a function to do so
    std::vector<double> charges;
    if (use_charges_) {
      if (pj_charge_ == nullptr)
        throw std::runtime_error("No function provided to get charges from PseudoJets");
      
      charges.resize(pjs.size());
      for (unsigned i = 0; i < mults_[0]; i++)
        charges[i] = pj_charge_(pjs[i]);  
    }

    // fill weights with powers of pts and charges
    set_weights(pts, charges, 0);

    // delegate EEC computation to subclass
    compute_eec(0);
    event_counter_++;
  }
#endif // __FASTJET_PSEUDOJET_HH__

private:

  // get weights as powers of pts and charges
  void set_weights(std::vector<double> & pts, const std::vector<double> & charges, int thread_i) {

    // normalize pts
    if (norm_) {
      double pttot(0);
      for (double pt : pts) pttot += pt;
      for (double & pt : pts) pt /= pttot;
    }

    unsigned mult(mults_[thread_i]);
    for (unsigned i = 0, npowers = pt_powers_.size(); i < npowers; i++) {
      std::vector<double> & weights(weights_[thread_i][i]);
      weights.resize(mult);

      // set weights to pt^ptpower
      double pt_power(pt_powers_[i]);
      if (pt_power == 1)
        for (unsigned j = 0; j < mult; j++)
          weights[j] = pts[j];
      else if (pt_power == 2)
        for (unsigned j = 0; j < mult; j++)
          weights[j] = pts[j]*pts[j];
      else if (pt_power == 0.5)
        for (unsigned j = 0; j < mult; j++)
          weights[j] = std::sqrt(pts[j]);
      else
        for (unsigned j = 0; j < mult; j++)
          weights[j] = std::pow(pts[j], pt_power);

      // include effect of charge
      if (use_charges_) {
        double ch_power(ch_powers_[i]);
        if (ch_power == 0) {}
        else if (ch_power == 1)
          for (unsigned j = 0; j < mult; j++)
            weights[j] *= charges[j];
        else if (ch_power == 2)
          for (unsigned j = 0; j < mult; j++)
            weights[j] *= charges[j]*charges[j];
        else
          for (unsigned j = 0; j < mult; j++)
            weights[j] *= std::pow(charges[j], ch_power);
      }
    }
  }

  void print_update(long long counter, long long nevents) {

    auto diff(std::chrono::steady_clock::now() - start_time_);
    double duration(std::chrono::duration_cast<std::chrono::duration<double>>(diff).count());

    unsigned nevents_width(std::to_string(nevents).size());
    oss_.str("  ");
    oss_ << std::setw(nevents_width) << counter << " / "
         << std::setw(nevents_width) << nevents << "  EMDs computed  - "
         << std::setprecision(2) << std::setw(6) << double(counter)/nevents*100
         << "% completed - "
         << std::setprecision(3) << duration << 's';

    // acquire Python GIL if in SWIG in order to check for signals and print message
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      if (print_every_ != 0) *print_stream_ << oss_.str() << std::endl;
      if (PyErr_CheckSignals() != 0)
        throw std::runtime_error("KeyboardInterrupt received in PairwiseEMD::compute");
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      if (print_every_ != 0) *print_stream_ << oss_.str() << std::endl;
    #endif
  }

}; // EECBase

} // namespace eec

#endif // EEC_BASE_HH