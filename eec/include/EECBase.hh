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

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "EECUtils.hh"
#include "EECEvents.hh"

namespace eec {

//-----------------------------------------------------------------------------
// Base class for all EEC computations
//-----------------------------------------------------------------------------

class EECBase {
private:

  // the pt and charge powers
  std::vector<double> orig_pt_powers_, pt_powers_;
  std::vector<unsigned> orig_ch_powers_, ch_powers_;

  // details of the EEC computation
  unsigned N_, nsym_, nfeatures_;
  bool norm_, use_charges_, check_degen_, average_verts_;
  int num_threads_, omp_chunksize_;
  long print_every_;
  std::string compname_;
  double total_weight_;

  // internal vectors used by the computations (outer axis is the thread axis)
  std::vector<std::vector<std::vector<double>>> weights_;
  std::vector<std::vector<double>> dists_;
  std::vector<double> event_weights_;
  std::vector<unsigned> mults_;

  // printing/tracking variables
  std::ostream * print_stream_;
  std::ostringstream oss_;

public:

  EECBase(unsigned N, bool norm,
          const std::vector<double> & pt_powers, const std::vector<unsigned> & ch_powers,
          int num_threads, long print_every, bool check_degen, bool average_verts) : 
    orig_pt_powers_(pt_powers), orig_ch_powers_(ch_powers),
    N_(N), nsym_(N_),
    norm_(norm), use_charges_(false), check_degen_(check_degen), average_verts_(average_verts),
    num_threads_(determine_num_threads(num_threads)),
    print_every_(print_every),
    total_weight_(0)
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
    nfeatures_ = (use_charges() ? 4 : 3); 
  }
  virtual ~EECBase() = default;

  // access computation details
  unsigned N() const { return N_; }
  unsigned nsym() const { return nsym_; }
  unsigned nfeatures() const { return nfeatures_; }
  bool norm() const { return norm_; }
  bool use_charges() const { return use_charges_; }
  bool check_degen() const { return check_degen_; }
  bool average_verts() const { return average_verts_; }
  int num_threads() const { return num_threads_; }
  long print_every() const { return print_every_; }
  double total_weight() const { return total_weight_; }

  // get/set some computation options
  int get_omp_chunksize() const { return omp_chunksize_; }
  void set_omp_chunksize(int chunksize) { omp_chunksize_ = std::abs(chunksize); }
  void set_print_stream(std::ostream & os) { print_stream_ = &os; }

  virtual std::string description(int = 1) const {
    std::ostringstream oss;
    oss << std::boolalpha
        << compname_ << '\n'
        << "  N - " << N() << '\n'
        << "  norm - " << norm() << '\n'
        << "  use_charges - " << use_charges() << '\n'
        << "  nfeatures - " << nfeatures() << '\n'
        << "  check_for_degeneracy - " << check_degen() << '\n'
        << "  average_verts - " << average_verts() << '\n'
        << "  print_every - " << print_every() << '\n'
        << "  num_threads - " << num_threads() << '\n';

    // record pt and charge powers
    oss << "  pt_powers - (" << pt_powers_[0];
    for (unsigned i = 1; i < orig_pt_powers_.size(); i++)
      oss << ", " << orig_pt_powers_[i];
    oss << ")\n"
        << "  ch_powers - (" << ch_powers_[0];
    for (unsigned i = 1; i < orig_ch_powers_.size(); i++)
      oss << ", " << orig_ch_powers_[i];
    oss << ")\n";

    return oss.str();
  }

  // allow EECs to be added together
  EECBase & operator+=(const EECBase & rhs) {

    auto error(std::invalid_argument("EEC computations must match in order to be added together"));
    if (N()             != rhs.N()            ||
        nsym()          != rhs.nsym()         ||
        use_charges()   != rhs.use_charges()  ||
        check_degen()   != rhs.check_degen()  ||
        average_verts() != rhs.average_verts())
      throw error;

    for (unsigned i = 0; i < N(); i++) {
      if (pt_powers_[i] != rhs.pt_powers_[i] || 
          ch_powers_[i] != rhs.ch_powers_[i])
        throw error;
    }

    total_weight_ += rhs.total_weight();

    return *this;
  }

  // run batch_compute from EECEvents object
  void batch_compute(const EECEvents & evs) {
    batch_compute(evs.events(), evs.mults(), evs.weights());
  }

  // compute on a vector of events (pointers to arrays of particles)
  void batch_compute(const std::vector<const double *> & events,
                     const std::vector<unsigned> & mults,
                     const std::vector<double> & weights) {
  
    // check number of events
    long nevents(events.size());
    if (events.size() != mults.size())
      throw std::runtime_error("events and mults are different sizes");
    if (events.size() != weights.size())
      throw std::runtime_error("events and weights are different sizes");

    // handle print_every
    long print_every(print_every_ == 0 ? -1 : print_every_);
    if (print_every < 0) {
      print_every = nevents/std::abs(print_every);
      if (print_every == 0 || (print_every_ != 0 && nevents % std::abs(print_every_) != 0))
        print_every++;
    }

    long start(0), counter(0);
    auto start_time(std::chrono::steady_clock::now());
    while (counter < nevents) {
      counter += print_every;
      if (counter > nevents) counter = nevents;

      #pragma omp parallel for num_threads(num_threads()) default(shared) schedule(dynamic, omp_chunksize_)
      for (long i = start; i < counter; i++)
        compute(events[i], mults[i], weights[i], get_thread_num());

      // update counter
      start = counter;

      // print update
      auto diff(std::chrono::steady_clock::now() - start_time);
      double duration(std::chrono::duration_cast<std::chrono::duration<double>>(diff).count());
      unsigned nevents_width(std::to_string(nevents).size());
      oss_.str("  ");
      oss_ << std::setw(nevents_width) << counter << " / "
           << std::setw(nevents_width) << nevents << "  EECs computed  - "
           << std::setprecision(2) << std::setw(6) << double(counter)/nevents*100
           << "% completed - "
           << std::setprecision(3) << duration << 's';

      // acquire Python GIL if in SWIG in order to check for signals and print message
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      if (print_every_ != 0) *print_stream_ << oss_.str() << std::endl;
      if (PyErr_CheckSignals() != 0)
        throw std::runtime_error("KeyboardInterrupt received in EECBase::batch_compute");
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      if (print_every_ != 0) *print_stream_ << oss_.str() << std::endl;
    #endif
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
    if (use_charges()) {
      charges.resize(mult);
      for (unsigned i = 0; i < mult; i++)
        charges[i] = particles[i*nfeatures() + 3];
    }

    // check for degeneracy
    if (check_degen()) {
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

    // run actual computation from pts and charges
    else
      init_weights_and_compute(pts, charges, thread_i);
  }

  // fastjet support
#ifdef EEC_FASTJET_SUPPORT

  void set_pseudojet_charge_func(double (*pj_charge)(const fastjet::PseudoJet &)) {
    pj_charge_ = pj_charge;
    if (pj_charge_ != nullptr && nfeatures_ != 4)
      throw std::runtime_error("nfeatures should be 4 if using charges");
    if (pj_charge_ == nullptr && nfeatures_ != 3)
      throw std::runtime_error("nfeatures should be 3 if not using charges");
  }

  void compute(const std::vector<fastjet::PseudoJet> & pjs, const double weight = 1.0) {

    event_weights_[0] = weight;
    mults_[0] = pjs.size();

    // compute pairwise distances and extract pts
    std::vector<double> & dists(dists_[0]);
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
    if (use_charges()) {
      if (pj_charge_ == nullptr)
        throw std::runtime_error("No function provided to get charges from PseudoJets");
      
      charges.resize(pjs.size());
      for (unsigned i = 0; i < mults_[0]; i++)
        charges[i] = pj_charge_(pjs[i]);  
    }

    // run actual computation from pts and charges
    else init_weights_and_compute(pts, charges, 0);
  }

#endif // EEC_FASTJET_SUPPORT

  template <class EEC>
  void save(std::ostream & os) {
    #ifdef EEC_SERIALIZATION
      #ifdef EEC_COMPRESSION

        // we want to use compression
        if (get_compression_mode() == CompressionMode::Zlib) {
          boost::iostreams::filtering_ostream fos;
          fos.push(boost::iostreams::zlib_compressor(boost::iostreams::zlib::best_compression));
          fos.push(os);
          if (get_archive_format() == ArchiveFormat::Binary) {
            boost::archive::binary_oarchive ar(fos);
            ar << dynamic_cast<EEC &>(*this);
          }
          else {
            boost::archive::text_oarchive ar(fos);
            ar << dynamic_cast<EEC &>(*this);
          }
          return;
        }

      #endif // EEC_COMPRESSION

      // no compression, binary
      if (get_archive_format() == ArchiveFormat::Binary) {
        boost::archive::binary_oarchive ar(os);
        ar << dynamic_cast<EEC &>(*this);
      }

      // no compression, text
      else {
        boost::archive::text_oarchive ar(os);
        ar << dynamic_cast<EEC &>(*this);
      }

    #endif // EEC_SERIALIZATION
  }

  template <class EEC>
  void load(std::istream & is) {
    #ifdef EEC_SERIALIZATION
      #ifdef EEC_COMPRESSION

        // we want to use compression
        if (get_compression_mode() == CompressionMode::Zlib) {
          boost::iostreams::filtering_istream fis;
          fis.push(boost::iostreams::zlib_decompressor());
          fis.push(is);
          if (get_archive_format() == ArchiveFormat::Binary) {
            boost::archive::binary_iarchive ar(fis);
            ar >> dynamic_cast<EEC &>(*this);
          }
          else {
            boost::archive::text_iarchive ar(fis);
            ar >> dynamic_cast<EEC &>(*this);
          }
          return;
        }

      #endif // EEC_COMPRESSION

      // no compression, binary
      if (get_archive_format() == ArchiveFormat::Binary) {
        boost::archive::binary_iarchive ar(is);
        ar >> dynamic_cast<EEC &>(*this);
      }

      // no compression, text
      else {
        boost::archive::text_iarchive ar(is);
        ar >> dynamic_cast<EEC &>(*this);
      }

    #endif // EEC_SERIALIZATION
  }

protected:

  // method that carries out the specific EEC computation
  virtual void compute_eec(int thread_i) = 0;

  // allow derived class to access to vector storage
  const std::vector<std::vector<double>> & weights(int thread_i) const { return weights_[thread_i]; }
  const std::vector<double> & dists(int thread_i) const { return dists_[thread_i]; }
  double event_weight(int thread_i) const { return event_weights_[thread_i]; }
  unsigned mult(int thread_i) const { return mults_[thread_i]; }

private:

  // initializes EECBase
  void init() {
    set_print_stream(std::cout);

    oss_ = std::ostringstream(std::ios_base::ate);
    oss_.setf(std::ios_base::fixed, std::ios_base::floatfield);

    weights_.resize(num_threads(), std::vector<std::vector<double>>(N()));
    dists_.resize(num_threads());
    event_weights_.resize(num_threads());
    mults_.resize(num_threads());
  }

  // get weights as powers of pts and charges
  void init_weights_and_compute(std::vector<double> & pts, const std::vector<double> & charges, int thread_i) {

    // tally weights
    #pragma omp atomic
    total_weight_ += event_weights_[thread_i];

    // tally pts
    double pttot(0);
    for (double pt : pts)
      pttot += pt;

    // normalize pts, EEC total will be event_weight
    double eectot(event_weights_[thread_i]);
    if (norm())
      for (double & pt : pts)
        pt /= pttot;

    // EEC total will be event_Weight * pttot^N
    else eectot *= std::pow(pttot, N());

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
      if (use_charges()) {
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

    // delegate EEC computation to subclass
    compute_eec(thread_i);
  }

#ifdef EEC_FASTJET_SUPPORT
  // pointer to function to get charge from a PseudoJet
  double (*pj_charge_)(const fastjet::PseudoJet &) = nullptr;
#endif

#ifdef BOOST_SERIALIZATION_ACCESS_HPP
  friend class boost::serialization::access;
#endif

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & orig_pt_powers_ & pt_powers_ & orig_ch_powers_ & ch_powers_
       & N_ & nsym_ & nfeatures_
       & norm_ & use_charges_ & check_degen_ & average_verts_
       & num_threads_ & print_every_ & omp_chunksize_;

    if (version >= 1)
      ar & total_weight_;

    init();
  }

}; // EECBase

} // namespace eec

#if !defined(SWIG_PREPROCESSOR) && defined(EEC_SERIALIZATION)
BOOST_SERIALIZATION_ASSUME_ABSTRACT(eex::EECBase)
BOOST_CLASS_VERSION(eec::EECBase, 1)
#endif

#endif // EEC_BASE_HH
