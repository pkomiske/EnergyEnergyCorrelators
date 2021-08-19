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
#include <vector>

#include "EECConfig.hh"
#include "EECEvent.hh"

BEGIN_EEC_NAMESPACE

//-----------------------------------------------------------------------------
// Base class for all EEC computations
//-----------------------------------------------------------------------------

class EECBase {
public:

  //////////////
  // CONSTRUCTOR
  //////////////

  EECBase(const EECConfig & config) : 
    config_(config),
    nsym_(config.N),
    total_weight_(0)
  {
    // this runs through error checking and sets orig_weight_powers and orig_charge_powers
    set_weight_powers(weight_powers(), true);
    set_charge_powers(charge_powers(), true);

    // performs some error checking
    set_num_threads(num_threads());

    // ensure base starts in an initialized state
    init_base();
  }

  virtual ~EECBase() = default;

  //////////////////////////////////////
  // METHODS FOR IMMEDIATE COMPUTATION
  //////////////////////////////////////

  // compute immediately on a vector of PseudoJets and (optionally) charges
  void compute(const std::vector<PseudoJet> & pjs,
               const std::vector<double> & charges = {},
               double event_weight = 1,
               int thread = 0) {
    compute_eec_internal(EECEvent(config(), event_weight, pjs, charges),
                         thread);
  }

#ifndef SWIG

  // compute immediately on provided raw weights, dists, and (optionally) charges (as vectors)
  void compute(const std::vector<double> & raw_weights,
               const std::vector<double> & dists,
               const std::vector<double> & charges = {},
               double event_weight = 1,
               int thread = 0) {
    compute_eec_internal(EECEvent(config(), event_weight, raw_weights, dists, charges),
                         thread);
  }

// SWIG is defined
#else

  // compute immediately on provided raw weights, dists, and (optionally) charges (as pointers)
  void compute(const double * raw_weights, unsigned weights_mult,
               const double * dists, unsigned d0, unsigned d1,
               const double * charges, unsigned charges_mult,
               double event_weight = 1,
               int thread = 0) {
    compute_eec_internal(EECEvent(config(), event_weight,
                                  raw_weights, weights_mult,
                                  dists, d0, d1,
                                  charges, charges_mult),
                         thread);
  }

  // compute immediately on provided event as a 2d array with columns [weights, xcoord, ycoord, [charge]]
  // charge is optional, and nfeatures should be how many columns to expect in this array
  // note that ycoord is assumed to be periodic with period 2pi
  void compute(const double * event_ptr, unsigned mult, unsigned nfeatures,
               double event_weight = 1,
               int thread = 0) {
    compute_eec_internal(EECEvent(config(), event_weight, event_ptr, mult, nfeatures),
                         thread);
  }

#endif // SWIG

  //////////////////////////////////////////
  // METHODS FOR MULTITHREADED COMPUTATION
  //////////////////////////////////////////

  // add event from vector of PseudoJets and (optionally) charges
  void push_back(const std::vector<PseudoJet> & pjs,
                 const std::vector<double> & charges = {},
                 double event_weight = 1) {
    events_.emplace_back(config(), event_weight, pjs, charges);
  }

#ifndef SWIG

  // add event from raw weights, dists, and (optionally) charges (as vectors)
  void push_back(const std::vector<double> & raw_weights,
                 const std::vector<double> & dists,
                 const std::vector<double> & charges = {},
                 double event_weight = 1) {
    events_.emplace_back(config(), event_weight, raw_weights, dists, charges);
  }

// SWIG is defined
#else

  // add event from raw weights, dists, and (optionally) charges (as pointers)
  void push_back(const double * raw_weights, unsigned weights_mult,
                 const double * dists, unsigned d0, unsigned d1,
                 const double * charges, unsigned charges_mult,
                 double event_weight = 1) {
    events_.emplace_back(config(), event_weight,
                         raw_weights, weights_mult,
                         dists, d0, d1,
                         charges, charges_mult);
  }

  // add event from 2d array with columns [weights, xcoord, ycoord, [charge]]
  // charge is optional, and nfeatures should be how many columns to expect in this array
  // note that ycoord is assumed to be periodic with period 2pi
  void push_back(const double * event_ptr, unsigned mult, unsigned nfeatures,
                 double event_weight = 1) {
    events_.emplace_back(config(), event_weight, event_ptr, mult, nfeatures);
  }

#endif // SWIG

  // clears internal events (and frees memory)
  void clear_events() {
    events_.clear();
    std::vector<EECEvent>().swap(events_);
  }

  // run multi-threaded computation on internally stored events
  void batch_compute() {

    // process number of events
    long nevents(events_.size()), print_every(this->print_every() == 0 ? -1 : this->print_every());
    if (print_every < 0) {
      print_every = nevents/std::abs(print_every);
      if (print_every == 0 || (this->print_every() != 0 && nevents % std::abs(this->print_every()) != 0))
        print_every++;
    }

    // variables for multithreaded loop over events
    long start(0), counter(0);
    auto start_time(std::chrono::steady_clock::now());

    // make function for printing an update
    auto print_function = [&counter, start_time, nevents]() {

      // setup output stream
      std::ostringstream oss(std::ios_base::ate);
      oss.setf(std::ios_base::fixed, std::ios_base::floatfield);

      auto diff(std::chrono::steady_clock::now() - start_time);
      double duration(std::chrono::duration_cast<std::chrono::duration<double>>(diff).count());
      unsigned nevents_width(std::to_string(nevents).size());
      oss.str("  ");
      oss << std::setw(nevents_width) << counter << " / "
          << std::setw(nevents_width) << nevents << "  EECs computed  - "
          << std::setprecision(2) << std::setw(6) << double(counter)/nevents*100
          << "% completed - "
          << std::setprecision(3) << duration << 's';

      // add new line to final printing
      if (counter == nevents) oss << '\n';

      // acquire Python GIL if in SWIG in order to check for signals and print message
      #ifdef SWIG
        SWIG_PYTHON_THREAD_BEGIN_BLOCK;
        std::cout << oss.str() << std::endl;
        if (PyErr_CheckSignals() != 0)
          throw std::runtime_error("KeyboardInterrupt received in EECBase::batch_compute");
        SWIG_PYTHON_THREAD_END_BLOCK;
      #else
        std::cout << oss.str() << std::endl;
      #endif
    };

    while (counter < nevents) {
      counter += print_every;
      if (counter > nevents) counter = nevents;

      #pragma omp parallel for num_threads(num_threads()) default(shared) schedule(dynamic, omp_chunksize())
      for (long i = start; i < counter; i++) {
        #pragma omp atomic
        total_weight_ += events_[i].event_weight();

        compute_eec_internal(events_[i], omp_get_thread_num());
      }

      // update counter
      start = counter;

      // print update
      if (this->print_every() != 0)
        print_function();
    }
  }

  ///////////////////
  // GETTER FUNCTIONS
  ///////////////////

  const EECConfig & config() const { return config_; }
  const std::string & compname() const { return compname_; }

  unsigned N() const { return config().N; }
  unsigned nfeatures() const { return config().nfeatures; }
  unsigned nsym() const { return nsym_; }

  const std::vector<double> & weight_powers() const { return config().weight_powers; }
  const std::vector<unsigned> & charge_powers() const { return config().charge_powers; }

  bool norm() const { return config().norm; }
  bool use_charges() const { return config().use_charges; }
  bool check_degen() const { return config().check_degen; }
  bool average_verts() const { return config().average_verts; }

  int num_threads() const { return config().num_threads; }
  int omp_chunksize() const { return config().omp_chunksize; }
  long print_every() const { return config().print_every; }

  double total_weight() const { return total_weight_; }

  ParticleWeight particle_weight() const { return config().particle_weight; }
  PairwiseDistance pairwise_distance() const { return config().pairwise_distance; }

  ///////////////////
  // SETTER FUNCTIONS
  ///////////////////

  void set_weight_powers(const std::vector<double> & wps, bool _in_constructor = false) {
    ensure_no_events();

    if (wps.size() == 1)
      orig_weight_powers_ = std::vector<double>(N(), wps[0]);
    else if (wps.size() != N())
      throw std::invalid_argument("weight_powers must be a vector of size 1 or " + std::to_string(N()));
    else orig_weight_powers_ = wps;

    // need to reinitialize after changes to weight_powers
    if (!_in_constructor) init_all();
  }

  void set_charge_powers(const std::vector<unsigned> & cps, bool _in_constructor = false) {
    ensure_no_events();

    if (cps.size() == 1)
      orig_charge_powers_ = std::vector<unsigned>(N(), cps[0]);
    else if (cps.size() != N())
      throw std::invalid_argument("weight_powers must be a vector of size 1 or " + std::to_string(N()));
    else orig_charge_powers_ = cps;

    // need to reinitialize after changes to charge_powers
    if (!_in_constructor) init_all();
  }

  void set_norm(bool n) { ensure_no_events(); config_.norm = n; }
  void set_check_degen(bool check) { ensure_no_events(); config_.check_degen = check; }
  void set_average_verts(bool averts) { ensure_no_events(); config_.average_verts = averts; init_subclass(); }
  
  void set_num_threads(int nthreads) { config_.num_threads = determine_num_threads(nthreads); }
  void set_omp_chunksize(int chunksize) { config_.omp_chunksize = std::abs(chunksize); }
  void set_print_every(long print_every) { config_.print_every = print_every; }

  void set_particle_weight(ParticleWeight pw) { ensure_no_events(); config_.particle_weight = pw; }
  void set_pairwise_distance(PairwiseDistance pd) { ensure_no_events(); config_.pairwise_distance = pd; }

  //////////////////////////////////
  // Allow EECs to be added together
  //////////////////////////////////

  bool operator!=(const EECBase & rhs) const { return !operator==(rhs); }
  bool operator==(const EECBase & rhs) const {
    if (N()                        != rhs.N()                        ||
        nfeatures()                != rhs.nfeatures()                ||
        nsym()                     != rhs.nsym()                     ||
        norm()                     != rhs.norm()                     ||
        use_charges()              != rhs.use_charges()              ||
        check_degen()              != rhs.check_degen()              ||
        average_verts()            != rhs.average_verts()            ||
        num_threads()              != rhs.num_threads()              ||
        omp_chunksize()            != rhs.omp_chunksize()            ||
        print_every()              != rhs.print_every()              ||
        particle_weight()          != rhs.particle_weight()          ||
        pairwise_distance()        != rhs.pairwise_distance()        ||
        weight_powers().size()     != rhs.weight_powers().size()     ||
        charge_powers().size()     != rhs.charge_powers().size()     ||
        orig_weight_powers_.size() != rhs.orig_weight_powers_.size() ||
        orig_charge_powers_.size() != rhs.orig_charge_powers_.size())
      return false;

    for (unsigned i = 0; i < weight_powers().size(); i++) {
      if (weight_powers()[i] != rhs.weight_powers()[i] || 
          charge_powers()[i] != rhs.charge_powers()[i])
        return false;
    }

    for (unsigned i = 0; i < orig_weight_powers_.size(); i++)
      if (orig_weight_powers_[i] != rhs.orig_weight_powers_[i])
        return false;

    for (unsigned i = 0; i < orig_charge_powers_.size(); i++)
      if (orig_charge_powers_[i] != rhs.orig_charge_powers_[i])
        return false;

    return true;
  }

  EECBase & operator+=(const EECBase & rhs) {

    if ((*this) != rhs)
      throw std::invalid_argument("EEC computations must match in order to be added together");

    total_weight_ += rhs.total_weight();

    return *this;
  }

protected:

  // overloaded by subclass to carry out specific EEC computation
  virtual void compute_eec_internal(const EECEvent & event, int thread) = 0;

  // subclass initialization
  virtual void init_subclass(bool = false) = 0;

  // ensures EECBase and the subclass are initialized consistently
  void init_all() {
    init_base();
    init_subclass();
  }

  // check that we haven't actually computed any events yet
  void ensure_no_events() const {
    if (total_weight() != 0 || !events_.empty())
      throw std::runtime_error("cannot alter settings after processing events");
  }

  // description of the EEC computation
  std::string description() const {

    std::ostringstream oss;
    oss << std::boolalpha
        << compname() << '\n'
        << "  N - " << N() << '\n'
        << "  norm - " << norm() << '\n'
        << "  use_charges - " << use_charges() << '\n'
        << "\n"
        << "  nfeatures (arrays only) - " << nfeatures() << '\n'
        << "  particle_weight (PseudoJets only) - "
              << particle_weight_name(particle_weight()) << '\n'
        << "  pairwise_distance (PseudoJets only) - "
              << pairwise_distance_name(pairwise_distance()) << '\n'
        << "\n"
        << "  check_degen - " << check_degen() << '\n'
        << "  average_verts - " << average_verts() << '\n'
        << "\n"
        << "  print_every - " << print_every() << '\n'
        << "  num_threads - " << num_threads() << '\n';

    // record pt and charge powers
    oss << "  weight_powers - (" << orig_weight_powers_[0];
    for (unsigned i = 1; i < orig_weight_powers_.size(); i++)
      oss << ", " << orig_weight_powers_[i];
    oss << ")\n"
        << "  ch_powers - (" << orig_charge_powers_[0];
    for (unsigned i = 1; i < orig_charge_powers_.size(); i++)
      oss << ", " << orig_charge_powers_[i];
    oss << ")\n";

    return oss.str();
  }

  ///////////////////////////////
  // SERIALIZATION METHODS
  ///////////////////////////////

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

private:

  ///////////////
  // DATA MEMBERS
  ///////////////

  // the EEC configuration
  EECConfig config_;

  // the original pt and charge powers
  std::vector<double> orig_weight_powers_;//, weight_powers();
  std::vector<unsigned> orig_charge_powers_;//, charge_powers();

  // details of the EEC computation
  unsigned nsym_;
  std::string compname_;
  double total_weight_;

  // vector of events
  std::vector<EECEvent> events_;

  //////////////////
  // INITIALIZATIONS
  //////////////////

  // (re)initializes EECBase
  // sets weight_powers, charge_powers, compname, nsym, nfeatures, use_charges
  void init_base() {

    ensure_no_events();

    // begin figuring out weight_powers and charge_powers to be used in config
    config_.weight_powers = orig_weight_powers_;
    config_.charge_powers = orig_charge_powers_;

    // check for symmetries in the different cases
    switch (N()) {
      case 0:
      case 1:
        throw std::invalid_argument("N must be 2 or greater");
        break;

      case 2: {
        if (weight_powers()[0] == weight_powers()[1] && charge_powers()[0] == charge_powers()[1]) {
          config_.weight_powers = {weight_powers()[0]};
          config_.charge_powers = {charge_powers()[0]};
          compname_ = "eec_ij_sym";
        }
        else {
          compname_ = "eec_no_sym";
          nsym_ = 0;
        }
        break;
      }

      case 3: {
        bool match01(weight_powers()[0] == weight_powers()[1] && charge_powers()[0] == charge_powers()[1]),
             match12(weight_powers()[1] == weight_powers()[2] && charge_powers()[1] == charge_powers()[2]),
             match02(weight_powers()[0] == weight_powers()[2] && charge_powers()[0] == charge_powers()[2]);
        if (match01 && match12) {
          config_.weight_powers = {weight_powers()[0]};
          config_.charge_powers = {charge_powers()[0]};
          compname_ = "eeec_ijk_sym";
        }
        else if (match01) {
          config_.weight_powers = {weight_powers()[0], weight_powers()[2]};
          config_.charge_powers = {charge_powers()[0], charge_powers()[2]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else if (match12) {
          config_.weight_powers = {weight_powers()[1], weight_powers()[0]};
          config_.charge_powers = {charge_powers()[1], charge_powers()[0]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else if (match02) {
          config_.weight_powers = {weight_powers()[2], weight_powers()[1]};
          config_.charge_powers = {charge_powers()[2], charge_powers()[1]};
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
          if (weight_powers()[i] != weight_powers()[0] || charge_powers()[i] != charge_powers()[0])
            throw std::invalid_argument("N = 4 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeec_ijkl_sym";
        break;
      }

      case 5: {
        for (int i = 1; i < 5; i++) {
          if (weight_powers()[i] != weight_powers()[0] || charge_powers()[i] != charge_powers()[0])
            throw std::invalid_argument("N = 5 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeeec_ijklm_sym";
        break;
      }

      case 6: {
        for (int i = 1; i < 6; i++) {
          if (weight_powers()[i] != weight_powers()[0] || charge_powers()[i] != charge_powers()[0])
            throw std::invalid_argument("N = 6 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeeeec_ijklmn_sym";
        break;
      }

      case 7: {
        for (int i = 1; i < 7; i++) {
          if (weight_powers()[i] != weight_powers()[0] || charge_powers()[i] != charge_powers()[0])
            throw std::invalid_argument("N = 7 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeeeeec_ijklmno_sym";
        break;
      }

      case 8: {
        for (int i = 1; i < 8; i++) {
          if (weight_powers()[i] != weight_powers()[0] || charge_powers()[i] != charge_powers()[0])
            throw std::invalid_argument("N = 8 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeeeeeeec_ijklmnop_sym";
        break;
      }

      default:
        for (unsigned i = 1; i < N(); i++) {
          if (weight_powers()[i] != weight_powers()[0] || charge_powers()[i] != charge_powers()[0])
            throw std::invalid_argument("this N only supports the fully symmetric correlator currently");
        }
        compname_ = "eNc_sym";
    }

    // check for using charges at all
    config_.use_charges = false;
    for (int ch_power : charge_powers())
      if (ch_power != 0)
        config_.use_charges = true;

    // set nfeatures
    config_.nfeatures = use_charges() ? 4 : 3;
  }

  //////////////////////
  // BOOST SERIALIZATION
  //////////////////////

#ifdef BOOST_SERIALIZATION_ACCESS_HPP
  friend class boost::serialization::access;
#endif

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {

    if (version < 3)
      ar & orig_weight_powers_ & config_.weight_powers
         & orig_charge_powers_ & config_.charge_powers
         & config_.N & nsym_ & config_.nfeatures
         & config_.norm & config_.use_charges & config_.check_degen & config_.average_verts
         & config_.num_threads & config_.print_every & config_.omp_chunksize;
    else
      ar & config_ & orig_weight_powers_ & orig_charge_powers_ & nsym_;

    if (version > 0)
      ar & total_weight_;

    if (version > 1)
      ar & compname_;

    // reset num threads in case maximum number is different on new machine
    set_num_threads(num_threads());
  }

}; // EECBase

END_EEC_NAMESPACE

#if !defined(SWIG_PREPROCESSOR) && defined(EEC_SERIALIZATION)
  BOOST_SERIALIZATION_ASSUME_ABSTRACT(EEC_NAMESPACE::EECBase)
  BOOST_CLASS_VERSION(EEC_NAMESPACE::EECBase, 3)
#endif

#endif // EEC_BASE_HH
