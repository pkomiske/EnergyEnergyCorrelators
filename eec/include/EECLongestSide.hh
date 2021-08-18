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
 *   _      ____  _   _  _____ ______  _____ _______    _____ _____ _____  ______
 *  | |    / __ \| \ | |/ ____|  ____|/ ____|__   __|  / ____|_   _|  __ \|  ____|
 *  | |   | |  | |  \| | |  __| |__  | (___    | |    | (___   | | | |  | | |__
 *  | |   | |  | | . ` | | |_ |  __|  \___ \   | |     \___ \  | | | |  | |  __|
 *  | |___| |__| | |\  | |__| | |____ ____) |  | |     ____) |_| |_| |__| | |____
 *  |______\____/|_| \_|\_____|______|_____/   |_|    |_____/|_____|_____/|______|
 */

#ifndef EEC_LONGESTSIDE_HH
#define EEC_LONGESTSIDE_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "EECBase.hh"
#include "EECHist1D.hh"
#include "EECMultinomial.hh"

BEGIN_EEC_NAMESPACE

//-----------------------------------------------------------------------------
// EEC class for histogramming according to the maximum particle distance
//-----------------------------------------------------------------------------

template<class Transform>
class EECLongestSide : public EECBase, public hist::EECHist1D<Transform> {
public:

  typedef EECLongestSide<Transform> Self;
  typedef hist::EECHist1D<Transform> EECHist;
  typedef typename EECHist::SimpleWeightedHist SimpleWeightedHist;

#ifndef SWIG_PREPROCESSOR

  // default constructor
  EECLongestSide() : EECLongestSide(EECConfig(), 1) {}

  // constructor taking EECConfig and Hist1D arguments
  EECLongestSide(const EECConfig & config,
                 unsigned nbins,
                 const std::array<double, 2> & axis_range = {1e-5, 1},
                 bool track_covariance = true,
                 bool variance_bound = true,
                 bool variance_bound_includes_overflows = true,
                 bool use_general_eNc = false) :

    // construct EECBase from provided EECConfig
    EECBase(config),

    // construct EECHist1D
    EECHist({nbins}, {axis_range}, EECBase::num_threads(),
              track_covariance, variance_bound, variance_bound_includes_overflows),

    // variables local to this class
    use_general_eNc_(use_general_eNc),
    N_choose_2_(N()*(N() - 1)/2)
  {
    init_subclass();
  }

#endif // !SWIG_PREPROCESSOR

  // constructor with all options (mostly useful for Python)
  EECLongestSide(unsigned N,
                 unsigned nbins,
                 const std::array<double, 2> & axis_range = {1e-5, 1},
                 bool norm = true,
                 const std::vector<double> & weight_powers = {1},
                 const std::vector<unsigned> & charge_powers = {0},
                 ParticleWeight particle_weight = ParticleWeight::TransverseMomentum,
                 PairwiseDistance pairwise_distance = PairwiseDistance::DeltaR,
                 int num_threads = -1,
                 int omp_chunksize = 10,
                 long print_every = -10,
                 bool check_degen = false,
                 bool average_verts = false,
                 bool track_covariance = true,
                 bool variance_bound = true,
                 bool variance_bound_includes_overflows = true,
                 bool use_general_eNc = false) :
    EECLongestSide(EECConfig(N, norm, weight_powers, charge_powers,
                             particle_weight, pairwise_distance,
                             num_threads, omp_chunksize, print_every,
                             check_degen, average_verts),
                   nbins, axis_range,
                   track_covariance,
                   variance_bound,
                   variance_bound_includes_overflows,
                   use_general_eNc)
  {}

  virtual ~EECLongestSide() = default;

  // getter/setter options
  bool use_general_eNc() const { return use_general_eNc_; }
  void set_use_general_eNc(bool general) { use_general_eNc_ = general; init_subclass(); }
  void set_num_threads(int threads) {
    EECBase::set_num_threads(threads);
    EECHist::set_num_threads(threads);
  }

  std::string description(int hist_level = 1) const {
    unsigned nh(this->nhists());

    std::ostringstream oss;
    oss << std::boolalpha
        << "EECLongestSide<" << EECHist::axes_description()
        << ">::" << EECBase::description() << '\n'
        << "  using eNc_sym - " << use_general_eNc_ << '\n'
        << "\n"
        << "  " << EECHist::hist_name() << " -  there " << (nh == 1 ? "is " : "are ") << nh << " histogram";

    if (nh == 1) 
      oss << '\n';

    else if (nh == 2)
      oss << "s, labeled according to the location of the largest side\n"
          << "    0 - the largest side is the one with identical vertices\n"
          << "    1 - the largest side is the one with different vertices\n";

    else if (nh == 3)
      oss << "s, labeled according to the location of the largest side\n"
          << "    0 - the largest side is between vertices 0 and 1\n"
          << "    1 - the largest side is between vertices 1 and 2\n"
          << "    2 - the largest side is between vertices 0 and 2\n";

    else 
      throw std::runtime_error("Unexpected number of histograms encountered");

    this->hists_as_text(hist_level, true, 16, &oss);

    return oss.str();
  }

  void load(std::istream & is) {
    EECBase::load<Self>(is);
  }

  void save(std::ostream & os) {
    EECBase::save<Self>(os);
  }

  bool operator!=(const EECLongestSide & rhs) const { return !operator==(rhs); }
  bool operator==(const EECLongestSide & rhs) const {
    return EECBase::operator==(rhs)                       &&
           EECHist::operator==(rhs)                       &&
           use_general_eNc()     == rhs.use_general_eNc() &&
           N_choose_2_           == rhs.N_choose_2_       &&
           compute_eec_func_ptr_ == rhs.compute_eec_func_ptr_;
  }

  EECLongestSide & operator+=(const EECLongestSide & rhs) {
    EECBase::operator+=(rhs);
    EECHist::operator+=(rhs);

    return *this;
  }

private:

  // data members local to this class
  bool use_general_eNc_;
  unsigned N_choose_2_;

  // function pointer to the actual computation that will be run
  void (EECLongestSide::*compute_eec_func_ptr_)(const EECEvent &, int);

  // provides initialization of this subclass from configuration
  void init_subclass(bool events_allowed = false) {
    if (!events_allowed)
      ensure_no_events();

    // default is general computation
    compute_eec_func_ptr_ = &EECLongestSide::eNc_sym;

    // set pointer to function that will do the computation
    switch (N()) {
      case 2:
        if (nsym() == 2) {
          if (!use_general_eNc_)
            compute_eec_func_ptr_ = &EECLongestSide::eec_ij_sym;
        }
        else
          compute_eec_func_ptr_ = &EECLongestSide::eec_no_sym;
        break;

      case 3:
        switch (nsym()) {
          case 3:
            if (!use_general_eNc_)
              compute_eec_func_ptr_ = &EECLongestSide::eeec_ijk_sym;
            break;

          case 2:
            compute_eec_func_ptr_ = &EECLongestSide::eeec_ij_sym;
            if (!average_verts()) this->duplicate_histograms(2);
            break;

          case 0:
            compute_eec_func_ptr_ = &EECLongestSide::eeec_no_sym;
            if (!average_verts()) this->duplicate_histograms(3);
            break;

          default:
            throw std::runtime_error("Invalid number of symmetries " + std::to_string(nsym()));
        }
        break;

      case 4:
        assert(nsym() == 4);
        if (!use_general_eNc_)
          compute_eec_func_ptr_ = &EECLongestSide::eeeec_ijkl_sym;
        break;

      case 5:
        assert(nsym() == 5);
        if (!use_general_eNc_)
          compute_eec_func_ptr_ = &EECLongestSide::eeeeec_ijklm_sym;
        break;

      case 6:
        assert(nsym() == 6);
        if (!use_general_eNc_)
          compute_eec_func_ptr_ = &EECLongestSide::eeeeeec_ijklmn_sym;
        break;

      case 7:
        assert(nsym() == 7);
        if (!use_general_eNc_)
          compute_eec_func_ptr_ = &EECLongestSide::eeeeeeec_ijklmno_sym;
        break;

      case 8:
        assert(nsym() == 8);
        if (!use_general_eNc_)
          compute_eec_func_ptr_ = &EECLongestSide::eeeeeeeec_ijklmnop_sym;
        break;

      default:
        if (N() >= FACTORIALS_LONG.size()) {
          std::ostringstream m;
          m << "N must be less than " << FACTORIALS_LONG.size()
            << " due to the use of " << sizeof(std::size_t) << "-bit integers";
          throw std::invalid_argument(m.str());
        }
    }
  }
  
  // implements pure virtual method from base class
  void compute_eec_internal(const EECEvent & event, int thread) {
    (this->*compute_eec_func_ptr_)(event, thread);
    this->fill_from_single_event(thread);
  }

  void eec_ij_sym(const EECEvent & event, int thread) {

    std::cout << "eec_ij_sym start" << std::endl;
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over symmetric pairs of particles
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      unsigned ixm(i*mult);

      // i == j term
      hist(hist::weight(weight_i * ws0[i]), 0);

      // off diagonal terms
      weight_i *= 2;
      for (unsigned j = 0; j < i; j++)
        hist(hist::weight(weight_i * ws0[j]), dists[ixm + j]);
    }

    std::cout << "eec_ij_sym done" << std::endl;
  }

  void eec_no_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & ws1(event.weights()[1]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over all pairs of particles
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult);

      for (unsigned j = 0; j < mult; j++)
        hist(hist::weight(weight_i * ws1[j]), dists[ixm + j]);
    }
  }

  void eeec_ijk_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over triplets of particles
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      unsigned ixm(i*mult);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        double dist_ij(dists[ixm + j]);
        unsigned jxm(j*mult);
        bool ij_match(i == j);

        for (unsigned k = 0; k <= j; k++) {
          double dist_ik(dists[ixm + k]), dist_jk(dists[jxm + k]);
          bool ik_match(i == k), jk_match(j == k);
          int sym(!(ij_match || ik_match || jk_match) ? 6 : (ij_match && ik_match ? 1 : 3));

          // determine maximum distance
          double max_dist(dist_ij);
          if (dist_jk > dist_ij) {
            if (dist_jk > dist_ik) max_dist = dist_jk;
            else max_dist = dist_ik;
          }
          else if (dist_ik > dist_ij) max_dist = dist_ik;

          // fill histogram
          hist(hist::weight(weight_ij * ws0[k] * sym), max_dist);
        }
      }
    }
  }

  void eeec_ij_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & ws1(event.weights()[1]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    std::vector<SimpleWeightedHist> & hists(this->per_event_hists(thread));

    // loop over triplets of particles
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j] * (i == j ? 1 : 2));
        if (weight_ij == 0) continue;
        double dist_ij(dists[ixm + j]);
        unsigned jxm(j*mult);

        for (unsigned k = 0; k < mult; k++) {
          double weight_ijk(weight_ij * ws1[k]), dist_ik(dists[ixm + k]), dist_jk(dists[jxm + k]);

          // determine maximum distance and if side ij is the largest
          double max_dist(dist_ij);
          unsigned hist_i(0);
          if (dist_jk > dist_ij) {
            hist_i = 1;
            if (dist_jk > dist_ik) max_dist = dist_jk;
            else max_dist = dist_ik;
          }
          else if (dist_ik > dist_ij) {
            hist_i = 1;
            max_dist = dist_ik;
          }

          // handle case of averaging over verts
          if (average_verts())
            hists[0](hist::weight(weight_ijk), max_dist);

          // no averaging here, fill the targeted hist
          else {
            hists[hist_i](hist::weight(weight_ijk), max_dist);

            // fill other histogram if max_dist is tied at zero
            if (max_dist == 0)
              hists[hist_i == 0 ? 1 : 0](hist::weight(weight_ijk), max_dist);  
          }
        }
      }
    }
  }

  void eeec_no_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & ws1(event.weights()[1]),
                              & ws2(event.weights()[2]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    std::vector<SimpleWeightedHist> & hists(this->per_event_hists(thread));

    // loop over unique triplets of particles
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult);

      for (unsigned j = 0; j < mult; j++) {
        double weight_ij(weight_i * ws1[j]);
        if (weight_ij == 0) continue;
        unsigned jxm(j*mult);
        double dist_ij(dists[ixm + j]);
        bool ij_match(i == j);

        for (unsigned k = 0; k < mult; k++) {
          double weight_ijk(weight_ij * ws2[k]), dist_ik(dists[ixm + k]), dist_jk(dists[jxm + k]);
          bool ik_match(i == k), jk_match(j == k);

          // determine maximum distance
          double max_dist(dist_ij);
          unsigned hist_i(0);
          if (dist_jk > dist_ij) {
            if (dist_jk > dist_ik) {
              max_dist = dist_jk;
              hist_i = 1;
            }
            else {
              max_dist = dist_ik;
              hist_i = 2;
            }
          }
          else if (dist_ik > dist_ij) {
            max_dist = dist_ik;
            hist_i = 2;
          }

          // always use hist_i = 0 if averaging over verts
          if (average_verts())
            hists[0](hist::weight(weight_ijk), max_dist);

          // no degeneracy at all
          else if (!(ij_match || ik_match || jk_match))
            hists[hist_i](hist::weight(weight_ijk), max_dist);

          // everything is degenerate
          else if (ij_match && ik_match) {
            hists[0](hist::weight(weight_ijk), max_dist);
            hists[1](hist::weight(weight_ijk), max_dist);
            hists[2](hist::weight(weight_ijk), max_dist);
          }

          // ij overlap, largest sides are ik and jk
          else if (ij_match) {
            hists[1](hist::weight(weight_ijk), max_dist);
            hists[2](hist::weight(weight_ijk), max_dist);
          }

          // jk overlap, largest sides are ij, ik
          else if (jk_match) {
            hists[0](hist::weight(weight_ijk), max_dist);
            hists[2](hist::weight(weight_ijk), max_dist);
          }

          // ik overlap, largest sides are ij, jk
          else if (ik_match) {
            hists[0](hist::weight(weight_ijk), max_dist);
            hists[1](hist::weight(weight_ijk), max_dist);
          }

          // should never get here
          else throw std::runtime_error("should never get here in EECLongestSide::eeec_no_sym");
        }
      }
    }
  }

  void eeeec_ijkl_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over quadruplets of particles
    std::array<double, 6> dists_arr;
    Multinomial<4> multinom;
    for (unsigned i = 0; i < mult; i++) {
      unsigned ixm(i*mult);
      double weight_i(event_weight * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists_arr[0] = dists[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult);
          double weight_ijk(weight_ij * ws0[k]);
          dists_arr[1] = dists[kxm + i];
          dists_arr[2] = dists[kxm + j];
          dists_arr[2] = *std::max_element(dists_arr.cbegin(), dists_arr.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists_arr[3] = dists[lxm + i];
            dists_arr[4] = dists[lxm + j];
            dists_arr[5] = dists[lxm + k];
            multinom.set_index_final(l);

            // fill histogram
            hist(hist::weight(multinom.value() * weight_ijkl),
                        *std::max_element(dists_arr.cbegin() + 2, dists_arr.cend()));
          }
        }
      }
    }
  }

  void eeeeec_ijklm_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over quintuplets of particles
    std::array<double, 10> dists_arr;
    Multinomial<5> multinom;
    for (unsigned i = 0; i < mult; i++) {
      unsigned ixm(i*mult);
      double weight_i(event_weight * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists_arr[0] = dists[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult);
          double weight_ijk(weight_ij * ws0[k]);
          dists_arr[1] = dists[kxm + i];
          dists_arr[2] = dists[kxm + j];
          dists_arr[2] = *std::max_element(dists_arr.cbegin(), dists_arr.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists_arr[3] = dists[lxm + i];
            dists_arr[4] = dists[lxm + j];
            dists_arr[5] = dists[lxm + k];
            dists_arr[5] = *std::max_element(dists_arr.cbegin() + 2, dists_arr.cbegin() + 6);
            multinom.set_index<3>(l);

            for (unsigned m = 0; m <= l; m++) {
              unsigned mxm(m*mult);
              double weight_ijklm(weight_ijkl * ws0[m]);
              dists_arr[6] = dists[mxm + i];
              dists_arr[7] = dists[mxm + j];
              dists_arr[8] = dists[mxm + k];
              dists_arr[9] = dists[mxm + l];
              multinom.set_index_final(m);

              // fill histogram
              hist(hist::weight(multinom.value() * weight_ijklm),
                          *std::max_element(dists_arr.cbegin() + 5, dists_arr.cend()));
            }
          }
        }
      }
    }
  }

  void eeeeeec_ijklmn_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over quintuplets of particles
    std::array<double, 15> dists_arr;
    Multinomial<6> multinom;
    for (unsigned i = 0; i < mult; i++) {
      unsigned ixm(i*mult);
      double weight_i(event_weight * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists_arr[0] = dists[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult);
          double weight_ijk(weight_ij * ws0[k]);
          dists_arr[1] = dists[kxm + i];
          dists_arr[2] = dists[kxm + j];
          dists_arr[2] = *std::max_element(dists_arr.cbegin(), dists_arr.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists_arr[3] = dists[lxm + i];
            dists_arr[4] = dists[lxm + j];
            dists_arr[5] = dists[lxm + k];
            dists_arr[5] = *std::max_element(dists_arr.cbegin() + 2, dists_arr.cbegin() + 6);
            multinom.set_index<3>(l);

            for (unsigned m = 0; m <= l; m++) {
              unsigned mxm(m*mult);
              double weight_ijklm(weight_ijkl * ws0[m]);
              dists_arr[6] = dists[mxm + i];
              dists_arr[7] = dists[mxm + j];
              dists_arr[8] = dists[mxm + k];
              dists_arr[9] = dists[mxm + l];
              dists_arr[9] = *std::max_element(dists_arr.cbegin() + 5, dists_arr.cbegin() + 10);
              multinom.set_index<4>(m);

              for (unsigned n = 0; n <= m; n++) {
                unsigned nxm(n*mult);
                double weight_ijklmn(weight_ijklm * ws0[n]);
                dists_arr[10] = dists[nxm + i];
                dists_arr[11] = dists[nxm + j];
                dists_arr[12] = dists[nxm + k];
                dists_arr[13] = dists[nxm + l];
                dists_arr[14] = dists[nxm + m];
                multinom.set_index_final(n);

                // fill histogram
                hist(hist::weight(multinom.value() * weight_ijklmn),
                            *std::max_element(dists_arr.cbegin() + 9, dists_arr.cend()));
              }
            }
          }
        }
      }
    }
  }

  void eeeeeeec_ijklmno_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over quintuplets of particles
    std::array<double, 21> dists_arr;
    Multinomial<7> multinom;
    for (unsigned i = 0; i < mult; i++) {
      unsigned ixm(i*mult);
      double weight_i(event_weight * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists_arr[0] = dists[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult);
          double weight_ijk(weight_ij * ws0[k]);
          dists_arr[1] = dists[kxm + i];
          dists_arr[2] = dists[kxm + j];
          dists_arr[2] = *std::max_element(dists_arr.cbegin(), dists_arr.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists_arr[3] = dists[lxm + i];
            dists_arr[4] = dists[lxm + j];
            dists_arr[5] = dists[lxm + k];
            dists_arr[5] = *std::max_element(dists_arr.cbegin() + 2, dists_arr.cbegin() + 6);
            multinom.set_index<3>(l);

            for (unsigned m = 0; m <= l; m++) {
              unsigned mxm(m*mult);
              double weight_ijklm(weight_ijkl * ws0[m]);
              dists_arr[6] = dists[mxm + i];
              dists_arr[7] = dists[mxm + j];
              dists_arr[8] = dists[mxm + k];
              dists_arr[9] = dists[mxm + l];
              dists_arr[9] = *std::max_element(dists_arr.cbegin() + 5, dists_arr.cbegin() + 10);
              multinom.set_index<4>(m);

              for (unsigned n = 0; n <= m; n++) {
                unsigned nxm(n*mult);
                double weight_ijklmn(weight_ijklm * ws0[n]);
                dists_arr[10] = dists[nxm + i];
                dists_arr[11] = dists[nxm + j];
                dists_arr[12] = dists[nxm + k];
                dists_arr[13] = dists[nxm + l];
                dists_arr[14] = dists[nxm + m];
                dists_arr[14] = *std::max_element(dists_arr.cbegin() + 9, dists_arr.cbegin() + 15);
                multinom.set_index<5>(n);

                for (unsigned o = 0; o <= n; o++) {
                  unsigned oxm(o*mult);
                  double weight_ijklmno(weight_ijklmn * ws0[o]);
                  dists_arr[15] = dists[oxm + i];
                  dists_arr[16] = dists[oxm + j];
                  dists_arr[17] = dists[oxm + k];
                  dists_arr[18] = dists[oxm + l];
                  dists_arr[19] = dists[oxm + m];
                  dists_arr[20] = dists[oxm + n];
                  multinom.set_index_final(o);

                  // fill histogram
                  hist(hist::weight(multinom.value() * weight_ijklmno),
                       *std::max_element(dists_arr.cbegin() + 14, dists_arr.cend()));
                }
              }
            }
          }
        }
      }
    }
  }

  void eeeeeeeec_ijklmnop_sym(const EECEvent & event, int thread) {
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // loop over quintuplets of particles
    std::array<double, 28> dists_arr;
    Multinomial<8> multinom;
    for (unsigned i = 0; i < mult; i++) {
      unsigned ixm(i*mult);
      double weight_i(event_weight * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists_arr[0] = dists[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult);
          double weight_ijk(weight_ij * ws0[k]);
          dists_arr[1] = dists[kxm + i];
          dists_arr[2] = dists[kxm + j];
          dists_arr[2] = *std::max_element(dists_arr.cbegin(), dists_arr.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists_arr[3] = dists[lxm + i];
            dists_arr[4] = dists[lxm + j];
            dists_arr[5] = dists[lxm + k];
            dists_arr[5] = *std::max_element(dists_arr.cbegin() + 2, dists_arr.cbegin() + 6);
            multinom.set_index<3>(l);

            for (unsigned m = 0; m <= l; m++) {
              unsigned mxm(m*mult);
              double weight_ijklm(weight_ijkl * ws0[m]);
              dists_arr[6] = dists[mxm + i];
              dists_arr[7] = dists[mxm + j];
              dists_arr[8] = dists[mxm + k];
              dists_arr[9] = dists[mxm + l];
              dists_arr[9] = *std::max_element(dists_arr.cbegin() + 5, dists_arr.cbegin() + 10);
              multinom.set_index<4>(m);

              for (unsigned n = 0; n <= m; n++) {
                unsigned nxm(n*mult);
                double weight_ijklmn(weight_ijklm * ws0[n]);
                dists_arr[10] = dists[nxm + i];
                dists_arr[11] = dists[nxm + j];
                dists_arr[12] = dists[nxm + k];
                dists_arr[13] = dists[nxm + l];
                dists_arr[14] = dists[nxm + m];
                dists_arr[14] = *std::max_element(dists_arr.cbegin() + 9, dists_arr.cbegin() + 15);
                multinom.set_index<5>(n);

                for (unsigned o = 0; o <= n; o++) {
                  unsigned oxm(o*mult);
                  double weight_ijklmno(weight_ijklmn * ws0[o]);
                  dists_arr[15] = dists[oxm + i];
                  dists_arr[16] = dists[oxm + j];
                  dists_arr[17] = dists[oxm + k];
                  dists_arr[18] = dists[oxm + l];
                  dists_arr[19] = dists[oxm + m];
                  dists_arr[20] = dists[oxm + n];
                  dists_arr[20] = *std::max_element(dists_arr.cbegin() + 14, dists_arr.cbegin() + 21);
                  multinom.set_index<6>(o);

                  for (unsigned p = 0; p <= o; p++) {
                    unsigned pxm(p*mult);
                    double weight_ijklmnop(weight_ijklmno * ws0[p]);
                    dists_arr[21] = dists[pxm + i];
                    dists_arr[22] = dists[pxm + j];
                    dists_arr[23] = dists[pxm + k];
                    dists_arr[24] = dists[pxm + l];
                    dists_arr[25] = dists[pxm + m];
                    dists_arr[26] = dists[pxm + n];
                    dists_arr[27] = dists[pxm + o];
                    multinom.set_index_final(p);

                    // fill histogram
                    hist(hist::weight(multinom.value() * weight_ijklmnop),
                         *std::max_element(dists_arr.cbegin() + 20, dists_arr.cend()));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void eNc_sym(const EECEvent & event, int thread) {

    std::cout << "eNc_sym begin" << std::endl;
    
    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & hist(this->per_event_hists(thread)[0]);

    // nothing to do for empty events
    if (mult == 0) return;

    // containers for computation
    std::vector<double> dists_local(N_choose_2_), weights(N() + 1);
    std::vector<unsigned> inds(N()+1);
    DynamicMultinomial multinom(N());
    
    // initialize dists
    for (unsigned i = 0; i < N_choose_2_; i++)
      dists_local[i] = 0;

    // initialize weights
    weights[0] = event_weight;
    for (unsigned i = 1; i <= N(); i++)
      weights[i] = weights[i-1]*ws0[0];

    // initialize multinom and inds
    inds[0] = mult - 1;
    for (unsigned i = 0; i < N();) {
      multinom.set_index(i, 0);
      inds[++i] = 0;
    }

    std::cout << "eNc_sym comp" << std::endl;

    // infinite loop
    double & max_dist(dists_local[N_choose_2_ - 1]), & weight(weights[N()]);
    while (true) {

      // fill hist
      hist(hist::weight(multinom.value() * weight), max_dist);

      // start w at N and work down to 0
      unsigned w(N());
      for (; w > 0; w--) {

        // try to increment inner-most loop, if we can't set inds[w] = 0 and move on
        if (++inds[w] > inds[w-1]) inds[w] = 0;

        // we could increment index at position w
        else {

          // update everything depending on index w and beyond
          for (unsigned k = w; k <= N(); k++) {

            // set max dist properly
            unsigned ikxm(inds[k]*mult), m((k-1)*(k-2)/2), mstart(m - 1);
            for (unsigned n = 1; n < k; m++, n++)
              dists_local[m] = dists[ikxm + inds[n]];

            // determine max element for the appropriate range
            if (k > 2)
              dists_local[m-1] = *std::max_element(dists_local.cbegin() + mstart, dists_local.cbegin() + m);

            // set weight
            weights[k] = weights[k-1]*ws0[inds[k]];

            // set multinom properly
            multinom.set_index(k-1, inds[k]);
          }

          // exit for loop, as we've found a good set of indices
          break;
        }
      }

      // if w hit 0 then we're done
      if (w == 0) break;
    }

    std::cout << "eNc_sym done" << std::endl;
  }

  #ifdef BOOST_SERIALIZATION_ACCESS_HPP
    friend class boost::serialization::access;
  #endif

  #ifdef EEC_SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */) {
      ar & boost::serialization::base_object<EECBase>(*this)
         & boost::serialization::base_object<EECHist>(*this)
         & use_general_eNc_ & N_choose_2_;

      init_subclass(true);
    }
  #endif

}; // EECLongestSide

END_EEC_NAMESPACE

#endif // EEC_LONGESTSIDE_HH
