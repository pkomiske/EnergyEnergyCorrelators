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
 *   _______ _____  _____          _   _  _____ _      ______    ____  _____  ______ 
 *  |__   __|  __ \|_   _|   /\   | \ | |/ ____| |    |  ____|  / __ \|  __ \|  ____|
 *     | |  | |__) | | |    /  \  |  \| | |  __| |    | |__    | |  | | |__) | |__   
 *     | |  |  _  /  | |   / /\ \ | . ` | | |_ | |    |  __|   | |  | |  ___/|  __|  
 *     | |  | | \ \ _| |_ / ____ \| |\  | |__| | |____| |____  | |__| | |    | |____ 
 *     |_|  |_|  \_\_____/_/    \_\_| \_|\_____|______|______|  \____/|_|    |______|
 */

#ifndef EEC_TRIANGLEOPE_HH
#define EEC_TRIANGLEOPE_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "EECBase.hh"
#include "EECHist3D.hh"

BEGIN_EEC_NAMESPACE

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

void argsort3(std::array<std::pair<double, int>, 3> & dists_inds) {

  // set integers
  dists_inds[0].second = 0;
  dists_inds[1].second = 1;
  dists_inds[2].second = 2;

  // sort according to dists
  std::sort(dists_inds.begin(), dists_inds.end(), 
            [](const auto & a, const auto & b){ return a.first < b.first; });
}

template<class Hist>
void fill_hist(Hist & hist, double weight, double xS, double xM, double xL) {

    // define coordinate mapping
    double xi(xS/(xM + REG)), diff(xL - xM), 
           phi(std::asin(std::sqrt(1 - diff*diff/(xS*xS + REG))));

    // fill histogram
    hist(hist::weight(weight), xL, xi, phi);
  }

//-----------------------------------------------------------------------------
// EEEC class fully differential in the three particle distances
//-----------------------------------------------------------------------------

template<class Transform0, class Transform1, class Transform2>
class EECTriangleOPE : public EECBase, public hist::EECHist3D<Transform0, Transform1, Transform2> {
public:

  typedef EECTriangleOPE<Transform0, Transform1, Transform2> Self;
  typedef hist::EECHist3D<Transform0, Transform1, Transform2> EECHist;
  typedef typename EECHist::SimpleWeightedHist SimpleWeightedHist;

#ifndef SWIG_PREPROCESSOR

  // default constructor
  EECTriangleOPE() : EECTriangleOPE(EECConfig(3), {1, 1, 1}) {}

  // constructor taking EECConfig and Hist3D arguments
  EECTriangleOPE(const EECConfig & config,
                 const std::array<unsigned, 3> & nbins,
                 const std::array<std::array<double, 2>, 3> & axes_range = {{{1e-5, 1}, {1e-5, 1}, {0, PI/2}}},
                 bool track_covariance = false,
                 bool variance_bound = true,
                 bool variance_bound_includes_overflows = true) :

    // construct EECBase from provided EECConfig
    EECBase(config),

    // construct EECHist3D from EECConfig of options
    EECHist(nbins, axes_range, EECBase::num_threads(),
            track_covariance, variance_bound, variance_bound_includes_overflows)
  {
    init_subclass();
  }

#endif // !SWIG_PREPROCESSOR

  // constructor with all options (mostly useful for Python)
  EECTriangleOPE(const std::array<unsigned, 3> & nbins,
                 const std::array<std::array<double, 2>, 3> & axes_range = {{{1e-5, 1}, {1e-5, 1}, {0, PI/2}}},
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
                 bool track_covariance = false,
                 bool variance_bound = true,
                 bool variance_bound_includes_overflows = true) :
    EECTriangleOPE(EECConfig(3, norm, weight_powers, charge_powers,
                             particle_weight, pairwise_distance,
                             num_threads, omp_chunksize, print_every,
                             check_degen, average_verts),
                   nbins, axes_range,
                   track_covariance,
                   variance_bound,
                   variance_bound_includes_overflows)
  {}

  virtual ~EECTriangleOPE() = default;

  void set_num_threads(int threads) {
    EECBase::set_num_threads(threads);
    EECHist::set_num_threads(threads);
  }

  std::string description(int hist_level = 1) const {
    unsigned nh(this->nhists());

    std::ostringstream oss;
    oss << "EECTriangleOPE<" << EECHist::axes_description()
        << ">::" << EECBase::description() << '\n'
        << "  " << EECHist::hist_name() << " -  there " << (nh == 1 ? "is " : "are ") << nh << " histogram";

    if (nh == 1) 
      oss << '\n';

    else if (nh == 3)
      oss << "s, labeled according to the location of the (distinguished) side with identical vertices\n"
          << "    0 - distinguished side is the small side\n"
          << "    1 - distinguished side is the medium side\n"
          << "    2 - distinguished side is the large side\n";

    else if (nh == 6)
      oss << "s, labeled according to the locations of sides ij and ik (vertices are i, j, k)\n"
          << "    0 - side ij is the small side, side ik is the medium side\n"
          << "    1 - side ij is the small side, side ik is the large side\n"
          << "    2 - side ij is the medium side, side ik is the small side\n"
          << "    3 - side ij is the medium side, side ik is the large side\n"
          << "    4 - side ij is the large side, side ik is the small side\n"
          << "    5 - side ij is the large side, side ik is the medium side\n";

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

  bool operator!=(const EECTriangleOPE & rhs) const { return !operator==(rhs); }
  bool operator==(const EECTriangleOPE & rhs) const {
    return EECBase::operator==(rhs)   &&
           EECHist::operator==(rhs) &&
           compute_eec_func_ptr_ == rhs.compute_eec_func_ptr_;
  }

  EECTriangleOPE & operator+=(const EECTriangleOPE & rhs) {
    EECBase::operator+=(rhs);
    EECHist::operator+=(rhs);

    return *this;
  }

  void clear() {
    EECHist::clear_hists();
    EECBase::clear_events();
  }

private:

  // function pointer to the actual computation that will be run
  void (EECTriangleOPE::*compute_eec_func_ptr_)(const EECEvent &, int);

  // provides initialization of this subclass from configuration
  void init_subclass(bool events_allowed = false) {
    if (!events_allowed)
      ensure_no_events();

    switch (nsym()) {
      case 3:
        compute_eec_func_ptr_ = &EECTriangleOPE::eeec_ijk_sym;
        break;

      case 2:
        compute_eec_func_ptr_ = &EECTriangleOPE::eeec_ij_sym;
        if (!average_verts()) this->duplicate_histograms(3);
        break;

      case 0:
        compute_eec_func_ptr_ = &EECTriangleOPE::eeec_no_sym;
        if (!average_verts()) this->duplicate_histograms(6);
        break;

      default:
        throw std::runtime_error("Invalid number of symmetries " + std::to_string(nsym()));
    }
  }

  // implements pure virtual method from base class
  void compute_eec_internal(const EECEvent & event, int thread) {
    (this->*compute_eec_func_ptr_)(event, thread);
    this->fill_from_single_event(thread);
  }

  void eeec_ijk_sym(const EECEvent & event, int thread) {

    // references to event data
    const std::vector<double> & ws0(event.weights()[0]),
                              & dists(event.dists());
    double event_weight(event.event_weight());
    unsigned mult(event.mult());

    // reference to event hists
    SimpleWeightedHist & simple_hist(this->per_event_hists(thread)[0]);

    // loop over symmetric triplets of particles
    std::array<double, 3> dists_arr;
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      unsigned ixm(i*mult);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        unsigned jxm(j*mult);
        bool ij_match(i == j);
        double dist_ij(dists[ixm + j]);

        for (unsigned k = 0; k <= j; k++) {
          bool ik_match(i == k), jk_match(j == k);
          int sym(!(ij_match || ik_match || jk_match) ? 6 : (ij_match && ik_match ? 1 : 3));
          dists_arr[0] = dist_ij;
          dists_arr[1] = dists[ixm + k];
          dists_arr[2] = dists[jxm + k];

          std::sort(dists_arr.begin(), dists_arr.end());
          fill_hist(simple_hist, weight_ij * ws0[k] * sym, dists_arr[0], dists_arr[1], dists_arr[2]);
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

    // first index is special, second is symmetric
    std::array<std::pair<double, int>, 3> dists_inds;
    for (unsigned i = 0; i < mult; i++) {
      double weight_i(event_weight * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j] * (i == j ? 1 : 2));
        if (weight_ij == 0) continue;
        unsigned jxm(j*mult);
        double dist_ij(dists[ixm + j]);

        for (unsigned k = 0; k < mult; k++) {
          double weight_ijk(weight_ij * ws1[k]);
          dists_inds[0].first = dist_ij;
          dists_inds[1].first = dists[ixm + k];
          dists_inds[2].first = dists[jxm + k];

          // (arg)sort distances
          argsort3(dists_inds);

          // check for overlapping particles
          bool ik_match(i == k), jk_match(j == k);

          // averaging over verts
          if (average_verts())
            fill_hist(hists[0], weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);

          // fill specific histogram
          else if (!(ik_match || jk_match))
            fill_hist(hists[dists_inds[0].second == 0 ? 0 : (dists_inds[1].second == 0 ? 1 : 2)],
                      weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);

          // fill all histograms
          else if (ik_match && jk_match) {
            fill_hist(hists[0], weight_ijk, 0, 0, 0);
            fill_hist(hists[1], weight_ijk, 0, 0, 0);
            fill_hist(hists[2], weight_ijk, 0, 0, 0);
          }

          // fill medium and large histograms
          else if (ik_match || jk_match) {
            fill_hist(hists[1], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(hists[2], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
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

    // all indices are different
    std::array<std::pair<double, int>, 3> dists_inds;
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
          double weight_ijk(weight_ij * ws2[k]);
          dists_inds[0].first = dist_ij;
          dists_inds[1].first = dists[ixm + k];
          dists_inds[2].first = dists[jxm + k];
          bool ik_match(i == k), jk_match(j == k);

          // (arg)sort distances
          argsort3(dists_inds);

          // check for averaging the vertices
          if (average_verts())
            fill_hist(hists[0], weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);

          // no degeneracy 
          else if (!(ij_match || ik_match || jk_match)) {
            unsigned hist_i;

            // check for 0 being ij
            if (dists_inds[0].second == 0) {

              // check for 1 being ik
              if (dists_inds[1].second == 1) hist_i = 0;
              else hist_i = 1;
            }

            // check for 0 being ik
            else if (dists_inds[0].second == 1) {

              // check for 1 being ij
              if (dists_inds[1].second == 0) hist_i = 2;
              else hist_i = 4;
            }

            // 0 must be jk in this case
            else {
              if (dists_inds[1].second == 0) hist_i = 3;
              else hist_i = 5;
            }

            fill_hist(hists[hist_i], weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);
          }

          // everything is degenerate
          else if (ij_match && ik_match) {
            fill_hist(hists[0], weight_ijk, 0, 0, 0);
            fill_hist(hists[1], weight_ijk, 0, 0, 0);
            fill_hist(hists[2], weight_ijk, 0, 0, 0);
            fill_hist(hists[3], weight_ijk, 0, 0, 0);
            fill_hist(hists[4], weight_ijk, 0, 0, 0);
            fill_hist(hists[5], weight_ijk, 0, 0, 0);
          }

          // ij are degenerate, fill hists 0 and 1
          else if (ij_match) {
            fill_hist(hists[0], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(hists[1], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // ik are degenerate, fill hists 2 and 4
          else if (ik_match) {
            fill_hist(hists[2], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(hists[4], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // jk are degenerate, fill hists 3 and 5
          else if (jk_match) {
            fill_hist(hists[3], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(hists[5], weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // should never get here
          else throw std::runtime_error("should never get here in EECTriangleOPE::eeec_no_sym");
        }
      }
    }
  }

  #ifdef BOOST_SERIALIZATION_ACCESS_HPP
    friend class boost::serialization::access;
  #endif

  #ifdef EEC_SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /* file_version */) {
      ar & boost::serialization::base_object<EECBase>(*this)
         & boost::serialization::base_object<EECHist>(*this);

      init_subclass(true);
    }
  #endif

}; // EECTriangleOPE

END_EEC_NAMESPACE

#endif // EEC_TRIANGLEOPE_HH
