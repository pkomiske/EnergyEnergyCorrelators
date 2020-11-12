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

#ifndef EEC_TRIANGLEOPE_HH
#define EEC_TRIANGLEOPE_HH

#include <algorithm>
#include <array>

#include "EECBase.hh"
#include "EECHist.hh"

// namespace for EEC code
namespace eec {

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

inline std::array<std::pair<double,int>, 3> argsort3(const std::array<double, 3> & dists) {

  // create array of dists with indices
  std::array<std::pair<double,int>, 3> dists_inds{std::make_pair(dists[0], 0),
                                                  std::make_pair(dists[1], 1),
                                                  std::make_pair(dists[2], 2)};

  // sort according to dists
  std::sort(dists_inds.begin(), dists_inds.end(), 
            [](const std::pair<double,int> & a, const std::pair<double,int> & b){return a.first < b.first;});

  return dists_inds;
}

//-----------------------------------------------------------------------------
// EEEC class fully differential in the three particle distances
//-----------------------------------------------------------------------------

template<class Transform0 = bh::axis::transform::id,
         class Transform1 = bh::axis::transform::id,
         class Transform2 = bh::axis::transform::id>
class EECTriangleOPE : public EECBase, public Hist3D<Transform0, Transform1, Transform2> {

  // function pointer to the actual computation that will be run
  void (EECTriangleOPE::*compute_eec_ptr_)();

public:

  EECTriangleOPE(unsigned nbins0, double axis0_min, double axis0_max,
                 unsigned nbins1, double axis1_min, double axis1_max,
                 unsigned nbins2, double axis2_min, double axis2_max,
                 bool norm = true,
                 const std::vector<double> & pt_powers = {1},
                 const std::vector<unsigned> & ch_powers = {0},
                 bool check_degen = false,
                 bool average_verts = false) :
    EECBase(pt_powers, ch_powers, 3, norm, check_degen, average_verts),
    Hist3D<Transform0, Transform1, Transform2>(nbins0, axis0_min, axis0_max,
                                               nbins1, axis1_min, axis1_max,
                                               nbins2, axis2_min, axis2_max)
  {
    switch (nsym_) {
      case 3:
        compute_eec_ptr_ = &EECTriangleOPE::eeec_ijk_sym;
        break;

      case 2:
        compute_eec_ptr_ = &EECTriangleOPE::eeec_ij_sym;
        if (!average_verts_) this->duplicate_hists(3);
        break;

      case 0:
        compute_eec_ptr_ = &EECTriangleOPE::eeec_no_sym;
        if (!average_verts_) this->duplicate_hists(6);
        break;

      default:
        throw std::runtime_error("Invalid number of symmetries " + std::to_string(nsym_));
    }
  }

  virtual ~EECTriangleOPE() {}

  std::string description() const {
    unsigned nh(this->nhists());

    std::ostringstream oss;
    oss << "EECTriangleOPE::" << compname_ << '\n'
        << EECBase::description()
        << '\n'
        << "  there " << (nh == 1 ? "is " : "are ") << nh << " histogram";

    if (nh == 1) 
      oss << '\n';

    else if (nh == 3)
      oss << "s, labeled according to the location of the side with identical vertices\n"
          << "    0 - distinguished side is the small side\n"
          << "    1 - distinguished side is the medium side\n"
          << "    2 - distinguished side is the large side\n";

    else if (nh == 6)
      oss << "s, labeled according to the locations of sides ij and ik\n"
          << "    0 - side ij is the small side, side ik is the medium side\n"
          << "    1 - side ij is the small side, side ik is the large side\n"
          << "    2 - side ij is the medium side, side ik is the small side\n"
          << "    3 - side ij is the medium side, side ik is the large side\n"
          << "    4 - side ij is the large side, side ik is the small side\n"
          << "    5 - side ij is the large side, side ik is the medium side\n";

    else 
      throw std::runtime_error("Unexpected number of histograms encountered");

    return oss.str();
  }

private:

  void compute_eec() { (this->*compute_eec_ptr_)(); }

  void eeec_ijk_sym() {
    const std::vector<double> & ws0(weights_[0]);
    std::array<double, 3> dists;

    // loop over symmetric triplets of particles
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        if (weight_ij == 0) continue;
        unsigned jxm(j*mult_);
        bool ij_match(i == j);
        double dist_ij(dists_[ixm + j]);

        for (unsigned k = 0; k <= j; k++) {
          //if (ws0[k] == 0) continue;
          bool ik_match(i == k), jk_match(j == k);
          int sym(!(ij_match || ik_match || jk_match) ? 6 : (ij_match && ik_match ? 1 : 3));
          dists[0] = dist_ij;
          dists[1] = dists_[ixm + k];
          dists[2] = dists_[jxm + k];

          std::sort(dists.begin(), dists.end());
          fill_hist(0, weight_ij * ws0[k] * sym, dists[0], dists[1], dists[2]);
        }
      }
    }
  }

  void eeec_ij_sym() {
    const std::vector<double> & ws0(weights_[0]), & ws1(weights_[1]);
    std::array<double, 3> dists;

    // first index is special, second is symmetric
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j] * (i == j ? 1 : 2));
        if (weight_ij == 0) continue;
        unsigned jxm(j*mult_);
        double dist_ij(dists_[ixm + j]);

        for (unsigned k = 0; k < mult_; k++) {
          //if (ws1[k] == 0) continue;
          double weight_ijk(weight_ij * ws1[k]);
          dists[0] = dist_ij;
          dists[1] = dists_[ixm + k];
          dists[2] = dists_[jxm + k];

          // (arg)sort distances
          std::array<std::pair<double,int>, 3> dists_inds(argsort3(dists));

          // check for overlapping particles
          bool ik_match(i == k), jk_match(j == k);

          // averaging over verts
          if (average_verts_)
            fill_hist(0, weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);

          // fill specific histogram
          else if (!(ik_match || jk_match))
            fill_hist(dists_inds[0].second == 0 ? 0 : (dists_inds[1].second == 0 ? 1 : 2),
                      weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);

          // fill medium and large histograms
          else if (ik_match || jk_match) {
            fill_hist(1, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(2, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // fill all histograms
          else if (ik_match && jk_match) {
            fill_hist(0, weight_ijk, 0, 0, 0);
            fill_hist(1, weight_ijk, 0, 0, 0);
            fill_hist(2, weight_ijk, 0, 0, 0);
          }  
        }
      }
    }
  }

  void eeec_no_sym() {
    const std::vector<double> & ws0(weights_[0]), & ws1(weights_[1]), & ws2(weights_[2]);
    std::array<double, 3> dists;

    // all indices are different
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j < mult_; j++) {
        double weight_ij(weight_i * ws1[j]);
        if (weight_ij == 0) continue;
        unsigned jxm(j*mult_);
        double dist_ij(dists_[ixm + j]);
        bool ij_match(i == j);

        for (unsigned k = 0; k < mult_; k++) {
          //if (ws2[k] == 0) continue;
          double weight_ijk(weight_ij * ws2[k]);
          dists[0] = dist_ij;
          dists[1] = dists_[ixm + k];
          dists[2] = dists_[jxm + k];
          bool ik_match(i == k), jk_match(j == k);

          // (arg)sort distances
          std::array<std::pair<double,int>, 3> dists_inds(argsort3(dists));

          // check for averaging the vertices
          if (average_verts_)
            fill_hist(0, weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);

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

            fill_hist(hist_i, weight_ijk, dists_inds[0].first, dists_inds[1].first, dists_inds[2].first);
          }

          // everything is degenerate
          else if (ij_match && ik_match) {
            for (unsigned hist_i : {0, 1, 2, 3, 4, 5})
              fill_hist(hist_i, weight_ijk, 0, 0, 0);
          }

          // ij are degenerate, fill hists 0 and 1
          else if (ij_match) {
            fill_hist(0, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(1, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // ik are degenerate, fill hists 2 and 4
          else if (ik_match) {
            fill_hist(2, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(4, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // jk are degenerate, fill hists 3 and 5
          else if (jk_match) {
            fill_hist(3, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
            fill_hist(5, weight_ijk, 0, dists_inds[1].first, dists_inds[2].first);
          }

          // should never get here
          else throw std::runtime_error("should never get here in EECTriangleOPE::eeec_no_sym");
        }
      }
    }
  }

  void fill_hist(unsigned hist_i, double weight, double xS, double xM, double xL) {

    // define coordinate mapping
    double xi(xS/(xM + REG)), diff(xL - xM), 
           phi(std::asin(std::sqrt(std::fabs(1 - diff*diff/(xS*xS + REG)))));

    // fill histogram
    this->hists[hist_i](bh::weight(weight), xL, xi, phi);
  }

}; // EECTriangleOPE

} // namespace eec

#endif // EEC_TRIANGLEOPE_HH
