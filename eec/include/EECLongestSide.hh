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

#ifndef EEC_LONGESTSIDE_HH
#define EEC_LONGESTSIDE_HH

#include <algorithm>
#include <array>
#include <cassert>

#include "EECBase.hh"
#include "EECHist.hh"

// namespace for EEC code
namespace eec {

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

const unsigned FACTORIALS[12] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800};

// multinomial factor on sorted indices
template<std::size_t N>
inline unsigned multinomial(const std::array<unsigned, N> & inds) noexcept {

  unsigned denom(1), count(1);
  for (unsigned i = 1; i < N; i++) {
    if (inds[i] == inds[i-1]) count++;
    else {
      denom *= FACTORIALS[count];
      count = 1;
    }
  }
  denom *= FACTORIALS[count];

  return FACTORIALS[N]/denom;
}

//-----------------------------------------------------------------------------
// EEC class for histogramming according to the maximum particle distance
//-----------------------------------------------------------------------------

template<class Transform = bh::axis::transform::id>
class EECLongestSide : public EECBase, public Hist1D<Transform> {

  // function pointer to the actual computation that will be run
  void (EECLongestSide::*compute_eec_ptr_)();

public:

  typedef typename Hist1D<Transform>::Hist Hist;

  EECLongestSide(unsigned nbins, double axis_min, double axis_max,
                 unsigned N, bool norm = true,
                 const std::vector<double> & pt_powers = {1},
                 const std::vector<unsigned> & ch_powers = {0},
                 bool check_degen = false,
                 bool average_verts = false) :
    EECBase(pt_powers, ch_powers, N, norm, check_degen, average_verts),
    Hist1D<Transform>(nbins, axis_min, axis_max)
  {
    // set pointer to function that will do the computation
    switch (N_) {
      case 2:
        compute_eec_ptr_ = (nsym_ == 2 ? &EECLongestSide::eec_ij_sym : &EECLongestSide::eec_no_sym);
        break;

      case 3:
        switch (nsym_) {
          case 3:
            compute_eec_ptr_ = &EECLongestSide::eeec_ijk_sym;
            break;

          case 2:
            compute_eec_ptr_ = &EECLongestSide::eeec_ij_sym;
            if (!average_verts_) this->duplicate_hists(2);
            break;

          case 0:
            compute_eec_ptr_ = &EECLongestSide::eeec_no_sym;
            if (!average_verts_) this->duplicate_hists(3);
            break;

          default:
            throw std::runtime_error("Invalid number of symmetries " + std::to_string(nsym_));
        }
        break;

      case 4:
        assert(nsym_ == 4);
        compute_eec_ptr_ = &EECLongestSide::eeeec_ijkl_sym;
        break;

      case 5:
        assert(nsym_ == 5);
        compute_eec_ptr_ = &EECLongestSide::eeeeec_ijklm_sym;
        break;
    }
  }

  virtual ~EECLongestSide() {}

  std::string description() const {
    unsigned nh(this->nhists());

    std::ostringstream oss;
    oss << "EECLongestSide::" << compname_ << '\n'
        << EECBase::description()
        << '\n'
        << "  There " << (nh == 1 ? "is " : "are ") << nh << " histogram";

    if (nh == 1) 
      oss << '\n';

    else if (nh == 2)
      oss << "s, labeled according to the location of the largest side\n"
          << "    0 - the largest side is the one with identical vertices\n"
          << "    1 - the largest side is the one with different vertices\n";

    else if (nh == 3)
      oss << "s, labeled according to the location of the largest side\n"
          << "    0 - the largest side is ij\n"
          << "    1 - the largest side is jk\n"
          << "    2 - the largest side is ik\n";

    else 
      throw std::runtime_error("Unexpected number of histograms encountered");

    return oss.str();
  }

private:
  
  void compute_eec() { (this->*compute_eec_ptr_)(); }

  void eec_ij_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // loop over symmetric pairs of particles
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      // i == j term
      hist(bh::weight(weight_i * ws0[i]), 0);

      // off diagonal terms
      weight_i *= 2;
      for (unsigned j = 0; j < i; j++)
        hist(bh::weight(weight_i * ws0[j]), dists_[ixm + j]);
    }
  }

  void eec_no_sym() {
    const std::vector<double> & ws0(weights_[0]), & ws1(weights_[1]);
    Hist & hist(this->hists[0]);

    // loop over all pairs of particles
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j < mult_; j++)
        hist(bh::weight(weight_i * ws1[j]), dists_[ixm + j]);
    }
  }

  void eeec_ijk_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // loop over triplets of particles
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        if (weight_ij == 0) continue;
        double dist_ij(dists_[ixm + j]);
        unsigned jxm(j*mult_);
        bool ij_match(i == j);

        for (unsigned k = 0; k <= j; k++) {
          double dist_ik(dists_[ixm + k]), dist_jk(dists_[jxm + k]);
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
          hist(bh::weight(weight_ij * ws0[k] * sym), max_dist);
        }
      }
    }
  }

  void eeec_ij_sym() {
    const std::vector<double> & ws0(weights_[0]), & ws1(weights_[1]);

    // loop over triplets of particles
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j] * (i == j ? 1 : 2));
        if (weight_ij == 0) continue;
        double dist_ij(dists_[ixm + j]);
        unsigned jxm(j*mult_);

        for (unsigned k = 0; k < mult_; k++) {
          double weight_ijk(weight_ij * ws1[k]), dist_ik(dists_[ixm + k]), dist_jk(dists_[jxm + k]);

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
          if (average_verts_)
            this->hists[0](bh::weight(weight_ijk), max_dist);

          // no averaging here, fill the targeted hist
          else {
            this->hists[hist_i](bh::weight(weight_ijk), max_dist);

            // fill other histogram if k == i or k == j
            if (k == i || k == j)
              this->hists[hist_i == 0 ? 1 : 0](bh::weight(weight_ijk), max_dist);  
          }
        }
      }
    }
  }

  void eeec_no_sym() {
    const std::vector<double> & ws0(weights_[0]), & ws1(weights_[1]), & ws2(weights_[2]);

    // loop over unique triplets of particles
    for (unsigned i = 0; i < mult_; i++) {
      double weight_i(weight_ * ws0[i]);
      if (weight_i == 0) continue;
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j < mult_; j++) {
        double weight_ij(weight_i * ws1[j]);
        if (weight_ij == 0) continue;
        double dist_ij(dists_[ixm + j]);
        unsigned jxm(j*mult_);
        bool ij_match(i == j);

        for (unsigned k = 0; k < mult_; k++) {
          double weight_ijk(weight_ij * ws2[k]), dist_ik(dists_[ixm + k]), dist_jk(dists_[jxm + k]);
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
          if (average_verts_)
            this->hists[0](bh::weight(weight_ijk), max_dist);

          // no degeneracy at all
          else if (!(ij_match || ik_match || jk_match))
            this->hists[hist_i](bh::weight(weight_ijk), max_dist);

          // everything is degenerate
          else if (ij_match && ik_match) {
            this->hists[0](bh::weight(weight_ijk), max_dist);
            this->hists[1](bh::weight(weight_ijk), max_dist);
            this->hists[2](bh::weight(weight_ijk), max_dist);
          }

          // ij overlap, largest sides are ik and jk
          else if (ij_match) {
            this->hists[1](bh::weight(weight_ijk), max_dist);
            this->hists[2](bh::weight(weight_ijk), max_dist);
          }

          // jk overlap, largest sides are ij, ik
          else if (jk_match) {
            this->hists[0](bh::weight(weight_ijk), max_dist);
            this->hists[2](bh::weight(weight_ijk), max_dist);
          }

          // ik overlap, largest sides are ij, jk
          else if (ik_match) {
            this->hists[0](bh::weight(weight_ijk), max_dist);
            this->hists[1](bh::weight(weight_ijk), max_dist);
          }

          // should never get here
          else throw std::runtime_error("should never get here in EECLongestSide::eeec_no_sym");
        }
      }
    }
  }

  void eeeec_ijkl_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // loop over quadruplets of particles
    std::array<unsigned, 4> inds;
    std::array<double, 6> dists;
    for (unsigned i = 0; i < mult_; i++) {
      inds[0] = i;
      double weight_i(weight_ * ws0[i]);
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j <= i; j++) {
        inds[1] = j;
        double weight_ij(weight_i * ws0[j]);
        dists[0] = dists_[ixm + j];
        unsigned jxm(j*mult_);

        for (unsigned k = 0; k <= j; k++) {
          inds[2] = k;
          double weight_ijk(weight_ij * ws0[k]);
          dists[1] = dists_[ixm + k];
          dists[2] = dists_[jxm + k];
          unsigned kxm(k*mult_);

          for (unsigned l = 0; l <= k; l++) {
            inds[3] = l;
            double weight_ijkl(multinomial(inds) * weight_ijk * ws0[l]);
            dists[3] = dists_[ixm + l];
            dists[4] = dists_[jxm + l];
            dists[5] = dists_[kxm + l];

            // fill histogram
            hist(bh::weight(weight_ijkl), *std::max_element(dists.begin(), dists.end()));
          }
        }
      }
    }
  }

  void eeeeec_ijklm_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // loop over quintuplets of particles
    std::array<unsigned, 5> inds;
    std::array<double, 10> dists;
    for (unsigned i = 0; i < mult_; i++) {
      inds[0] = i;
      double weight_i(weight_ * ws0[i]);
      unsigned ixm(i*mult_);

      for (unsigned j = 0; j <= i; j++) {
        inds[1] = j;
        double weight_ij(weight_i * ws0[j]);
        dists[0] = dists_[ixm + j];
        unsigned jxm(j*mult_);

        for (unsigned k = 0; k <= j; k++) {
          inds[2] = k;
          double weight_ijk(weight_ij * ws0[k]);
          dists[1] = dists_[ixm + k];
          dists[2] = dists_[jxm + k];
          unsigned kxm(k*mult_);

          for (unsigned l = 0; l <= k; l++) {
            inds[3] = l;
            double weight_ijkl(weight_ijk * ws0[l]);
            dists[3] = dists_[ixm + l];
            dists[4] = dists_[jxm + l];
            dists[5] = dists_[kxm + l];
            unsigned lxm(l*mult_);

            for (unsigned m = 0; m <= l; m++) {
              inds[4] = m;
              double weight_ijklm(multinomial(inds) * weight_ijkl * ws0[m]);
              dists[6] = dists_[ixm + m];
              dists[7] = dists_[jxm + m];
              dists[8] = dists_[kxm + m];
              dists[9] = dists_[lxm + m];

              // fill histogram
              hist(bh::weight(weight_ijklm), *std::max_element(dists.begin(), dists.end()));
            }
          }
        }
      }
    }
  }
};

} // namespace eec

#endif // EEC_LONGESTSIDE_HH
