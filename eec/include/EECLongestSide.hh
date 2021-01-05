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

//  ______ ______ _____ 
// |  ____|  ____/ ____|
// | |__  | |__ | |     
// |  __| |  __|| |     
// | |____| |___| |____ 
// |______|______\_____|
//  _      ____  _   _  _____ ______  _____ _______    _____ _____ _____  ______ 
// | |    / __ \| \ | |/ ____|  ____|/ ____|__   __|  / ____|_   _|  __ \|  ____|
// | |   | |  | |  \| | |  __| |__  | (___    | |    | (___   | | | |  | | |__   
// | |   | |  | | . ` | | |_ |  __|  \___ \   | |     \___ \  | | | |  | |  __|  
// | |___| |__| | |\  | |__| | |____ ____) |  | |     ____) |_| |_| |__| | |____ 
// |______\____/|_| \_|\_____|______|_____/   |_|    |_____/|_____|_____/|______|

#ifndef EEC_LONGESTSIDE_HH
#define EEC_LONGESTSIDE_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <stdexcept>
#include <type_traits>

#include "EECBase.hh"
#include "EECHist.hh"
#include "EECMultinomial.hh"

// namespace for EEC code
namespace eec {

//-----------------------------------------------------------------------------
// EEC class for histogramming according to the maximum particle distance
//-----------------------------------------------------------------------------

template<class Transform = bh::axis::transform::id>
class EECLongestSide : public EECBase, public Hist1D<Transform> {

  bool use_general_eNc_;
  unsigned N_choose_2_;

  // function pointer to the actual computation that will be run
  void (EECLongestSide::*compute_eec_ptr_)();

public:

  typedef typename Hist1D<Transform>::Hist Hist;

  EECLongestSide(unsigned nbins, double axis_min, double axis_max,
                 unsigned N, bool norm = true,
                 const std::vector<double> & pt_powers = {1},
                 const std::vector<unsigned> & ch_powers = {0},
                 bool check_degen = false,
                 bool average_verts = false,
                 bool use_general_eNc = false) :
    EECBase(pt_powers, ch_powers, N, norm, check_degen, average_verts),
    Hist1D<Transform>(nbins, axis_min, axis_max),
    use_general_eNc_(use_general_eNc),
    N_choose_2_(N_*(N_-1)/2),
    compute_eec_ptr_(&EECLongestSide::eNc_sym)
  {
    // set pointer to function that will do the computation
    switch (N_) {
      case 2:
        if (nsym_ == 2) {
          if (!use_general_eNc_)
            compute_eec_ptr_ = &EECLongestSide::eec_ij_sym;
        }
        else
          compute_eec_ptr_ = &EECLongestSide::eec_no_sym;
        break;

      case 3:
        switch (nsym_) {
          case 3:
            if (!use_general_eNc_)
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
        if (!use_general_eNc_)
          compute_eec_ptr_ = &EECLongestSide::eeeec_ijkl_sym;
        break;

      case 5:
        assert(nsym_ == 5);
        if (!use_general_eNc_)
          compute_eec_ptr_ = &EECLongestSide::eeeeec_ijklm_sym;
        break;

      case 6:
        assert(nsym_ == 6);
        if (!use_general_eNc_)
          compute_eec_ptr_ = &EECLongestSide::eeeeeec_ijklmn_sym;
        break;

      default:
        if (N_ > 12)
          throw std::invalid_argument("N must be 12 or less due to the use of 32-bit integers (13! > 2^32)");
    }
  }

  virtual ~EECLongestSide() {}

  std::string description() const {
    unsigned nh(this->nhists());

    std::ostringstream oss;
    oss << "EECLongestSide::" << compname_ << '\n'
        << EECBase::description()
        << '\n'
        << "  using eNc_sym - " << (use_general_eNc_ ? "true" : "false") << '\n'
        << "  there " << (nh == 1 ? "is " : "are ") << nh << " histogram";

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
        unsigned jxm(j*mult_);
        double dist_ij(dists_[ixm + j]);
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
    std::array<double, 6> dists;
    Multinomial<4> multinom;
    for (unsigned i = 0; i < mult_; i++) {
      unsigned ixm(i*mult_);
      double weight_i(weight_ * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists[0] = dists_[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult_);
          double weight_ijk(weight_ij * ws0[k]);
          dists[1] = dists_[kxm + i];
          dists[2] = dists_[kxm + j];
          dists[2] = *std::max_element(dists.cbegin(), dists.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult_);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists[3] = dists_[lxm + i];
            dists[4] = dists_[lxm + j];
            dists[5] = dists_[lxm + k];
            multinom.set_index_final(l);

            // fill histogram
            hist(bh::weight(multinom.value() * weight_ijkl),
                 *std::max_element(dists.cbegin() + 2, dists.cend()));
          }
        }
      }
    }
  }

  void eeeeec_ijklm_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // loop over quintuplets of particles
    std::array<double, 10> dists;
    Multinomial<5> multinom;
    for (unsigned i = 0; i < mult_; i++) {
      unsigned ixm(i*mult_);
      double weight_i(weight_ * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists[0] = dists_[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult_);
          double weight_ijk(weight_ij * ws0[k]);
          dists[1] = dists_[kxm + i];
          dists[2] = dists_[kxm + j];
          dists[2] = *std::max_element(dists.cbegin(), dists.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult_);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists[3] = dists_[lxm + i];
            dists[4] = dists_[lxm + j];
            dists[5] = dists_[lxm + k];
            dists[5] = *std::max_element(dists.cbegin() + 2, dists.cbegin() + 6);
            multinom.set_index<3>(l);

            for (unsigned m = 0; m <= l; m++) {
              unsigned mxm(m*mult_);
              double weight_ijklm(weight_ijkl * ws0[m]);
              dists[6] = dists_[mxm + i];
              dists[7] = dists_[mxm + j];
              dists[8] = dists_[mxm + k];
              dists[9] = dists_[mxm + l];
              multinom.set_index_final(m);

              // fill histogram
              hist(bh::weight(multinom.value() * weight_ijklm),
                   *std::max_element(dists.cbegin() + 5, dists.cend()));
            }
          }
        }
      }
    }
  }

  void eeeeeec_ijklmn_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // loop over quintuplets of particles
    std::array<double, 15> dists;
    Multinomial<6> multinom;
    for (unsigned i = 0; i < mult_; i++) {
      unsigned ixm(i*mult_);
      double weight_i(weight_ * ws0[i]);
      multinom.set_index_0(i);

      for (unsigned j = 0; j <= i; j++) {
        double weight_ij(weight_i * ws0[j]);
        dists[0] = dists_[ixm + j];
        multinom.set_index<1>(j);

        for (unsigned k = 0; k <= j; k++) {
          unsigned kxm(k*mult_);
          double weight_ijk(weight_ij * ws0[k]);
          dists[1] = dists_[kxm + i];
          dists[2] = dists_[kxm + j];
          dists[2] = *std::max_element(dists.cbegin(), dists.cbegin() + 3);
          multinom.set_index<2>(k);

          for (unsigned l = 0; l <= k; l++) {
            unsigned lxm(l*mult_);
            double weight_ijkl(weight_ijk * ws0[l]);
            dists[3] = dists_[lxm + i];
            dists[4] = dists_[lxm + j];
            dists[5] = dists_[lxm + k];
            dists[5] = *std::max_element(dists.cbegin() + 2, dists.cbegin() + 6);
            multinom.set_index<3>(l);

            for (unsigned m = 0; m <= l; m++) {
              unsigned mxm(m*mult_);
              double weight_ijklm(weight_ijkl * ws0[m]);
              dists[6] = dists_[mxm + i];
              dists[7] = dists_[mxm + j];
              dists[8] = dists_[mxm + k];
              dists[9] = dists_[mxm + l];
              dists[9] = *std::max_element(dists.cbegin() + 5, dists.cbegin() + 10);
              multinom.set_index<4>(m);

              for (unsigned n = 0; n <= m; n++) {
                unsigned nxm(n*mult_);
                double weight_ijklmn(weight_ijklm * ws0[n]);
                dists[10] = dists_[nxm + i];
                dists[11] = dists_[nxm + j];
                dists[12] = dists_[nxm + k];
                dists[13] = dists_[nxm + l];
                dists[14] = dists_[nxm + m];
                multinom.set_index_final(n);

                // fill histogram
                hist(bh::weight(multinom.value() * weight_ijklmn),
                     *std::max_element(dists.cbegin() + 9, dists.cend()));
              }
            }
          }
        }
      }
    }
  }

  void eNc_sym() {
    const std::vector<double> & ws0(weights_[0]);
    Hist & hist(this->hists[0]);

    // nothing to do for empty events
    if (mult_ == 0) return;

    // containers for computation
    std::vector<double> dists(N_choose_2_), weights(N_+1);
    std::vector<unsigned> inds(N_+1);
    DynamicMultinomial multinom(N_);
    
    // initialize dists
    for (unsigned i = 0; i < N_choose_2_; i++)
      dists[i] = 0;

    // initialize weights
    weights[0] = 1;
    for (unsigned i = 1; i <= N_; i++)
      weights[i] = weights[i-1]*ws0[0];

    // initialize multinom and inds
    inds[0] = mult_ - 1;
    for (unsigned i = 0; i < N_;) {
      multinom.set_index(i, 0);
      inds[++i] = 0;
    }

    // infinite loop
    double & max_dist(dists[N_choose_2_ - 1]), & weight(weights[N_]);
    while (true) {

      // fill hist
      hist(bh::weight(multinom.value() * weight), max_dist);

      // start w at N and work down to 0
      unsigned w(N_);
      for (; w > 0; w--) {

        // try to increment inner-most loop, if we can't set inds[w] = 0 and move on
        if (++inds[w] > inds[w-1]) inds[w] = 0;

        // we could increment index at position w
        else {

          // update everything depending on index w and beyond
          for (unsigned k = w; k <= N_; k++) {

            // set max dist properly
            unsigned ikxm(inds[k]*mult_), m((k-1)*(k-2)/2), mstart(m - 1);
            for (unsigned n = 1; n < k; m++, n++)
              dists[m] = dists_[ikxm + inds[n]];

            // determine max element for the appropriate range
            if (k > 2)
              dists[m-1] = *std::max_element(dists.cbegin() + mstart, dists.cbegin() + m);

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
  }

}; // EECLongestSide

} // namespace eec

#endif // EEC_LONGESTSIDE_HH
