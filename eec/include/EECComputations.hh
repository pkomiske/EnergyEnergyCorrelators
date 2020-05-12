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

#ifndef EEC_COMPUTATIONS_HH
#define EEC_COMPUTATIONS_HH

#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

#include "boost/histogram.hpp"

#ifdef FORMATTED_OUTPUT
#include <iostream>
#include <ostream>
#include "boost/format.hpp"
#endif

#ifdef EVENT_PRODUCER
#include "EventProducer.hh"
#endif

// namespace for EEC code
namespace eec {

// naming shortcuts
namespace bh = boost::histogram;
typedef unsigned int uint;

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------

const uint factorials[12] = {1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800};
const double reg = 1e-50;
const double PI = 3.14159265358979323846;
const double TWOPI = 6.28318530717958647693;

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

// multinomial factor on sorted indices
uint multinomial(const uint n, const uint * inds) noexcept {

  uint denom(1), count(1);
  for (uint i = 1; i < n; i++) {
    if (inds[i] == inds[i-1]) count++;
    else {
      denom *= factorials[count];
      count = 1;
    }
  }
  denom *= factorials[count];

  return factorials[n]/denom;
}

// gets bin centers from an axis
template<class Axis>
inline std::vector<double> get_bin_centers(const Axis & axis) {
  std::vector<double> bin_centers_vec(axis.size());
  for (int i = 0; i < axis.size(); i++)
    bin_centers_vec[i] = axis.bin(i).center();
  return bin_centers_vec;
}

// gets bin edges from an axis
template<class Axis>
inline std::vector<double> get_bin_edges(const Axis & axis) {
  if (axis.size() == 0) return std::vector<double>();

  std::vector<double> bins_vec(axis.size() + 1);
  bins_vec[0] = axis.bin(0).lower();
  for (int i = 0; i < axis.size(); i++)
    bins_vec[i+1] = axis.bin(i).upper();
  return bins_vec;
}

#ifdef FORMATTED_OUTPUT
template<class Axis>
inline void output_axis(std::ostream & os, const Axis & axis) {
  for (int i = 0; i < axis.size(); i++) {
    const auto & bin(axis.bin(i));
    os << boost::format("%i : %.16g %.16g %.16g\n") % i % bin.lower() % bin.center() % bin.upper();
  }
  os << '\n';
}

template<class Hist>
inline void output_1d_hist(std::ostream & os, const Hist & hist) {
  for (const auto & x : bh::indexed(hist, bh::coverage::all)) {
    const auto & wsum(*x);
    os << boost::format("%u : %.16g %.16g\n") % x.index(0) % wsum.value() % sqrt(wsum.variance());
  }
  os << '\n';
}

template<class Hist>
inline void output_3d_hist(std::ostream & os, const Hist & hist) {
  for (const auto & x : bh::indexed(hist, bh::coverage::all)) {
    const auto & wsum(*x);
    os << boost::format("%u %u %u : %.16g %.16g\n") % x.index(0) % x.index(1) % x.index(2) % wsum.value() % sqrt(wsum.variance());
  }
  os << '\n';
}
#endif // FORMATTED_OUTPUT

// base class for EEC Computations
class EECComputation {
protected:

  // for storing pairwise distances
  std::vector<double> dists_;

  // the pts of the current event
  std::vector<double> pts_;

  // the current weight of the event
  double weight_;

  // the current multiplicity of the event
  size_t mult_;

  // methods that will be overloaded by specific subclasses
  virtual void compute_eec() = 0;
  virtual size_t get_hist_size(bool) const = 0;
  virtual void get_hist(double *, double *, size_t, bool) const = 0;

  void store_ptyphis(const double * ptyphis, size_t mult, double weight) {

    // store weight and number of particles internally for this event
    weight_ = weight;
    mult_ = mult;

    // compute pairwise distances and extract pts
    dists_.resize(mult_*mult_);
    pts_.resize(mult_);
    for (size_t i = 0; i < mult_; i++) {

      // store pt
      pts_[i] = ptyphis[3*i];

      // zero out diagonal
      dists_[i * mult_ + i] = 0;

      double y_i(ptyphis[3*i+1]), phi_i(ptyphis[3*i+2]);
      for (size_t j = 0; j < i; j++) {
        double ydiff(y_i - ptyphis[3*j+1]), phidiffabs(std::fabs(phi_i - ptyphis[3*j+2]));

        // ensure that the phi difference is properly handled
        if (phidiffabs > PI) phidiffabs -= TWOPI;
        dists_[i * mult_ + j] = dists_[j * mult_ + i] = sqrt(ydiff*ydiff + phidiffabs*phidiffabs);
      }
    }
  }

public:

  EECComputation() {}
  virtual ~EECComputation() {}

  // fastjet support
#ifdef __FASTJET_PSEUDOJET_HH__
  void compute(const std::vector<fastjet::PseudoJet> & pjs, const double weight = 1.0) {

    weight_ = weight;
    mult_ = pjs.size()

    for (size_t i = 0; i < mult_; i++) {
      pts_[i] = pjs[i].pt();
      dists_[i * mult_ + i] = 0;

      for (size_t j = 0; j < i; j++)
        dists_[i * mult_ + j] = dists_[j * mult_ + i] = pjs[i].delta_R(pjs[j]);
    }

    // delegate EEC computation to subclass
    compute_eec();
  }
#endif // __FASTJET_PSEUDOJET_HH__

  // compute on a vector of ptyphis
  void compute(const std::vector<double> & ptyphis, const double weight = 1.0) { compute(ptyphis.data(), ptyphis.size(), weight); }

  // compute on a C array of ptyphis
  void compute(const double * ptyphis, size_t mult, const double weight = 1.0) {

    store_ptyphis(ptyphis, mult, weight);    

    // delegate EEC computation to subclass
    compute_eec();
  }

#ifdef EVENT_PRODUCER
  // compute on an event producer
  void compute(EventProducer & evp) {

    while (evp.next()) {
      store_ptyphis(evp.ptyphis(), evp.mult(), evp.weight());
      compute_eec();
    }
  }
#endif // EVENT_PRODUCER

  // get histogram and errors
  std::pair<std::vector<double>, std::vector<double>> get_hist(bool include_overflows = true) {
    size_t hist_size(get_hist_size(include_overflows));
    std::vector<double> hist_vals(hist_size), hist_errs(hist_size);
    get_hist(hist_vals.data(), hist_errs.data(), hist_size, include_overflows);
    return std::make_pair(hist_vals, hist_errs);
  }

};

// triangular ope computation
template<class Transform0 = bh::axis::transform::id,
         class Transform1 = bh::axis::transform::id,
         class Transform2 = bh::axis::transform::id>
class EECTriangleOPE : public EECComputation {
public:

  // axes
  typedef bh::axis::regular<double, Transform0> axis0_t;
  typedef bh::axis::regular<double, Transform1> axis1_t;
  typedef bh::axis::regular<double, Transform2> axis2_t;
  axis0_t axis0;
  axis1_t axis1;
  axis2_t axis2;

  // histogram
  typedef decltype(bh::make_weighted_histogram(axis0, axis1, axis2)) hist_t;
  hist_t hist;

  EECTriangleOPE() {}
  EECTriangleOPE(uint nbins0, double axis0_min, double axis0_max,
                 uint nbins1, double axis1_min, double axis1_max,
                 uint nbins2, double axis2_min, double axis2_max) :
    axis0(nbins0, axis0_min, axis0_max),
    axis1(nbins1, axis1_min, axis1_max),
    axis2(nbins2, axis2_min, axis2_max),
    hist(bh::make_weighted_histogram(axis0, axis1, axis2))
  {}

  std::vector<double> bin_centers(int i) const {
    if (i == 0) return get_bin_centers(axis0);
    else if (i == 1) return get_bin_centers(axis1);
    else if (i == 2) return get_bin_centers(axis2);
    else throw std::invalid_argument("axis index i must be 0, 1, or 2");
    return std::vector<double>();
  }

  std::vector<double> bin_edges(int i) const {
    if (i == 0) return get_bin_edges(axis0);
    else if (i == 1) return get_bin_edges(axis1);
    else if (i == 2) return get_bin_edges(axis2);
    else throw std::invalid_argument("axis index i must be 0, 1, or 2");
    return std::vector<double>();
  }

#ifdef FORMATTED_OUTPUT
  void output(std::ostream & os = std::cout) const {
    os << "axis0\n";
    output_axis(os, axis0);

    os << "axis1\n";
    output_axis(os, axis1);

    os << "axis2\n";
    output_axis(os, axis2);

    os << "hist-3d\n";
    output_3d_hist(os, hist);
  }
#endif // FORMATTED_OUTPUT

  size_t get_hist_size(bool include_overflows = true) const {
    ssize_t start(include_overflows ? -1 : 0), end_offset(include_overflows ? 1 : 0), 
            diff(end_offset - start),
            size((axis0.size() + diff)*(axis1.size() + diff)*(axis2.size() + diff));
    return size;
  }

  void get_hist(double * hist_vals, double * hist_errs, size_t size, bool include_overflows = true) const {

    if (size != get_hist_size(include_overflows)) 
      throw std::invalid_argument("Size of histogram doesn't match provided arrays");

    ssize_t start(include_overflows ? -1 : 0), end_off(include_overflows ? 1 : 0), a(0);
    for (int i = start; i < axis0.size() + end_off; i++) {
      for (int j = start; j < axis1.size() + end_off; j++) {
        for (int k = start; k < axis2.size() + end_off; k++, a++) {
          const auto & x(hist.at(i, j, k));
          hist_vals[a] = x.value();
          hist_errs[a] = std::sqrt(x.variance());
        }
      }
    }
  }

private:

  void compute_eec() {

    // loop over pairs of particles
    for (uint i = 0; i < mult_; i++) {
      double weight_i(weight_ * pts_[i]);
      uint ixn(i*mult_);

      for (uint j = 0; j <= i; j++) {
        double weight_ij(weight_i * pts_[j]);
        double dist_ij(dists_[ixn + j]);
        uint jxn(j*mult_);
        bool ij_match(i == j);

        for (uint k = 0; k <= j; k++) {
          double weight_ijk(weight_ij * pts_[k]);
          double dist_ik(dists_[ixn + k]), dist_jk(dists_[jxn + k]);
          bool ik_match(i == k), jk_match(j == k);
          int sym(!(ij_match || ik_match || jk_match) ? 6 : (ij_match && ik_match ? 1 : 3));

          // determine distances (checked this very carefully)
          double xL(dist_jk), xM(dist_ik), xS(dist_ij);
          if (dist_ij > dist_jk) {
            if (dist_ij > dist_ik) {
              xL = dist_ij;
              if (dist_jk > dist_ik) {
                xM = dist_jk;
                xS = dist_ik;
              }
              else xS = dist_jk;
            }
            else {
              xL = dist_ik;
              xM = dist_ij;
              xS = dist_jk;
            }
          }
          else if (dist_ik > dist_jk) {
            xL = dist_ik;
            xM = dist_jk;
            xS = dist_ij;
          }
          else if (dist_ij > dist_ik) {
            xM = dist_ij;
            xS = dist_ik;
          }

          // define coordinate mapping
          double xi(xS/(xM + reg)), 
                 diff(xL - xM),
                 phi(asin(sqrt(fabs(1 - diff*diff/(xS*xS + reg)))));

          // fill histogram
          hist(bh::weight(sym * weight_ijk), xL, xi, phi);
        }
      }
    }
  }
};

// longest side computation
template<class Transform = bh::axis::transform::id>
class EECLongestSide : public EECComputation {
private:

  uint N_;
  void (EECLongestSide::*compute_eec_ptr_)();

  void compute_eec() { (this->*compute_eec_ptr_)(); }

public:

  // axis
  typedef bh::axis::regular<double, Transform> axis0_t;
  axis0_t axis0;

  // histogram
  typedef decltype(bh::make_weighted_histogram(axis0)) hist_t;
  hist_t hist;

  EECLongestSide() {}
  EECLongestSide(uint N, uint nbins, double axis_min, double axis_max) :
    N_(N),
    axis0(nbins, axis_min, axis_max),
    hist(bh::make_weighted_histogram(axis0))
  {
    if (N_ == 2) compute_eec_ptr_ = &EECLongestSide::eec;
    else if (N_ == 3) compute_eec_ptr_ = &EECLongestSide::eeec;
    else if (N_ == 4) compute_eec_ptr_ = &EECLongestSide::eeeec;
    else if (N_ == 5) compute_eec_ptr_ = &EECLongestSide::eeeeec;
    else throw std::invalid_argument("N must be 2, 3, 4, or 5");
  }

  std::vector<double> bin_centers() const { return get_bin_centers(axis0); }
  std::vector<double> bin_edges() const { return get_bin_edges(axis0); }

#ifdef FORMATTED_OUTPUT
  void output(std::ostream & os = std::cout) const {
    os << "axis0\n";
    output_axis(os, axis0);

    os << "hist-1d\n";
    output_1d_hist(os, hist);
  }
#endif // FORMATTED_OUTPUT

  size_t get_hist_size(bool include_overflows = true) const {
    ssize_t start(include_overflows ? -1 : 0), end_off(include_overflows ? 1 : 0),
            size(axis0.size() + end_off - start);
    return size;
  }

  void get_hist(double * hist_vals, double * hist_errs, size_t size, bool include_overflows = true) const {

    if (size != get_hist_size(include_overflows)) 
      throw std::invalid_argument("Size of histogram doesn't match provided arrays");

    uint a(0);
    for (int i = (include_overflows ? -1 : 0); i < axis0.size() + (include_overflows ? 1 : 0); i++, a++) {
      const auto & x(hist.at(i));
      hist_vals[a] = x.value();
      hist_errs[a] = sqrt(x.variance());
    }
  }

private:

  void eec() {

    // loop over pairs of particles
    for (uint i = 0; i < mult_; i++) {
      double weight_i(weight_ * pts_[i]);
      uint ixn(i*mult_);

      for (uint j = 0; j <= i; j++) {
        int sym(i == j ? 1 : 2);

        // fill histogram
        hist(bh::weight(sym * weight_i * pts_[j]), dists_[ixn + j]);
      }
    }
  }

  void eeec() {

    // loop over pairs of particles
    for (uint i = 0; i < mult_; i++) {
      double weight_i(weight_ * pts_[i]);
      uint ixn(i*mult_);

      for (uint j = 0; j <= i; j++) {
        double weight_ij(weight_i * pts_[j]);
        double dist_ij(dists_[ixn + j]);
        uint jxn(j*mult_);
        bool ij_match(i == j);

        for (uint k = 0; k <= j; k++) {
          double weight_ijk(weight_ij * pts_[k]);
          double dist_ik(dists_[ixn + k]), dist_jk(dists_[jxn + k]);
          bool ik_match(i == k), jk_match(j == k);
          int sym(!(ij_match || ik_match || jk_match) ? 6 : (ij_match && ik_match ? 1 : 3));

          // determine maximum distance
          double max_dist(dist_jk);
          if (dist_ij > dist_jk) {
            if (dist_ij > dist_ik) max_dist = dist_ij;
            else max_dist = dist_ik;
          }
          else if (dist_ik > dist_jk) max_dist = dist_ik;

          // fill histogram
          hist(bh::weight(sym * weight_ijk), max_dist);
        }
      }
    }
  }

  void eeeec() {

    // loop over pairs of particles
    uint inds[4];
    double dists_ijkl[6];
    for (uint i = 0; i < mult_; i++) {
      inds[0] = i;
      double weight_i(weight_ * pts_[i]);
      uint ixn(i*mult_);

      for (uint j = 0; j <= i; j++) {
        inds[1] = j;
        double weight_ij(weight_i * pts_[j]);
        dists_ijkl[0] = dists_[ixn + j];
        uint jxn(j*mult_);

        for (uint k = 0; k <= j; k++) {
          inds[2] = k;
          double weight_ijk(weight_ij * pts_[k]);
          dists_ijkl[1] = dists_[ixn + k];
          dists_ijkl[2] = dists_[jxn + k];
          uint kxn(k*mult_);

          for (uint l = 0; l <= k; l++) {
            inds[3] = l;
            double weight_ijkl(weight_ijk * pts_[l]);
            dists_ijkl[3] = dists_[ixn + l];
            dists_ijkl[4] = dists_[jxn + l];
            dists_ijkl[5] = dists_[kxn + l];

            // get symmetry factor
            uint sym(multinomial(4, inds));

            // get maximum distance
            double max_dist(0);
            for (uint m = 0; m < 6; m++) 
              if (dists_ijkl[m] > max_dist) 
                max_dist = dists_ijkl[m];

            // fill histogram
            hist(bh::weight(sym * weight_ijkl), max_dist);
          }
        }
      }
    }
  }

  void eeeeec() {

    // loop over pairs of particles
    uint inds[5];
    double dists_ijklm[10];
    for (uint i = 0; i < mult_; i++) {
      inds[0] = i;
      double weight_i(weight_ * pts_[i]);
      uint ixn(i*mult_);

      for (uint j = 0; j <= i; j++) {
        inds[1] = j;
        double weight_ij(weight_i * pts_[j]);
        dists_ijklm[0] = dists_[ixn + j];
        uint jxn(j*mult_);

        for (uint k = 0; k <= j; k++) {
          inds[2] = k;
          double weight_ijk(weight_ij * pts_[k]);
          dists_ijklm[1] = dists_[ixn + k];
          dists_ijklm[2] = dists_[jxn + k];
          uint kxn(k*mult_);

          for (uint l = 0; l <= k; l++) {
            inds[3] = l;
            double weight_ijkl(weight_ijk * pts_[l]);
            dists_ijklm[3] = dists_[ixn + l];
            dists_ijklm[4] = dists_[jxn + l];
            dists_ijklm[5] = dists_[kxn + l];
            uint lxn(l*mult_);

            for (uint m = 0; m <= l; m++) {
              inds[4] = m;
              double weight_ijklm(weight_ijkl * pts_[m]);
              dists_ijklm[6] = dists_[ixn + m];
              dists_ijklm[7] = dists_[jxn + m];
              dists_ijklm[8] = dists_[kxn + m];
              dists_ijklm[9] = dists_[lxn + m];

              // get symmetry factor
              uint sym(multinomial(5, inds));

              // get maximum distances
              double max_dist(0);
              for (uint m = 0; m < 10; m++) 
                if (dists_ijklm[m] > max_dist) 
                  max_dist = dists_ijklm[m];

              // fill histogram
              hist(bh::weight(sym * weight_ijklm), max_dist);
            }
          }
        }
      }
    }
  }
};

// some typedefs for the classes we just declared
typedef EECTriangleOPE<> EECTriangleOPE_id_id_id;
typedef EECTriangleOPE<bh::axis::transform::log> EECTriangleOPE_log_id_id;
typedef EECTriangleOPE<bh::axis::transform::id, bh::axis::transform::log> EECTriangleOPE_id_log_id;
typedef EECTriangleOPE<bh::axis::transform::log, bh::axis::transform::log> EECTriangleOPE_log_log_id;
typedef EECLongestSide<> EECLongestSide_id;
typedef EECLongestSide<bh::axis::transform::log> EECLongestSide_log;

} // namespace eec

#endif // EEC_COMPUTATIONS_HH
