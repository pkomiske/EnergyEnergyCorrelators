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

#ifndef EEC_HIST_HH
#define EEC_HIST_HH

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "boost/histogram.hpp"

#ifdef EEC_HIST_FORMATTED_OUTPUT
#include <iostream>
#include "boost/format.hpp"
#endif

namespace eec {

// naming shortcuts
namespace bh = boost::histogram;

//-----------------------------------------------------------------------------
// Histogram helper functions
//-----------------------------------------------------------------------------

// gets bin centers from an axis
template<class Axis>
std::vector<double> get_bin_centers(const Axis & axis) {
  std::vector<double> bin_centers_vec(axis.size());
  for (int i = 0; i < axis.size(); i++)
    bin_centers_vec[i] = axis.bin(i).center();
  return bin_centers_vec;
}

// gets bin edges from an axis
template<class Axis>
std::vector<double> get_bin_edges(const Axis & axis) {
  if (axis.size() == 0) return std::vector<double>();

  std::vector<double> bins_vec(axis.size() + 1);
  bins_vec[0] = axis.bin(0).lower();
  for (int i = 0; i < axis.size(); i++)
    bins_vec[i+1] = axis.bin(i).upper();
  return bins_vec;
}

#ifdef EEC_HIST_FORMATTED_OUTPUT
template<class Axis>
void output_axis(std::ostream & os, const Axis & axis) {
  for (int i = 0; i < axis.size(); i++) {
    const auto & bin(axis.bin(i));
    os << boost::format("%i : %.16g %.16g %.16g\n") % i % bin.lower() % bin.center() % bin.upper();
  }
  os << '\n';
}
template<class Hist>
void output_1d_hist(std::ostream & os, const Hist & hist) {
  for (const auto & x : bh::indexed(hist, bh::coverage::all)) {
    const auto & wsum(*x);
    os << boost::format("%u : %.16g %.16g\n") % x.index(0) % wsum.value() % sqrt(wsum.variance());
  }
  os << '\n';
}
template<class Hist>
void output_3d_hist(std::ostream & os, const Hist & hist) {
  for (const auto & x : bh::indexed(hist, bh::coverage::all)) {
    const auto & wsum(*x);
    os << boost::format("%u %u %u : %.16g %.16g\n") % x.index(0) % x.index(1) % x.index(2) % wsum.value() % sqrt(wsum.variance());
  }
  os << '\n';
}
#endif // EEC_HIST_FORMATTED_OUTPUT

//-----------------------------------------------------------------------------
// Base class for histograms
//-----------------------------------------------------------------------------

class HistBase {
public:

  virtual std::size_t get_hist_size(bool) const = 0;
  virtual void get_hist(double *, double *, std::size_t, bool, unsigned) const = 0;

  // get histogram and errors
  std::pair<std::vector<double>, std::vector<double>> get_hist(bool include_overflows = true, unsigned hist_i = 0) {
    std::size_t hist_size(get_hist_size(include_overflows));
    std::vector<double> hist_vals(hist_size), hist_errs(hist_size);
    get_hist(hist_vals.data(), hist_errs.data(), hist_size, hist_i, include_overflows);
    return std::make_pair(std::move(hist_vals), std::move(hist_errs));
  }

}; // HistBase

//-----------------------------------------------------------------------------
// 1D histogram class
//-----------------------------------------------------------------------------

template<class T>
class Hist1D : public HistBase {
public:

  // axis
  typedef T Transform;
  typedef bh::axis::regular<double, Transform> Axis;
  Axis axis;

  // histogram
  typedef decltype(bh::make_weighted_histogram(axis)) Hist;
  std::vector<Hist> hists;

  Hist1D(unsigned nbins, double axis_min, double axis_max) :
    axis(nbins, axis_min, axis_max),
    hists(1, bh::make_weighted_histogram(axis))
  {}
  virtual ~Hist1D() {}

  void duplicate_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    if (nhists > 1)
      hists.insert(hists.end(), nhists - 1, bh::make_weighted_histogram(axis));
  }

  std::size_t nhists() const { return hists.size(); }

  std::vector<double> bin_centers() const { return get_bin_centers(axis); }
  std::vector<double> bin_edges() const { return get_bin_edges(axis); }

#ifdef EEC_HIST_FORMATTED_OUTPUT
  void output(std::ostream & os = std::cout, unsigned hist_i = 0) const {
    os << "axis0\n";
    output_axis(os, axis);

    os << "hist-1d\n";
    output_1d_hist(os, hists[0]);
  }
#endif

  std::size_t get_hist_size(bool include_overflows = true) const {
    std::ptrdiff_t start(include_overflows ? -1 : 0), end_off(include_overflows ? 1 : 0),
                   size(axis.size() + end_off - start);
    return size;
  }

  void get_hist(double * hist_vals, double * hist_errs, std::size_t size, bool include_overflows = true, unsigned hist_i = 0) const {

    if (size != get_hist_size(include_overflows)) 
      throw std::invalid_argument("Size of histogram doesn't match provided arrays");

    const Hist & hist(hists[0]);
    unsigned a(0);
    for (int i = (include_overflows ? -1 : 0); i < axis.size() + (include_overflows ? 1 : 0); i++, a++) {
      const auto & x(hist.at(i));
      hist_vals[a] = x.value();
      hist_errs[a] = std::sqrt(x.variance());
    }
  }

}; // Hist1D

//-----------------------------------------------------------------------------
// 3D histogram class
//-----------------------------------------------------------------------------

template<class T0, class T1, class T2>
class Hist3D : public HistBase {
public:

  // typedefs of transform types
  typedef T0 Transform0;
  typedef T1 Transform1;
  typedef T2 Transform2;

  // axis
  typedef bh::axis::regular<double, Transform0> Axis0;
  typedef bh::axis::regular<double, Transform1> Axis1;
  typedef bh::axis::regular<double, Transform2> Axis2;

  Axis0 axis0;
  Axis1 axis1;
  Axis2 axis2;

  // histogram
  typedef decltype(bh::make_weighted_histogram(axis0, axis1, axis2)) Hist;
  std::vector<Hist> hists;

  Hist3D(unsigned nbins0, double axis0_min, double axis0_max,
         unsigned nbins1, double axis1_min, double axis1_max,
         unsigned nbins2, double axis2_min, double axis2_max) :
    axis0(nbins0, axis0_min, axis0_max),
    axis1(nbins1, axis1_min, axis1_max),
    axis2(nbins2, axis2_min, axis2_max),
    hists(1, bh::make_weighted_histogram(axis0, axis1, axis2))
  {}
  virtual ~Hist3D() {}

  void duplicate_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    if (nhists > 1)
      hists.insert(hists.end(), nhists - 1, bh::make_weighted_histogram(axis0, axis1, axis2));
  }

  std::size_t nhists() const { return hists.size(); }

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

#ifdef EEC_HIST_FORMATTED_OUTPUT
  void output(std::ostream & os = std::cout, unsigned hist_i = 0) const {
    os << "axis0\n";
    output_axis(os, axis0);

    os << "axis1\n";
    output_axis(os, axis1);

    os << "axis2\n";
    output_axis(os, axis2);

    os << "hist-3d\n";
    output_3d_hist(os, hists[hist_i]);
  }
#endif

  std::size_t get_hist_size(bool include_overflows = true) const {
    std::ptrdiff_t start(include_overflows ? -1 : 0), end_offset(include_overflows ? 1 : 0), 
                   diff(end_offset - start),
                   size((axis0.size() + diff)*(axis1.size() + diff)*(axis2.size() + diff));
    return size;
  }

  void get_hist(double * hist_vals, double * hist_errs, std::size_t size, bool include_overflows = true, unsigned hist_i = 0) const {

    if (size != get_hist_size(include_overflows)) 
      throw std::invalid_argument("Size of histogram doesn't match provided arrays");
    if (hist_i >= hists.size())
      throw std::out_of_range("Requested histogram out of range");

    const Hist & hist(hists[hist_i]);
    int start(include_overflows ? -1 : 0), end_off(include_overflows ? 1 : 0);
    std::size_t a(0);
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

}; // Hist3D

} // namespace eec

#endif // EEC_HIST_HH