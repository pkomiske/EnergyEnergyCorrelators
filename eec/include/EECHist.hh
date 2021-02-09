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
//  _    _ _____  _____ _______ 
// | |  | |_   _|/ ____|__   __|
// | |__| | | | | (___    | |   
// |  __  | | |  \___ \   | |   
// | |  | |_| |_ ____) |  | |   
// |_|  |_|_____|_____/   |_|   

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

#include "EECUtils.hh"

namespace eec {

#ifndef SWIG_PREPROCESSOR
  // naming shortcuts
  namespace bh = boost::histogram;
#endif

//-----------------------------------------------------------------------------
// boost::histogram::axis::transform using statements
//-----------------------------------------------------------------------------

namespace axis {

using id = boost::histogram::axis::transform::id;
using log = boost::histogram::axis::transform::log;

}

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

// tally a specific histogram across threads (assumed to be the first axis)
template<class Hist>
Hist combine_hists(const std::vector<std::vector<Hist>> & hists, unsigned hist_i) {

  // do the combining
  Hist hist;
  for (unsigned thread_i = 0; thread_i < hists.size(); thread_i++)
    hist += hists[thread_i][hist_i];

  return hist;
}

#ifdef EEC_HIST_FORMATTED_OUTPUT
template<class Axis>
void output_axis(std::ostream & os, const Axis & axis, int nspaces=0) {
  for (int i = 0; i < axis.size(); i++) {
    const auto & bin(axis.bin(i));
    os << std::string(nspaces, ' ')
       << boost::format("%i: %.16g %.16g %.16g\n") % i % bin.lower() % bin.center() % bin.upper();
  }
  os << '\n';
}
template<class Hist>
void output_1d_hist(std::ostream & os, const Hist & hist, int nspaces=0) {
  for (auto && x : bh::indexed(hist, bh::coverage::all)) {
    const auto & wsum(*x);
    os << std::string(nspaces, ' ')
       << boost::format("%u: %.16g %.16g\n") % x.index(0) % wsum.value() % sqrt(wsum.variance());
  }
  os << '\n';
}
template<class Hist>
void output_3d_hist(std::ostream & os, const Hist & hist, int nspaces=0) {
  for (auto && x : bh::indexed(hist, bh::coverage::all)) {
    const auto & wsum(*x);
    os << std::string(nspaces, ' ')
       << boost::format("%u %u %u: %.16g %.16g\n") % x.index(0) % x.index(1) % x.index(2) % wsum.value() % sqrt(wsum.variance());
  }
  os << '\n';
}
#endif // EEC_HIST_FORMATTED_OUTPUT

//-----------------------------------------------------------------------------
// Base class for histograms
//-----------------------------------------------------------------------------

class HistBase {
public:

  HistBase() : HistBase(1) {}
  HistBase(int num_threads) : num_threads_(determine_num_threads(num_threads)) {}

  int num_threads() const { return num_threads_; }

  virtual size_t get_hist_size(bool) const = 0;
  virtual void get_hist(double *, double *, size_t, bool, unsigned) const = 0;

  // return histogram and errors as a pair of vectors
  std::pair<std::vector<double>, std::vector<double>> get_hist_errs(bool include_overflows = true, unsigned hist_i = 0) {
    size_t hist_size(get_hist_size(include_overflows));
    std::vector<double> hist_vals(hist_size), hist_errs(hist_size);
    get_hist(hist_vals.data(), hist_errs.data(), hist_size, include_overflows, hist_i);
    return std::make_pair(std::move(hist_vals), std::move(hist_errs));
  }

private:

  int num_threads_;

}; // HistBase

//-----------------------------------------------------------------------------
// 1D histogram class
//-----------------------------------------------------------------------------

template<class T>
class Hist1D : public HistBase {
public:

#ifndef SWIG_PREPROCESSOR
  // typedefs
  typedef T Transform;
  typedef bh::axis::regular<double, Transform> Axis;
private:
  Axis axis_;
public:
  typedef decltype(bh::make_weighted_histogram(axis_)) Hist;
private:
  std::vector<std::vector<Hist>> hists_;
#endif

public:

  Hist1D() : Hist1D(0, 0, 0) {}
  Hist1D(unsigned nbins, double axis_min, double axis_max, int num_threads = 1) :
    HistBase(num_threads),
    axis_(nbins, axis_min, axis_max),
    hists_(this->num_threads(), {bh::make_weighted_histogram(axis())})
  {}

  virtual ~Hist1D() {}

  void duplicate_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    int nnewhists(int(nhists) - int(this->nhists()));
    if (nnewhists > 0) {
      for (int i = 0; i < num_threads(); i++)
        hists(i).insert(hists(i).end(), nnewhists, bh::make_weighted_histogram(axis()));
    }
  }
  
#ifndef SWIG_PREPROCESSOR
  // access functions for axis and histogram
  const Axis & axis() const { return axis_; }
  std::vector<Hist> & hists(int thread_i = 0) { return hists_[thread_i]; }
  const std::vector<Hist> & hists(int thread_i = 0) const { return hists_[thread_i]; }

  // combined histograms
  Hist combined_hist(unsigned hist_i) const { return combine_hists(hists_, hist_i); }
#endif

  unsigned nhists() const { return hists().size(); }

  // bin information
  std::vector<double> bin_centers() const { return get_bin_centers(axis()); }
  std::vector<double> bin_edges() const { return get_bin_edges(axis()); }

#ifdef EEC_HIST_FORMATTED_OUTPUT
  void output(std::ostream & os = std::cout, int nspaces = 0) const {
    os << "axis0\n";
    output_axis(os, axis(), nspaces);

    for (unsigned i = 0; i < nhists(); i++) {
      os << "hist-1d " << i << '\n';
      output_1d_hist(os, combined_hist(i), nspaces);
    }
  }
#endif

  size_t get_hist_size(bool include_overflows = true) const {
    auto size(axis().size());
    if (include_overflows)
      size += 2;
    return size;
  }

  void get_hist(double * hist_vals, double * hist_errs, size_t size, bool include_overflows = true, unsigned hist_i = 0) const {

    if (size != get_hist_size(include_overflows)) 
      throw std::invalid_argument("Size of histogram doesn't match provided arrays");
    if (hist_i >= nhists())
      throw std::out_of_range("Requested histogram out of range");

    Hist hist(combined_hist(hist_i));
    unsigned a(0);
    for (int i = (include_overflows ? -1 : 0), end = axis().size() + (include_overflows ? 1 : 0); i < end; i++, a++) {
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

#ifndef SWIG_PREPROCESSOR
  // typedefs
  typedef T0 Transform0;
  typedef T1 Transform1;
  typedef T2 Transform2;
  typedef bh::axis::regular<double, Transform0> Axis0;
  typedef bh::axis::regular<double, Transform1> Axis1;
  typedef bh::axis::regular<double, Transform2> Axis2;
private:
  Axis0 axis0_;
  Axis1 axis1_;
  Axis2 axis2_;
public:
  typedef decltype(bh::make_weighted_histogram(axis0_, axis1_, axis2_)) Hist;
private:
  std::vector<std::vector<Hist>> hists_;
#endif

public:

  Hist3D(unsigned nbins0, double axis0_min, double axis0_max,
         unsigned nbins1, double axis1_min, double axis1_max,
         unsigned nbins2, double axis2_min, double axis2_max,
         int num_threads = 1) :
    HistBase(num_threads),
    axis0_(nbins0, axis0_min, axis0_max),
    axis1_(nbins1, axis1_min, axis1_max),
    axis2_(nbins2, axis2_min, axis2_max),
    hists_(this->num_threads(), {bh::make_weighted_histogram(axis0(), axis1(), axis2())})
  {}

  virtual ~Hist3D() {}

  void duplicate_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    int nnewhists(int(nhists) - int(this->nhists()));
    if (nnewhists > 0) {
      for (int i = 0; i < num_threads(); i++)
        hists(i).insert(hists(i).end(), nnewhists, bh::make_weighted_histogram(axis0(), axis1(), axis2()));
    }
  }

#ifndef SWIG_PREPROCESSOR
  // access functions for axis and histogram
  const Axis0 & axis0() const { return axis0_; }
  const Axis1 & axis1() const { return axis1_; }
  const Axis2 & axis2() const { return axis2_; }
  std::vector<Hist> & hists(int thread_i = 0) { return hists_[thread_i]; }
  const std::vector<Hist> & hists(int thread_i = 0) const { return hists_[thread_i]; }

  // combined histograms
  Hist combined_hist(unsigned hist_i) const { return combine_hists(hists_, hist_i); }
#endif

  unsigned nhists() const { return hists().size(); }

  std::vector<double> bin_centers(int i) const {
    if (i == 0) return get_bin_centers(axis0());
    else if (i == 1) return get_bin_centers(axis1());
    else if (i == 2) return get_bin_centers(axis2());
    else throw std::invalid_argument("axis index i must be 0, 1, or 2");
    return {};
  }

  std::vector<double> bin_edges(int i) const {
    if (i == 0) return get_bin_edges(axis0());
    else if (i == 1) return get_bin_edges(axis1());
    else if (i == 2) return get_bin_edges(axis2());
    else throw std::invalid_argument("axis index i must be 0, 1, or 2");
    return {};
  }

#ifdef EEC_HIST_FORMATTED_OUTPUT
  void output(std::ostream & os = std::cout, int nspaces = 0) const {
    os << "axis0\n";
    output_axis(os, axis0(), nspaces);

    os << "axis1\n";
    output_axis(os, axis1(), nspaces);

    os << "axis2\n";
    output_axis(os, axis2(), nspaces);

    for (unsigned i = 0; i < nhists(); i++) {
      os << "hist-3d " << i << '\n';
      output_3d_hist(os, combined_hist(i), nspaces);
    }
  }
#endif

  size_t get_hist_size(bool include_overflows = true) const {
    size_t diff(include_overflows ? 2 : 0),
           size((axis0().size() + diff)*(axis1().size() + diff)*(axis2().size() + diff));
    return size;
  }

  void get_hist(double * hist_vals, double * hist_errs, size_t size, bool include_overflows = true, unsigned hist_i = 0) const {

    if (size != get_hist_size(include_overflows)) 
      throw std::invalid_argument("Size of histogram doesn't match provided arrays");
    if (hist_i >= nhists())
      throw std::out_of_range("Requested histogram out of range");

    Hist hist(combined_hist(hist_i));
    int start(include_overflows ? -1 : 0), end_off(include_overflows ? 1 : 0);
    size_t a(0);
    for (int i = start; i < axis0().size() + end_off; i++) {
      for (int j = start; j < axis1().size() + end_off; j++) {
        for (int k = start; k < axis2().size() + end_off; k++, a++) {
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