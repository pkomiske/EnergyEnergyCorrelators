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

/*   ______ ______ _____ 
 *  |  ____|  ____/ ____|
 *  | |__  | |__ | |     
 *  |  __| |  __|| |     
 *  | |____| |___| |____ 
 *  |______|______\_____|
 *   _    _ _____  _____ _______ 
 *  | |  | |_   _|/ ____|__   __|
 *  | |__| | | | | (___    | |   
 *  |  __  | | |  \___ \   | |   
 *  | |  | |_| |_ ____) |  | |   
 *  |_|  |_|_____|_____/   |_|   
 */

#ifndef EEC_HIST_HH
#define EEC_HIST_HH

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "boost/histogram.hpp"

#include "EECUtils.hh"

namespace eec {
namespace hist {

#ifndef SWIG_PREPROCESSOR
  // naming shortcuts
  namespace bh = boost::histogram;
  using bh::weight;
#endif

//-----------------------------------------------------------------------------
// boost::histogram::axis::transform using statements
//-----------------------------------------------------------------------------

namespace axis {

using id = bh::axis::transform::id;
using log = bh::axis::transform::log;

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

template<class T>
std::string name_transform() {
  if (std::is_base_of<bh::axis::transform::log, T>::value)
    return "log";
  else if (std::is_base_of<bh::axis::transform::id, T>::value)
    return "id";
  else if (std::is_base_of<bh::axis::transform::sqrt, T>::value)
    return "sqrt";
  else if (std::is_base_of<bh::axis::transform::pow, T>::value)
    return "pow";
  else
    return "unknown";
}

template<class Axis>
void output_axis(std::ostream & os, const Axis & axis, int precision = 16) {
  os.precision(precision);
  os << "# " << name_transform<Axis>() << " axis, "
     << axis.size() << " bins, (" << axis.value(0) << ", " << axis.value(axis.size()) << ")\n";
  os << "bin_edges :";
  for (double edge : get_bin_edges(axis))
    os << ' ' << edge;
  os << '\n'
     << "bin_centers :";
  for (double center : get_bin_centers(axis))
    os << ' ' << center;
  os << "\n\n";
}

template<class Hist>
void output_hist(std::ostream & os, const Hist & hist, int precision = 16,
                 bool include_overflows = true, int hist_i = -1) {
  os.precision(precision);
  os << "# hist";
  if (hist_i != -1) os << ' ' << hist_i;
  os << ", rank " << hist.rank() << ", " << hist.size() << " total bins\n"
     << "# bin_multi_index : bin_value bin_variance\n";
  for (auto && x : bh::indexed(hist, include_overflows ? bh::coverage::all : bh::coverage::inner)) {
    for (int index : x.indices())
      os << index << ' ';
    os << ": " << x->value() << ' ' << x->variance() << '\n';
  }
  os << '\n';
}

//-----------------------------------------------------------------------------
// Custom accumulator that tracks only the sum of the weights
//-----------------------------------------------------------------------------

#ifndef SWIG_PREPROCESSOR
namespace accumulators {

template <class ValueType = double>
class simple_weighted_sum {
  static_assert(std::is_floating_point<ValueType>::value,
                "ValueType must be a floating point type");

public:
  using value_type = ValueType;
  using const_reference = const value_type&;

  simple_weighted_sum() = default;

  /// Allow implicit conversion from simple_weighted_sum<T>
  template <class T>
  simple_weighted_sum(const simple_weighted_sum<T>& s) noexcept : simple_weighted_sum(s.value()) {}

  /// Initialize simple_weighted_sum explicitly
  simple_weighted_sum(const_reference value) noexcept
      : value_(value) {}

  /// Increment simple_weighted_sum by one
  simple_weighted_sum& operator++() noexcept {
    ++value_;
    return *this;
  }

  /// Increment simple_weighted_sum by weight
  template <class T>
  simple_weighted_sum& operator+=(const bh::weight_type<T>& w) noexcept {
    value_ += w.value;
    return *this;
  }

  /// Add another simple_weighted_sum
  simple_weighted_sum& operator+=(const simple_weighted_sum& rhs) noexcept {
    value_ += rhs.value_;
    return *this;
  }

  /// Scale by value
  simple_weighted_sum& operator*=(const_reference value) noexcept {
    value_ *= value;
    return *this;
  }

  bool operator==(const simple_weighted_sum& rhs) const noexcept {
    return value_ == rhs.value_;
  }

  bool operator!=(const simple_weighted_sum& rhs) const noexcept { return !operator==(rhs); }

  /// Return value of the simple_weighted_sum.
  const_reference value() const noexcept { return value_; }

  // lossy conversion to value type must be explicit
  explicit operator const_reference() const noexcept { return value_; }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    ar& make_nvp("value", value_);
  }

private:
  value_type value_{};
}; // simple_weighted_sum

} // namespace accumulators
#endif // SWIG_PREPROCESSOR

/// Dense storage which tracks sums of weights
using simple_weight_storage = bh::dense_storage<accumulators::simple_weighted_sum<>>;

//-----------------------------------------------------------------------------
// Base class for EEC histograms
//-----------------------------------------------------------------------------

// forward declaration of traits
template<class T>
struct EECHistTraits;

template<class EECHist>
class EECHistBase {
public:

  typedef typename EECHistTraits<EECHist>::Hist Hist;
  typedef typename EECHistTraits<EECHist>::SimpleHist SimpleHist;

  EECHistBase(int num_threads) :
    num_threads_(determine_num_threads(num_threads)),
    hists_(num_threads_),
    simple_hists_(num_threads_)
  {}
  virtual ~EECHistBase() {}

  std::string axes_description() const { return ""; }
  int num_threads() const { return num_threads_; }
  std::size_t nhists() const { return hists().size(); }
  std::size_t nbins(unsigned i = 0) const { return axis(i).size(); }
  std::size_t hist_size(bool include_overflows = true, int i = -1) const {
    if (i == -1) {
      if (include_overflows)
        return hists()[0].size();
      else {
        std::size_t size(1);
        hists()[0].for_each_axis([&size](const auto & a){ size *= a.size(); });
        return size;
      }
    }
    return axis(i).size() + (include_overflows ? 2 : 0);
  }

  std::vector<double> bin_centers(unsigned i = 0) const { return get_bin_centers(hists()[0].axis(i)); }
  std::vector<double> bin_edges(unsigned i = 0) const { return get_bin_edges(hists()[0].axis(i)); }

  void get_hist_errs(double * hist_vals, double * hist_errs,
                     bool include_overflows = true, unsigned hist_i = 0) const {

    if (hist_i >= this->nhists())
      throw std::out_of_range("Requested histogram out of range");

    auto hist(this->combined_hist(hist_i));
    for (auto && x : bh::indexed(hist, (include_overflows ? bh::coverage::all : bh::coverage::inner))) {
      *(hist_vals++) = x->value();
      *(hist_errs++) = std::sqrt(x->variance());
    }
  }

  // return histogram and errors as a pair of vectors
  std::pair<std::vector<double>, std::vector<double>> get_hist_errs(bool include_overflows = true, unsigned hist_i = 0) {
    std::size_t hist_size(this->hist_size(include_overflows));
    auto hist_errs(std::make_pair(std::vector<double>(hist_size), std::vector<double>(hist_size)));
    get_hist_errs(hist_errs.first.data(), hist_errs.second.data(), include_overflows, hist_i);
    return hist_errs;
  }

  std::string hists_as_text(int precision = 16, bool include_overflows = true, std::ostringstream * os = nullptr) const {

    bool os_null(os == nullptr);
    if (os_null)
      os = new std::ostringstream();

    hists()[0].for_each_axis([=](const auto & a){ output_axis(*os, a, precision); });

    // loop over hists
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
      output_hist(*os, combined_hist(hist_i), precision, include_overflows, hist_i);

    return os_null ? os->str() : "";
  }

protected:

#ifndef SWIG_PREPROCESSOR
  auto axis(unsigned i = 0) const { return hists()[0].axis(i); } 
#endif

  // histogram access functions histogram
  std::vector<Hist> & hists(int thread_i = 0) { return hists_[thread_i]; }
  const std::vector<Hist> & hists(int thread_i = 0) const { return hists_[thread_i]; }
  std::vector<SimpleHist> & simple_hists(int thread_i = 0) { return simple_hists_[thread_i]; }
  const std::vector<SimpleHist> & simple_hists(int thread_i = 0) const { return simple_hists_[thread_i]; }

  // combined histograms
  Hist combined_hist(unsigned hist_i) const {

    Hist hist(hists(0)[hist_i]);
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      hist += hists(thread_i)[hist_i];

    return hist;
  }

  // these will be overridden in derived classes
  Hist make_hist() const { assert(false); }
  SimpleHist make_simple_hist() const { assert(false); }

  void duplicate_internal_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    int nnewhists(int(nhists) - int(this->nhists()));
    if (nnewhists > 0) {
      for (int i = 0; i < num_threads(); i++) {
        hists(i).insert(hists(i).end(), nnewhists, static_cast<EECHist &>(*this).make_hist());
        simple_hists(i).insert(simple_hists(i).end(), nnewhists, static_cast<EECHist &>(*this).make_simple_hist());
      }
    }
  }

  void fill_hist_with_simple_hist(int thread_i = 0) {
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {
      auto h_it(hists(thread_i)[hist_i].begin()), h_end(hists(thread_i)[hist_i].end());
      for (auto sh_it = simple_hists(thread_i)[hist_i].begin(); h_it != h_end; ++h_it, ++sh_it)
        *h_it += hist::weight(sh_it->value());
      simple_hists(thread_i)[hist_i].reset();
    }
  }

private:

  int num_threads_;
  
  std::vector<std::vector<Hist>> hists_;
  std::vector<std::vector<SimpleHist>> simple_hists_;

}; // EECHistBase

//-----------------------------------------------------------------------------
// 1D histogram class
//-----------------------------------------------------------------------------

template<class Tr>
class EECHist1D : public EECHistBase<EECHist1D<Tr>> {
public:
  typedef EECHistTraits<EECHist1D<Tr>> Traits;
  typedef typename Traits::Axis Axis;

private:
  Axis axis_;

public:

  EECHist1D(unsigned nbins, double axis_min, double axis_max, int num_threads = 1) :
    EECHistBase<EECHist1D<Tr>>(num_threads),
    axis_(nbins, axis_min, axis_max)
  {
    this->duplicate_internal_hists(1);
  }
  virtual ~EECHist1D() {}

  std::string axes_description() const { return name_transform<Tr>(); }

#ifndef SWIG_PREPROCESSOR
  auto make_hist() const { return bh::make_weighted_histogram(axis_); }
  auto make_simple_hist() const { return bh::make_histogram_with(simple_weight_storage(), axis_); }
#endif

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

}; // EECHist1D

// EECHistTraits for EECHist1D
template<class T>
struct EECHistTraits<EECHist1D<T>> {
  typedef T Transform;
  typedef bh::axis::regular<double, Transform> Axis;
#ifndef SWIG_PREPROCESSOR
  typedef decltype(bh::make_weighted_histogram(Axis())) Hist;
  typedef decltype(bh::make_histogram_with(simple_weight_storage(), Axis())) SimpleHist;
#endif
};

//-----------------------------------------------------------------------------
// 3D histogram class
//-----------------------------------------------------------------------------

template<class Tr0, class Tr1, class Tr2>
class EECHist3D : public EECHistBase<EECHist3D<Tr0, Tr1, Tr2>> {
public:
  typedef EECHistTraits<EECHist3D<Tr0, Tr1, Tr2>> Traits;
  typedef typename Traits::Axis0 Axis0;
  typedef typename Traits::Axis1 Axis1;
  typedef typename Traits::Axis2 Axis2;

private:
  Axis0 axis0_;
  Axis1 axis1_;
  Axis2 axis2_;

public:

  EECHist3D(unsigned nbins0, double axis0_min, double axis0_max,
            unsigned nbins1, double axis1_min, double axis1_max,
            unsigned nbins2, double axis2_min, double axis2_max,
            int num_threads = 1) :
    EECHistBase<EECHist3D<Tr0, Tr1, Tr2>>(num_threads),
    axis0_(nbins0, axis0_min, axis0_max),
    axis1_(nbins1, axis1_min, axis1_max),
    axis2_(nbins2, axis2_min, axis2_max)
  {
    this->duplicate_internal_hists(1);
  }
  virtual ~EECHist3D() {}

  std::string axes_description() const {
    std::ostringstream os;
    os << name_transform<Tr0>() << ", "
       << name_transform<Tr1>() << ", "
       << name_transform<Tr2>();
    return os.str(); }

#ifndef SWIG_PREPROCESSOR
  auto make_hist() const { return bh::make_weighted_histogram(axis0_, axis1_, axis2_); }
  auto make_simple_hist() const { return bh::make_histogram_with(simple_weight_storage(), axis0_, axis1_, axis2_); }
#endif

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

}; // EECHist3D

// EECHistTraits for EECHist3D
template<class T0, class T1, class T2>
struct EECHistTraits<EECHist3D<T0, T1, T2>> {
  typedef T0 Transform0;
  typedef T1 Transform1;
  typedef T2 Transform2;
  typedef bh::axis::regular<double, Transform0> Axis0;
  typedef bh::axis::regular<double, Transform1> Axis1;
  typedef bh::axis::regular<double, Transform2> Axis2;
#ifndef SWIG_PREPROCESSOR
  typedef decltype(bh::make_weighted_histogram(Axis0(), Axis1(), Axis2())) Hist;
  typedef decltype(bh::make_histogram_with(simple_weight_storage(), Axis0(), Axis1(), Axis2())) SimpleHist;
#endif
};

} // namespace hist
} // namespace eec

#endif // EEC_HIST_HH