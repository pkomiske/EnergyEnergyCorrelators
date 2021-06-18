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
 *   _    _ _____  _____ _______ _    _ _______ _____ _       _____ 
 *  | |  | |_   _|/ ____|__   __| |  | |__   __|_   _| |     / ____|
 *  | |__| | | | | (___    | |  | |  | |  | |    | | | |    | (___  
 *  |  __  | | |  \___ \   | |  | |  | |  | |    | | | |     \___ \ 
 *  | |  | |_| |_ ____) |  | |  | |__| |  | |   _| |_| |____ ____) |
 *  |_|  |_|_____|_____/   |_|   \____/   |_|  |_____|______|_____/ 
 */

#ifndef EEC_HISTUTILS_HH
#define EEC_HISTUTILS_HH

#include <ostream>
#include <string>
#include <vector>

// boost histogram
#include "boost/histogram.hpp"

#include "EECUtils.hh"

BEGIN_EEC_NAMESPACE
namespace hist {

namespace bh = boost::histogram;

#ifndef SWIG_PREPROCESSOR  
  using bh::weight;
#endif

//------------------------------------------------------------------------------
// boost::histogram::axis using statements
//------------------------------------------------------------------------------

namespace axis {

using id = bh::axis::transform::id;
using log = bh::axis::transform::log;
using index_type = bh::axis::index_type;

}

//------------------------------------------------------------------------------
// Histogram helper functions
//------------------------------------------------------------------------------

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

// gets boost histogram coverage tag from boolean
inline bh::coverage get_coverage(bool overflows) {
  return overflows ? bh::coverage::all : bh::coverage::inner;
}

// names the various axis transforms
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
void output_axis(std::ostream & os, const Axis & axis, int hist_level, int precision) {
  os.precision(precision);
  if (hist_level > 1) os << "# ";
  else os << "  ";
  if (hist_level > 0)
    os << name_transform<Axis>() << " axis, "
       << axis.size() << " bins, (" << axis.value(0) << ", " << axis.value(axis.size()) << ")\n";
  if (hist_level > 1) {
    os << "bin_edges :";
    for (double edge : get_bin_edges(axis))
      os << ' ' << edge;
    os << '\n'
       << "bin_centers :";
    for (double center : get_bin_centers(axis))
      os << ' ' << center;
    os << "\n\n";
  }
}

} // namespace hist

//------------------------------------------------------------------------------
// Custom accumulators for boost histograms
//------------------------------------------------------------------------------

#ifndef SWIG_PREPROCESSOR
namespace hist {
namespace accumulators {

// tracks just the sum of weights in a histogram
template <class ValueType>
class simple_weighted_sum {
  static_assert(std::is_floating_point<ValueType>::value,
                "ValueType must be a floating point type");

public:
  using value_type = ValueType;
  using const_reference = const value_type &;

  simple_weighted_sum() = default;

  /// Allow implicit conversion from simple_weighted_sum<T>
  template <class T>
  simple_weighted_sum(const simple_weighted_sum<T> & s) noexcept : simple_weighted_sum(s.value()) {}

  /// Initialize simple_weighted_sum explicitly
  simple_weighted_sum(const_reference value) noexcept : value_(value) {}

  /// Increment simple_weighted_sum by one
  simple_weighted_sum & operator++() noexcept {
    ++value_;
    return *this;
  }

  /// Increment simple_weighted_sum by weight
  template <class T>
  simple_weighted_sum & operator+=(const bh::weight_type<T> & w) noexcept {
    value_ += w.value;
    return *this;
  }

  /// Add another simple_weighted_sum
  simple_weighted_sum & operator+=(const simple_weighted_sum & rhs) noexcept {
    value_ += rhs.value_;
    return *this;
  }

  /// Scale by value
  simple_weighted_sum & operator*=(const_reference value) noexcept {
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
    ar & value_;
  }

private:

  value_type value_{};

}; // simple_weighted_sum

} // namespace accumulators

/// Dense storage which tracks sums of weights
using simple_weight_storage = bh::dense_storage<accumulators::simple_weighted_sum<double>>;

} // namespace hist
#endif // SWIG_PREPROCESSOR

END_EEC_NAMESPACE

#endif // EEC_HISTUTILS_HH
