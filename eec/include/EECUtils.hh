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
 *   _    _ _______ _____ _       _____ 
 *  | |  | |__   __|_   _| |     / ____|
 *  | |  | |  | |    | | | |    | (___  
 *  | |  | |  | |    | | | |     \___ \ 
 *  | |__| |  | |   _| |_| |____ ____) |
 *   \____/   |_|  |_____|______|_____/ 
 */

#ifndef EEC_UTILS_HH
#define EEC_UTILS_HH

#include <ostream>
#include <string>
#include <vector>

// boost histogram
#include "boost/histogram.hpp"

// OpenMP for multithreading
#ifdef _OPENMP
# include <omp.h>
#endif

#ifdef SWIG_FASTJET
# include "fastjet/PseudoJet.hh"
#endif

// serialization code based on boost serialization
#ifdef EEC_SERIALIZATION
# include <boost/archive/binary_iarchive.hpp>
# include <boost/archive/binary_oarchive.hpp>
# include <boost/archive/text_iarchive.hpp>
# include <boost/archive/text_oarchive.hpp>
# include <boost/serialization/array.hpp>
# include <boost/serialization/map.hpp>
# include <boost/serialization/string.hpp>
# include <boost/serialization/vector.hpp>

// compression based on boost iostreams
# ifdef EEC_COMPRESSION
#  include <boost/iostreams/filtering_stream.hpp>
#  include <boost/iostreams/filter/zlib.hpp>
# endif

#endif // EEC_SERIALIZATION

// check for fastjet support
#if defined(__FASTJET_PSEUDOJET_HH__) || defined(SWIG_FASTJET)
# ifndef EEC_FASTJET_SUPPORT
# define EEC_FASTJET_SUPPORT
# endif
#endif

namespace eec {

//-----------------------------------------------------------------------------
// enums
//-----------------------------------------------------------------------------

enum class ArchiveFormat { Text, Binary };
enum class CompressionMode { Auto, Plain, Zlib };

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------

// these should be accessed using the below functions
#ifndef SWIG_PREPROCESSOR
static ArchiveFormat archform_ = ArchiveFormat::Text;
static CompressionMode compmode_ = CompressionMode::Auto;
#endif

const double REG = 1e-100;
const double PI = 3.14159265358979323846;
const double TWOPI = 6.28318530717958647693;

#ifdef SWIG
constexpr bool HAS_PICKLE_SUPPORT = 
  #ifdef EEC_SERIALIZATION
    true;
  #else
    false;
  #endif
#endif

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

// get/set some serialization options
inline ArchiveFormat get_archive_format() { return archform_; }
inline CompressionMode get_compression_mode() {
  if (compmode_ == CompressionMode::Auto) {
    #ifdef EEC_COMPRESSION
      return CompressionMode::Zlib;
    #else
      return CompressionMode::Plain;
    #endif
  }
  return compmode_;
}
inline void set_archive_format(ArchiveFormat a) { archform_ = a; }
inline void set_compression_mode(CompressionMode c) {

  // error if compression specifically requested and not available
  #ifndef EEC_COMPRESSION
    if (c != CompressionMode::Auto && c != CompressionMode::Plain)
      throw std::invalid_argument("compression not available with this build");
  #endif

  compmode_ = c;
}

// determine the number of threads to use
inline int determine_num_threads(int num_threads) {
#ifdef _OPENMP
  if (num_threads == -1 || num_threads > omp_get_max_threads())
    return omp_get_max_threads();
  if (num_threads < 1) return 1;
  return num_threads;
#else
  return 1;
#endif
}

// gets thread num if OpenMP is enabled, otherwise returns 0
inline int get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

namespace hist {

namespace bh = boost::histogram;

#ifndef SWIG_PREPROCESSOR  
  using bh::weight;
#endif

//------------------------------------------------------------------------------
// boost::histogram::axis::transform using statements
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
template <class ValueType = double>
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
using simple_weight_storage = bh::dense_storage<accumulators::simple_weighted_sum<>>;

} // namespace hist
#endif // SWIG_PREPROCESSOR

} // namespace eec

#endif // EEC_UTILS_HH