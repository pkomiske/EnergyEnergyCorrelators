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
 *   _    _ _____  _____ _______ 
 *  | |  | |_   _|/ ____|__   __|
 *  | |__| | | | | (___    | |   
 *  |  __  | | |  \___ \   | |   
 *  | |  | |_| |_ ____) |  | |   
 *  |_|  |_|_____|_____/   |_|   
 */

#ifndef EEC_HIST_HH
#define EEC_HIST_HH

#include <array>
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

namespace bh = boost::histogram;

#ifndef SWIG_PREPROCESSOR  
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

#ifdef SWIG
constexpr bool HAS_PICKLE_SUPPORT = 
  #ifdef EEC_SERIALIZATION
    true;
  #else
    false;
  #endif
#endif

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
  /*simple_weighted_sum& operator++() noexcept {
    ++value_;
    return *this;
  }*/

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
    ar & value_;
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

private:

  std::vector<std::vector<Hist>> hists_;
  std::vector<std::vector<SimpleHist>> simple_hists_;

  int num_threads_;

public:

  EECHistBase(int num_threads) : num_threads_(determine_num_threads(num_threads)) {}
  virtual ~EECHistBase() = default;

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

  // reduce histograms
  void reduce(const std::vector<bh::algorithm::reduce_command> & rcs) {

    for (int thread_i = 0; thread_i < this->num_threads(); thread_i++) {
      auto & hs(this->hists(thread_i));
      for (unsigned hist_i = 0; hist_i < this->nhists(); hist_i++) {
        if (rcs.size() == 1)
          hs[hist_i] = bh::algorithm::reduce(hs[hist_i], rcs[0]);
        else if (rcs.size() == 2)
          hs[hist_i] = bh::algorithm::reduce(hs[hist_i], rcs[0], rcs[1]);
        else if (rcs.size() == 3)
          hs[hist_i] = bh::algorithm::reduce(hs[hist_i], rcs[0], rcs[1], rcs[2]);
        else
          throw std::invalid_argument("too many reduce_commands");
      }
    }

    static_cast<EECHist &>(*this).reset_axes();
  }

  // tally histograms
  double sum(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");

    auto s(bh::algorithm::sum(hists(0)[hist_i]));
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      s += bh::algorithm::sum(hists(thread_i)[hist_i]);

    return s.value();
  }

  // compute combined histograms
  Hist combined_hist(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");

    Hist hist(hists(0)[hist_i]);
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      hist += hists(thread_i)[hist_i];

    return hist;
  }

  // operator to add histograms together
  EECHistBase & operator+=(const EECHistBase<EECHist> & rhs) {
    if (nhists() != rhs.nhists())
      throw std::invalid_argument("cannot add different numbers of histograms together");

    // add everything from rhs to thread 0 histogram WLOG
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
      add(rhs.combined_hist(hist_i), hist_i);

    return *this;
  }

  // function to add specific histograms
  void add(const Hist & h, unsigned hist_i = 0) {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");

    hists()[hist_i] += h;
  }

  std::vector<double> bin_centers(unsigned i = 0) const { return get_bin_centers(hists()[0].axis(i)); }
  std::vector<double> bin_edges(unsigned i = 0) const { return get_bin_edges(hists()[0].axis(i)); }

  void get_hist_vars(double * hist_vals, double * hist_vars,
                     unsigned hist_i = 0, bool include_overflows = true) const {

    if (hist_i >= this->nhists())
      throw std::invalid_argument("Requested histogram out of range");
    auto hist(this->combined_hist(hist_i));

    // calculate strides
    int extra(include_overflows ? 2 : 0);
    std::array<std::size_t, hist.rank()> strides;
    strides.back() = 1;
    for (int r = hist.rank() - 1; r > 0; r--)
      strides[r-1] = strides[r] * (axis(r).size() + extra);
    
    extra = (include_overflows ? 1 : 0);
    for (auto && x : bh::indexed(hist, (include_overflows ? bh::coverage::all : bh::coverage::inner))) {

      // get linearized C-style index
      std::size_t ind(0);
      int r(0);
      for (int index : x.indices())
        ind += strides[r++] * (index + extra);

      hist_vals[ind] = x->value();
      hist_vars[ind] = x->variance();
    }
  }

  // return histogram and errors as a pair of vectors
  std::pair<std::vector<double>, std::vector<double>> get_hist_vars(bool include_overflows = true, unsigned hist_i = 0) {
    std::size_t hist_size(this->hist_size(include_overflows));
    auto hist_vars(std::make_pair(std::vector<double>(hist_size), std::vector<double>(hist_size)));
    get_hist_vars(hist_vars.first.data(), hist_vars.second.data(), include_overflows, hist_i);
    return hist_vars;
  }

  std::string hists_as_text(int hist_level = 3, int precision = 16,
                            bool include_overflows = true, std::ostringstream * os = nullptr) const {

    bool os_null(os == nullptr);
    if (os_null)
      os = new std::ostringstream();

    hists()[0].for_each_axis([=](const auto & a){ output_axis(*os, a, hist_level, precision); });

    // loop over hists
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
      output_hist(*os, hist_i, hist_level, precision, include_overflows);

    if (os_null) {
      std::string s(os->str());
      delete os;
      return s;
    }

    return "";
  }

protected:

  void init(unsigned nh) {
    hists_.clear();
    simple_hists_.clear();
    hists_.resize(num_threads());
    simple_hists_.resize(num_threads());
    resize_internal_hists(nh);
  }

  std::string axes_description() const { return ""; }

#ifndef SWIG_PREPROCESSOR
  auto axis(unsigned i = 0) const { return hists()[0].axis(i); } 
#endif

  // access to simple histograms
  std::vector<Hist> & hists(int thread_i = 0) { return hists_[thread_i]; }
  const std::vector<Hist> & hists(int thread_i = 0) const { return hists_[thread_i]; }
  std::vector<SimpleHist> & simple_hists(int thread_i = 0) { return simple_hists_[thread_i]; }
  const std::vector<SimpleHist> & simple_hists(int thread_i = 0) const { return simple_hists_[thread_i]; }

  // these will be overridden in derived classes
  Hist make_hist() const { assert(false); }
  SimpleHist make_simple_hist() const { assert(false); }

  void resize_internal_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    int nnewhists(int(nhists) - int(this->nhists()));
    for (int i = 0; i < num_threads(); i++) {
      if (nnewhists > 0) {
        hists(i).insert(hists(i).end(), nnewhists, static_cast<EECHist &>(*this).make_hist());
        simple_hists(i).insert(simple_hists(i).end(), nnewhists, static_cast<EECHist &>(*this).make_simple_hist());
      }
      else {
        hists(i).resize(nhists);
        simple_hists(i).resize(nhists);
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

  void output_hist(std::ostream & os, int hist_i, int hist_level,
                                      int precision, bool include_overflows) const {
    os.precision(precision);
    if (hist_level > 2) os << "# ";
    else os << "  ";
    if (hist_level > 0 && hist_i == 0) {
      if (hist_i != -1 && hist_level > 2) os << "hist " << hist_i;
      os << "rank " << hists()[hist_i].rank()
         << " hist, " << hists()[hist_i].size() << " total bins including overflows\n";
    }
    if (hist_level > 2) {
      os << "# bin_multi_index : bin_value bin_variance\n";
      auto hist(combined_hist(hist_i));
      for (auto && x : bh::indexed(hist, include_overflows ? bh::coverage::all : bh::coverage::inner)) {
        for (int index : x.indices())
          os << index << ' ';
        os << ": " << x->value() << ' ' << x->variance() << '\n';
      }
      os << '\n';
    }
  }

#if defined(EEC_SERIALIZATION) && !defined(SWIG_PREPROCESSOR)
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive & ar, const unsigned int /* file_version */) const {
    ar & num_threads_ & nhists();
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
      ar & combined_hist(hist_i);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int /* file_version */) {
    std::size_t nh;
    ar & num_threads_ & nh;

    // initialize with a specific number of histograms
    init(nh);

    // for each hist, load it into thread 0
    for (unsigned hist_i = 0; hist_i < nh; hist_i++)
      ar & hists()[hist_i];
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

}; // EECHistBase

//-----------------------------------------------------------------------------
// 1D histogram class
//-----------------------------------------------------------------------------

template<class Tr>
class EECHist1D : public EECHistBase<EECHist1D<Tr>> {
public:
  typedef EECHist1D<Tr> Self;
  typedef EECHistBase<Self> Base;
  typedef EECHistTraits<Self> Traits;
  typedef typename Traits::Axis Axis;

private:
  
  unsigned nbins_;
  double axis_min_, axis_max_;

public:

  EECHist1D(unsigned nbins, double axis_min, double axis_max, int num_threads = 1) :
    EECHistBase<EECHist1D<Tr>>(num_threads),
    nbins_(nbins), axis_min_(axis_min), axis_max_(axis_max)
  {
    this->init(1);
  }
  virtual ~EECHist1D() = default;

  void reduce(const bh::algorithm::reduce_command & rc) {
    Base::reduce({rc});
  }

#ifndef SWIG_PREPROCESSOR
  void reset_axes() {
    nbins_ = this->nbins();
    axis_min_ = this->axis().value(0);
    axis_max_ = this->axis().value(nbins_);
  }
  auto make_hist() const { return bh::make_weighted_histogram(Axis(nbins_, axis_min_, axis_max_)); }
  auto make_simple_hist() const { return bh::make_histogram_with(simple_weight_storage(), Axis(nbins_, axis_min_, axis_max_)); }
#endif

protected:

  std::string axes_description() const { return name_transform<Tr>(); }

private:

#ifdef EEC_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & nbins_ & axis_min_ & axis_max_;
    ar & boost::serialization::base_object<Base>(*this);
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
  typedef EECHist3D<Tr0, Tr1, Tr2> Self;
  typedef EECHistBase<Self> Base;
  typedef EECHistTraits<Self> Traits;
  typedef typename Traits::Axis0 Axis0;
  typedef typename Traits::Axis1 Axis1;
  typedef typename Traits::Axis2 Axis2;

private:
  
  std::array<unsigned, 3> nbins_;
  std::array<double, 3> axis_mins_;
  std::array<double, 3> axis_maxs_;

public:

  EECHist3D(unsigned nbins0, double axis0_min, double axis0_max,
            unsigned nbins1, double axis1_min, double axis1_max,
            unsigned nbins2, double axis2_min, double axis2_max,
            int num_threads = 1) :
    Base(num_threads),
    nbins_({nbins0, nbins1, nbins2}),
    axis_mins_({axis0_min, axis1_min, axis2_min}),
    axis_maxs_({axis0_max, axis1_max, axis2_max})
  {
    this->init(1);
  }
  virtual ~EECHist3D() = default;

#ifndef SWIG_PREPROCESSOR
  void reset_axes() {
    for (unsigned i = 0; i < 3; i++) {
      nbins_[i] = this->nbins(i);
      axis_mins_[i] = this->axis(i).value(0);
      axis_maxs_[i] = this->axis(i).value(nbins_[i]);
    }
  }

  auto make_hist() const {
    return bh::make_weighted_histogram(Axis0(nbins_[0], axis_mins_[0], axis_maxs_[0]),
                                       Axis1(nbins_[1], axis_mins_[1], axis_maxs_[1]),
                                       Axis2(nbins_[2], axis_mins_[2], axis_maxs_[2]));
  }
  auto make_simple_hist() const {
    return bh::make_histogram_with(simple_weight_storage(), Axis0(nbins_[0], axis_mins_[0], axis_maxs_[0]),
                                                            Axis1(nbins_[1], axis_mins_[1], axis_maxs_[1]),
                                                            Axis2(nbins_[2], axis_mins_[2], axis_maxs_[2]));
  }
#endif

protected:

  std::string axes_description() const {
    std::ostringstream os;
    os << name_transform<Tr0>() << ", "
       << name_transform<Tr1>() << ", "
       << name_transform<Tr2>();
    return os.str();
  }

private:

#ifdef EEC_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & nbins_ & axis_mins_ & axis_maxs_;
    ar & boost::serialization::base_object<Base>(*this);
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