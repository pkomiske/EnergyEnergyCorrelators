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
 *   _    _ _____  _____ _______ ____           _____ ______ 
 *  | |  | |_   _|/ ____|__   __|  _ \   /\    / ____|  ____|
 *  | |__| | | | | (___    | |  | |_) | /  \  | (___ | |__   
 *  |  __  | | |  \___ \   | |  |  _ < / /\ \  \___ \|  __|  
 *  | |  | |_| |_ ____) |  | |  | |_) / ____ \ ____) | |____ 
 *  |_|  |_|_____|_____/   |_|  |____/_/    \_\_____/|______|
 */ 

#ifndef EEC_HISTBASE_HH
#define EEC_HISTBASE_HH

#include <algorithm>
#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// this includes "boost/histogram.hpp"
#include "EECUtils.hh"

namespace eec {
namespace hist {

//------------------------------------------------------------------------------
// EECHistTraits - helps with histogram types
//------------------------------------------------------------------------------

#ifndef SWIG_PREPROCESSOR

// forward declaration of 1D, 2D, and 3D histograms
template<class T0> class EECHist1D;
template<class T0, class T1> class EECHist2D;
template<class T0, class T1, class T2> class EECHist3D;
template<class Hist> struct EECHistTraits;

#endif // SWIG_PREPROCESSOR

//------------------------------------------------------------------------------
// Base class for EEC histograms
//------------------------------------------------------------------------------

template<class EECHist>
class EECHistBase {
public:

  typedef EECHistTraits<EECHist> Traits;
  typedef typename Traits::WeightedHist WeightedHist;
  typedef typename Traits::SimpleWeightedHist SimpleWeightedHist;
  typedef typename Traits::CovarianceHist CovarianceHist;

private:

  // hists - keep track of the central EEC value and naive variance estimate
  std::vector<std::vector<WeightedHist>> hists_;

  // per_event_hists - used to aggregate EEC values during computation for a single event
  std::vector<std::vector<SimpleWeightedHist>> per_event_hists_;

  // these track the covariance of the central EEC value
  std::vector<std::vector<CovarianceHist>> covariance_hists_;

  // these upper bound the variance of the central EEC value
  std::vector<std::vector<SimpleWeightedHist>> variance_bound_hists_;

  std::vector<std::size_t> event_counters_;
  int num_threads_;
  bool track_covariance_, variance_bound_, variance_bound_includes_overflows_;

public:

  EECHistBase(int num_threads,
              bool track_covariance,
              bool variance_bound,
              bool variance_bound_includes_overflows) :
    num_threads_(determine_num_threads(num_threads)),
    track_covariance_(track_covariance),
    variance_bound_(variance_bound),
    variance_bound_includes_overflows_(variance_bound_includes_overflows)
  {}
  virtual ~EECHistBase() = default;

  int num_threads() const { return num_threads_; }
  bool track_covariance() const { return track_covariance_; }
  bool variance_bound() const { return variance_bound_; }
  bool variance_bound_includes_overflows() const { return variance_bound_includes_overflows_; }

  std::size_t nhists() const { return hists_[0].size(); }
  std::size_t nbins(unsigned i = 0) const { return axis(i).size(); }
  static constexpr unsigned rank() { return Traits::rank; }
  std::size_t hist_size(bool overflows = true, int axis_ind = -1) const {
    if (axis_ind == -1) {
      if (overflows)
        return hists_[0][0].size();
      else {
        std::size_t size(1);
        hists_[0][0].for_each_axis([&size](const auto & a){ size *= a.size(); });
        return size;
      }
    }
    return axis(axis_ind).size() + (overflows ? 2 : 0);
  }
  std::size_t covariance_size(bool overflows = true) const {
    auto s(hist_size(overflows));
    return s*s;
  }

  std::size_t event_counter(int thread_i = -1) const {
    if (thread_i >= 0)
      return event_counters_[thread_i];

    // thread_i == -1 means return the sum
    std::size_t event_count(0);
    for (std::size_t count : event_counters_)
      event_count += count;

    return event_count;
  }

#ifndef SWIG
  // low-level access to hists
  std::vector<WeightedHist> & hists(int thread_i = 0) { return hists_[thread_i]; }
  std::vector<CovarianceHist> & covariance_hists(int thread_i = 0) { return covariance_hists_[thread_i]; }
  std::vector<SimpleWeightedHist> & variance_bound_hists(int thread_i = 0) { return variance_bound_hists_[thread_i]; }

  // read-only access
  const std::vector<Hist> & hists(int thread_i = 0) const { return hists_[thread_i]; }
  const std::vector<CovarianceHist> & covariance_hists(int thread_i = 0) const { return covariance_hists_[thread_i]; }
  const std::vector<SimpleWeightedHist> & variance_bound_hists(int thread_i = 0) const { return variance_bound_hists_[thread_i]; }
  
#endif

#ifndef SWIG_PREPROCESSOR
  // access axis of hist
  auto axis(unsigned i = 0) const { return hists_[0][0].axis(i); }
#endif

  // this method wraps rc in a vector and passes on to other reduce
  void reduce(const bh::algorithm::reduce_command & rc) {
    reduce(std::vector<bh::algorithm::reduce_command>{rc});
  }

  // reduce histograms
  void reduce(const std::vector<bh::algorithm::reduce_command> & rcs) {

    if (rcs.size() == 0) return;
    if (rcs.size() > 3) throw std::invalid_argument("too many reduce_commands");

    // get commands for covariance
    std::vector<bh::algorithm::reduce_command> cov_rcs;
    if (track_covariance()) {
      cov_rcs = rcs;
      for (const bh::algorithm::reduce_command & rc : rcs) {

        // check if axis is unset, just push to back of cov_rcs
        if (rc.iaxis == bh::algorithm::reduce_command::unset)
          cov_rcs.push_back(rc);

        // need to increment axis by rank()
        else {
          bh::algorithm::reduce_command new_rc(rc);
          new_rc.iaxis += rank();
          cov_rcs.push_back(new_rc);
        }
      }
    }

    #pragma omp parallel for num_threads(num_threads()) default(shared) schedule(static)
    for (int thread_i = 0; thread_i < num_threads(); thread_i++) {
      for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {
        hists_[thread_i][hist_i] = bh::algorithm::reduce(hists_[thread_i][hist_i], rcs);
        per_event_hists_[thread_i][hist_i] = bh::algorithm::reduce(per_event_hists_[thread_i][hist_i], rcs);
        if (track_covariance())
          covariance_hists_[thread_i][hist_i] = bh::algorithm::reduce(covariance_hists_[thread_i][hist_i], cov_rcs);
        if (variance_bound())
          variance_bound_hists_[thread_i][hist_i] = bh::algorithm::reduce(variance_bound_hists_[thread_i][hist_i], rcs);
      }
    }

    static_cast<EECHist &>(*this).reset_axes();
  }

  // tally histograms
  double sum(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");

    auto s(bh::algorithm::sum(hists_[0][hist_i]));
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      s += bh::algorithm::sum(hists_[thread_i][hist_i]);

    return s.value();
  }

  // compute combined histograms
  WeightedHist combined_hist(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");

    WeightedHist hist(hists_[0][hist_i]);
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      hist += hists_[thread_i][hist_i];

    return hist;
  }

  // compute combined variance bound histograms
  CovarianceHist combined_covariance(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");
    if (!track_covariance())
      throw std::runtime_error("not tracking covariances");

    CovarianceHist covariance_hist(covariance_hists_[0][hist_i]);
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      covariance_hist += covariance_hists_[thread_i][hist_i];

    return covariance_hist;
  }

  // compute combined variance bound histograms
  SimpleWeightedHist combined_variance_bound(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");
    if (!variance_bound())
      throw std::runtime_error("not tracking variance bounds");

    SimpleWeightedHist variance_bound_hist(variance_bound_hists_[0][hist_i]);
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      variance_bound_hist += variance_bound_hists_[thread_i][hist_i];

    return variance_bound_hist;
  }

  // operator to add histograms together
  EECHistBase & operator+=(const EECHistBase & rhs) {
    if (nhists() != rhs.nhists())
      throw std::invalid_argument("cannot add different numbers of histograms together");
    if (track_covariance() != rhs.track_covariance())
      throw std::invalid_argument("track_covariance flags do not match");
    if (variance_bound() != rhs.variance_bound())
      throw std::invalid_argument("variance_bound flags do not match");

    // add everything from rhs to thread 0 histogram WLOG
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {

      // add primary hists
      hists_[0][hist_i] += rhs.combined_hist(hist_i);

      // consider scaling covariances
      if (track_covariance())
        covariance_hists_[0][hist_i] += rhs.combined_covariance(hist_i);

      // consider scaling variance bound
      if (variance_bound())
        variance_bound_hists_[0][hist_i] += rhs.combined_variance_bound(hist_i);
    }

    // include events that rhs has seen in overall event counter
    event_counters_[0] += rhs.event_counter();

    return *this;
  }

  // scale all histograms by a constant
  EECHistBase & operator*=(const double x) {

    // scale each histogram in each thread
    #pragma omp parallel for num_threads(num_threads()) default(shared) schedule(static)
    for (int thread_i = 0; thread_i < num_threads(); thread_i++)
      for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {

        // scale primary histograms
        hists_[thread_i][hist_i] *= x;

        // consider scaling covariances
        if (track_covariance())
          covariance_hists_[thread_i][hist_i] *= x * x;

        // consider scaling variance bound
        if (variance_bound())
          variance_bound_hists_[thread_i][hist_i] *= x * x;
      }

    return *this;
  }

  std::vector<double> bin_centers(unsigned i = 0) const {
    return get_bin_centers(hists_[0][0].axis(i));
  }
  std::vector<double> bin_edges(unsigned i = 0) const {
    return get_bin_edges(hists_[0][0].axis(i));
  }

  std::string hists_as_text(int hist_level = 3, bool overflows = true,
                            int precision = 16, std::ostringstream * os = nullptr) const {

    bool os_null(os == nullptr);
    if (os_null)
      os = new std::ostringstream();

    hists_[0][0].for_each_axis([=](const auto & a){ output_axis(*os, a, hist_level, precision); });

    // some global hist information
    std::string start(hist_level > 1 ? "# " : "  ");
    if (hist_level > 0)
      *os << std::boolalpha
          << start << "track_covariance - " << track_covariance() << '\n'
          << start << "variance_bound - " << variance_bound() << '\n'
          << start << "variance_bound_includes_overflows - " << variance_bound_includes_overflows() << '\n';

    // loop over hists
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
      output_hist(*os, hist_i, hist_level, precision, overflows);

    if (os_null) {
      std::string s(os->str());
      delete os;
      return s;
    }

    return "";
  }

#ifndef SWIG

  // return histogram and errors as a pair of vectors
  std::pair<std::vector<double>, std::vector<double>>
  get_hist_vars(unsigned hist_i = 0, bool overflows = true) {
    auto vars(std::make_pair(std::vector<double>(hist_size(overflows)),
                             std::vector<double>(hist_size(overflows))));
    get_hist_vars(vars.first.data(), vars.second.data(), hist_i, overflows);
    return vars;
  }

  // return covariance and errors as a pair of vectors
  std::vector<double>
  get_covariance(unsigned hist_i = 0, bool overflows = true) {
    std::vector<double> covariance(covariance_size(overflows));
    get_covariance(covariance.data(), hist_i, overflows);
    return covariance;
  }

  // return covariance and errors as a pair of vectors
  std::vector<double>
  get_variance_bound(unsigned hist_i = 0, bool overflows = true) {
    std::vector<double> variance_bound(hist_size(overflows));
    get_variance_bound(variance_bound.data(), hist_i, overflows);
    return variance_bound;
  }

// make low-level functions with pointers private
private:

#endif // SWIG

  void get_hist_vars(double * hist_vals, double * vars,
                     unsigned hist_i = 0, bool overflows = true) const {

    // this will check hist_i for validity
    WeightedHist hist(combined_hist(hist_i));

    // calculate strides
    auto strides(construct_strides<rank()>(overflows));
    axis::index_type extra(overflows ? 1 : 0);
    for (auto && x : bh::indexed(hist, get_coverage(overflows))) {

      // get linearized C-style index
      std::size_t ind(0);
      unsigned r(0);
      for (axis::index_type index : x.indices())
        ind += strides[r++] * (index + extra);

      hist_vals[ind] = x->value();
      vars[ind] = x->variance();
    }
  }

  void get_covariance(double * covariance,
                      unsigned hist_i = 0, bool overflows = true) const {

    // this will check hist_i for validity and that we're tracking variance bounds
    WeightedHist hist(combined_hist(hist_i));
    CovarianceHist covariance_hist(combined_covariance(hist_i));

    // zero out the input
    std::fill(covariance, covariance + covariance_size(overflows), 0);

    // calculate strides
    auto strides(construct_strides<2*rank()>(overflows));

    // iterate over pairs of simple_hist bins
    axis::index_type extra(overflows ? 1 : 0);
    const double event_count(event_counter());
    for (auto && x : bh::indexed(covariance_hist, get_coverage(overflows))) {
      if (x->value() == 0) continue;

      // get linearized C-style index
      std::size_t ind(0), indT(0);
      unsigned r(0);
      std::array<std::array<axis::index_type, hist.rank()>, 2> hist_inds;
      for (axis::index_type index : x.indices()) {
        hist_inds[r/hist.rank()][r%hist.rank()] = index;
        std::size_t i(index + extra);
        indT += strides[(r + rank()) % strides.size()] * i;
        ind += strides[r++] * i;
      }

      // only upper triangular covariance was stored, so ensure we yield correct symmetric result
      double cov(x->value() - hist[hist_inds[0]].value()*hist[hist_inds[1]].value()/event_count);
      if (ind == indT)
        covariance[ind] = cov;
      else {
        covariance[ind] += cov;
        covariance[indT] += cov;
      }
    }
  }

  void get_variance_bound(double * variance_bound,
                          unsigned hist_i = 0, bool overflows = true) const {

    // this will check hist_i for validity and that we're tracking variance bounds
    SimpleWeightedHist variance_bound_hist(combined_variance_bound(hist_i));

    // todo: make function for calculating strides

    // calculate strides
    auto strides(construct_strides<rank()>(overflows));
    axis::index_type extra(overflows ? 1 : 0);
    for (auto && x : bh::indexed(variance_bound_hist, get_coverage(overflows))) {

      // get linearized C-style index
      std::size_t ind(0);
      unsigned r(0);
      for (axis::index_type index : x.indices())
        ind += strides[r++] * (index + extra);

      variance_bound[ind] = x->value();
    }
  }

protected:

  void init(unsigned nh) {
    hists_.clear();
    per_event_hists_.clear();
    covariance_hists_.clear();
    variance_bound_hists_.clear();

    hists_.resize(num_threads());
    per_event_hists_.resize(num_threads());
    if (track_covariance()) covariance_hists_.resize(num_threads());
    if (variance_bound()) variance_bound_hists_.resize(num_threads());

    resize_internal_hists(nh);

    event_counters_.resize(num_threads(), 0);
  }

  void resize_internal_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    for (int thread_i = 0; thread_i < num_threads(); thread_i++) {
      hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_hist());
      per_event_hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_simple_hist());
      if (track_covariance())
        covariance_hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_covariance_hist());
      if (variance_bound())
        variance_bound_hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_simple_hist());
    }
  }

  // for use in derived EEC classes
  std::string axes_description() const { return ""; }

  // access to simple hists
  std::vector<SimpleWeightedHist> & per_event_hists(int thread_i = 0) { return per_event_hists_[thread_i]; }
  const std::vector<SimpleWeightedHist> & per_event_hists(int thread_i = 0) const { return per_event_hists_[thread_i]; }

  // these will be overridden in derived classes
  WeightedHist make_hist() const { throw std::runtime_error("method should be overridden"); }
  SimpleWeightedHist make_simple_hist() const { throw std::runtime_error("method should be overridden"); }
  CovarianceHist make_covariance_hist() const { throw std::runtime_error("method should be overridden"); }

  // fills histograms for a specific thread with the values currently in per_event_hists
  void fill_from_single_event(int thread_i = 0, double event_weight = 1) {

    // increment number of events for this thread
    event_counters_[thread_i]++;

    // for each histogram
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {

      // track covariances
      if (track_covariance()) {

        CovarianceHist & cov_hist(covariance_hists_[thread_i][hist_i]);
        std::array<axis::index_type, 2*Traits::rank> cov_inds;

        // iterate over pairs of simple_hist bins
        auto outer_ind_range(bh::indexed(per_event_hists_[thread_i][hist_i], bh::coverage::all));
        auto outer_it = outer_ind_range.begin(), end(outer_ind_range.end());

        // specialize for 2D covariance
        if (cov_hist.rank() == 2) {
          for (; outer_it != end; ++outer_it) {
            const double outer_bin_val((*outer_it)->value());
            if (outer_bin_val == 0) continue;

            // store bin index in cov_inds
            cov_inds[0] = outer_it->index(0);

            // inner loop picks up from where outer loop is
            for (auto inner_it = outer_it; inner_it != end; ++inner_it) {
              cov_inds[1] = inner_it->index(0);
              cov_hist[cov_inds] += outer_bin_val * (*inner_it)->value();
            }
          }
        }
        else {
          for (; outer_it != end; ++outer_it) {
            const double outer_bin_val((*outer_it)->value());
            if (outer_bin_val == 0) continue;

            // store bin indices in cov_inds
            auto outer_inds(outer_it->indices());
            std::copy(outer_inds.begin(), outer_inds.end(), cov_inds.begin());

            // inner loop picks up from where outer loop is
            for (auto inner_it = outer_it; inner_it != end; ++inner_it) {
              auto inner_inds(inner_it->indices());
              std::copy(inner_inds.begin(), inner_inds.end(), cov_inds.begin() + Traits::rank);
              cov_hist[cov_inds] += outer_bin_val * (*inner_it)->value();
            }
          } 
        }
      }

      // iterator over hist
      auto h_it(hists_[thread_i][hist_i].begin());

      // we're keeping track of the variance bound
      if (variance_bound()) {
        const double simple_hist_sum(event_weight *
                                     bh::algorithm::sum(per_event_hists_[thread_i][hist_i],
                                                        (variance_bound_includes_overflows_ ?
                                                         bh::coverage::all :
                                                         bh::coverage::inner)).value());

        auto eb_it(variance_bound_hists_[thread_i][hist_i].begin());
        for (auto sh_it = per_event_hists_[thread_i][hist_i].begin(),
                 sh_end = per_event_hists_[thread_i][hist_i].end();
             sh_it != sh_end;
             ++sh_it, ++h_it, ++eb_it) {

          // can skip all zeros
          if (sh_it->value() != 0) {
            *h_it += hist::weight(sh_it->value());
            *eb_it += simple_hist_sum * sh_it->value();
            *sh_it = 0;
          }
        }
      }

      // not keeping track of variance bound
      else {
        for (auto sh_it = per_event_hists_[thread_i][hist_i].begin(),
                 sh_end = per_event_hists_[thread_i][hist_i].end();
             sh_it != sh_end;
             ++sh_it, ++h_it) {

          // can skip all zeros
          if (sh_it->value() != 0) {
            *h_it += hist::weight(sh_it->value());
            *sh_it = 0;
          }
        }
      }
    }
  }

private:

  template<unsigned N>
  std::array<std::size_t, N> construct_strides(bool overflows) const {

    // calculate strides
    std::array<std::size_t, N> strides;
    strides.back() = 1;
    for (axis::index_type r = N - 1; r > 0; r--)
      strides[r-1] = strides[r] * hist_size(overflows, r % rank());

    return strides;
  }

  void output_hist(std::ostream & os, int hist_i, int hist_level,
                                      int precision, bool overflows) const {
    os.precision(precision);
    if (hist_level > 2) os << "# ";
    else os << "  ";
    if (hist_level > 0 && hist_i == 0) {
      if (hist_i != -1 && hist_level > 2) os << "hist " << hist_i;
      os << "rank " << hists_[0][hist_i].rank()
         << " hist, " << hist_size(overflows) << " total bins, "
         << (overflows ? "including" : "excluding") << " overflows\n";
    }
    if (hist_level > 2) {
      os << "# bin_multi_index : bin_value bin_variance\n";
      auto hist(combined_hist(hist_i));
      for (auto && x : bh::indexed(hist, get_coverage(overflows))) {
        for (axis::index_type index : x.indices())
          os << index << ' ';
        os << ": " << x->value() << ' ' << x->variance() << '\n';
      }
      os << '\n';
    }
  }

#ifdef BOOST_SERIALIZATION_ACCESS_HPP
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

  template<class Archive>
  void save(Archive & ar, const unsigned int /* file_version */) const {
    ar & num_threads_ & nhists() & event_counters_
       & track_covariance_
       & variance_bound_  & variance_bound_includes_overflows_;

    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {
      ar & combined_hist(hist_i);
      if (track_covariance())
        ar & combined_covariance(hist_i);
      if (variance_bound())
        ar & combined_variance_bound(hist_i);
    }
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int /* file_version */) {
    std::size_t nh;
    ar & num_threads_ & nh & event_counters_
       & track_covariance_
       & variance_bound_ & variance_bound_includes_overflows_;

    // initialize with a specific number of histograms
    init(nh);

    // for each hist, load it into thread 0
    for (unsigned hist_i = 0; hist_i < nh; hist_i++) {
      ar & hists_[0][hist_i];
      if (track_covariance())
        ar & covariance_hists_[0][hist_i];
      if (variance_bound())
        ar & variance_bound_hists_[0][hist_i];
    }
  }

}; // EECHistBase

} // namespace hist
} // namespace eec

#endif // EEC_HISTBASE_HH
