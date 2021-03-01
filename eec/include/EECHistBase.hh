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

#ifndef EEC_HIST_HH
#define EEC_HIST_HH

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
  typedef typename Traits::Hist Hist;
  typedef typename Traits::SimpleHist SimpleHist;
  typedef typename Traits::CovarianceHist CovarianceHist;

private:

  // hists - keep track of the central EEC value and naive variance estimate
  std::vector<std::vector<Hist>> hists_;

  // simple_hists - used to aggregate EEC values during computation for a single event
  std::vector<std::vector<SimpleHist>> simple_hists_;

  // these upper bound the variance of the central EEC value
  std::vector<std::vector<SimpleHist>> error_bound_hists_;

  // these track the covariance of the central EEC value
  std::vector<std::vector<CovarianceHist>> covariance_hists_;

  int num_threads_;
  bool error_bound_, track_covariance_, error_bound_include_overflows_;

public:

  EECHistBase(int num_threads,
              bool error_bound,
              bool track_covariance,
              bool error_bound_include_overflows) :
    num_threads_(determine_num_threads(num_threads)),
    error_bound_(error_bound),
    track_covariance_(track_covariance),
    error_bound_include_overflows_(error_bound_include_overflows)
  {}
  virtual ~EECHistBase() = default;

  int num_threads() const { return num_threads_; }
  bool error_bound() const { return error_bound_; }
  bool track_covariance() const { return track_covariance_; }

  std::size_t nhists() const { return hists_[0].size(); }
  std::size_t nbins(unsigned i = 0) const { return axis(i).size(); }
  constexpr unsigned rank() const { return Traits::rank; }
  std::size_t hist_size(bool include_overflows = true, int i = -1) const {
    if (i == -1) {
      if (include_overflows)
        return hists_[0][0].size();
      else {
        std::size_t size(1);
        hists_[0][0].for_each_axis([&size](const auto & a){ size *= a.size(); });
        return size;
      }
    }
    return axis(i).size() + (include_overflows ? 2 : 0);
  }

  // reduce histograms
  void reduce(const std::vector<bh::algorithm::reduce_command> & rcs) {

    unsigned r(rcs.size());
    if (r == 0) return;
    if (r > 3) throw std::invalid_argument("too many reduce_commands");

    // lambda function for reducing hist
    auto reduce = [](const auto & hist, const auto & rcs) {
      if (rcs.size() == 1) return bh::algorithm::reduce(hist, rcs[0]);
      if (rcs.size() == 2) return bh::algorithm::reduce(hist, rcs[0], rcs[1]);
      return bh::algorithm::reduce(hist, rcs[0], rcs[1], rcs[2]);
    };

    for (int thread_i = 0; thread_i < this->num_threads(); thread_i++) {
      for (unsigned hist_i = 0; hist_i < this->nhists(); hist_i++) {
        hists_[thread_i][hist_i] = reduce(hists_[thread_i][hist_i], rcs);
        simple_hists_[thread_i][hist_i] = reduce(simple_hists_[thread_i][hist_i], rcs);
        if (error_bound_)
          error_bound_hists_[thread_i][hist_i] = reduce(error_bound_hists_[thread_i][hist_i], rcs);
        if (track_covariance_)
          covariance_hists_[thread_i][hist_i] = reduce(covariance_hists_[thread_i][hist_i], rcs);
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
  Hist combined_hist(unsigned hist_i = 0) const {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");

    Hist hist(hists_[0][hist_i]);
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      hist += hists_[thread_i][hist_i];

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

  EECHistBase & operator*=(const double x) {

    // scale each histogram in each thread
    for (int thread_i = 1; thread_i < num_threads(); thread_i++)
      for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
        scale(x, hist_i, thread_i);

    return *this;
  }

  // function to add specific histograms
  void add(const Hist & h, unsigned hist_i = 0, int thread_i = 0) {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");
    hists_[thread_i][hist_i] += h;
  }

  // function to scale specific histogram by a value
  void scale(const double x, unsigned hist_i = 0, int thread_i = 0) {
    if (hist_i >= nhists())
      throw std::invalid_argument("invalid histogram index");
    hists_[thread_i][hist_i] *= x;
  }

  std::vector<double> bin_centers(unsigned i = 0) const {
    return get_bin_centers(hists_[0][0].axis(i));
  }
  std::vector<double> bin_edges(unsigned i = 0) const {
    return get_bin_edges(hists_[0][0].axis(i));
  }

  void get_hist_vars(double * hist_vals, double * hist_vars,
                     unsigned hist_i = 0, bool include_overflows = true) const {

    if (hist_i >= this->nhists())
      throw std::invalid_argument("Requested histogram out of range");
    auto hist(this->combined_hist(hist_i));

    // calculate strides
    axis::index_type extra(include_overflows ? 2 : 0);
    std::array<std::size_t, 2*hist.rank()> strides;
    strides.back() = 1;
    for (axis::index_type r = 2*hist.rank() - 1; r > 0; r--)
      strides[r-1] = strides[r] * (axis(r % hist.rank()).size() + extra);
    
    extra = (include_overflows ? 1 : 0);
    for (auto && x : bh::indexed(hist, include_overflows ? bh::coverage::all : bh::coverage::inner)) {

      // get linearized C-style index
      std::size_t ind(0);
      axis::index_type r(hist.rank());
      for (axis::index_type index : x.indices())
        ind += strides[r++] * (index + extra);

      hist_vals[ind] = x->value();
      hist_vars[ind] = x->variance();
    }
  }

  // return histogram and errors as a pair of vectors
  std::pair<std::vector<double>, std::vector<double>>
  get_hist_vars(bool include_overflows = true, unsigned hist_i = 0) {
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

    hists_[0][0].for_each_axis([=](const auto & a){ output_axis(*os, a, hist_level, precision); });

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
    if (error_bound_) error_bound_hists_.resize(num_threads());
    if (track_covariance_) covariance_hists_.resize(num_threads());
    resize_internal_hists(nh);
  }

  std::string axes_description() const { return ""; }

#ifndef SWIG_PREPROCESSOR
  auto axis(unsigned i = 0) const { return hists_[0][0].axis(i); } 
#endif

  // access to simple hists
  std::vector<SimpleHist> & simple_hists(int thread_i = 0) { return simple_hists_[thread_i]; }
  const std::vector<SimpleHist> & simple_hists(int thread_i = 0) const { return simple_hists_[thread_i]; }

  // these will be overridden in derived classes
  Hist make_hist() const { throw std::logic_error("method should be overridden"); }
  SimpleHist make_simple_hist() const { throw std::logic_error("method should be overridden"); }
  CovarianceHist make_covariance_hist() const { throw std::logic_error("method should be overridden"); }

  void resize_internal_hists(unsigned nhists) {
    if (nhists == 0)
      throw std::invalid_argument("nhists must be at least 1");

    // create histograms
    for (int thread_i = 0; thread_i < num_threads(); thread_i++) {
      hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_hist());
      simple_hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_simple_hist());
      if (error_bound_)
        error_bound_hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_simple_hist());
      if (track_covariance_)
        covariance_hists_[thread_i].resize(nhists, static_cast<EECHist &>(*this).make_covariance_hist());
    }
  }

  void fill_hist_with_simple_hist(int thread_i = 0, double event_weight = 1) {

    // for each histogram
    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++) {

      // track covariances
      if (track_covariance_) {

        CovarianceHist & cov_hist(covariance_hists_[thread_i][hist_i]);
        std::array<axis::index_type, 2*Traits::rank> cov_inds;

        // iterate over pairs of simple_hist bins
        auto outer_ind_range(bh::indexed(simple_hists_[thread_i][hist_i], bh::coverage::all));
        auto outer_it = outer_ind_range.begin(), end(outer_ind_range.end());

        // specialize for 2D covariance
        if (cov_hist.rank() == 2) {
          for (; outer_it != end; ++outer_it) {
            const double outer_bin_val((*outer_it)->value());
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
            auto outer_inds(outer_it->indices());
            std::copy(outer_inds.begin(), outer_inds.end(), cov_inds.begin());

            // inner loop picks up from where outer loop is
            for (auto inner_it = outer_it; inner_it != end; ++inner_it) {
              auto inner_inds(inner_it->indices());
              std::copy(inner_inds.begin(), inner_inds.end(), cov_inds.begin() + rank());
              cov_hist[cov_inds] += outer_bin_val * (*inner_it)->value();
            }
          } 
        }
      }

      // iterator over hist
      auto h_it(hists_[thread_i][hist_i].begin());

      // we're keeping track of the error bound
      if (error_bound_) {
        const double simple_hist_sum(event_weight *
                                     bh::algorithm::sum(simple_hists_[thread_i][hist_i],
                                                        (error_bound_include_overflows_ ?
                                                         bh::coverage::all :
                                                         bh::coverage::inner)).value());

        auto eb_it(error_bound_hists_[thread_i][hist_i].begin());
        for (auto sh_it = simple_hists_[thread_i][hist_i].begin(),
                 sh_end = simple_hists_[thread_i][hist_i].end();
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

      // not keeping track of error bound
      else {
        for (auto sh_it = simple_hists_[thread_i][hist_i].begin(),
                 sh_end = simple_hists_[thread_i][hist_i].end();
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

  void output_hist(std::ostream & os, int hist_i, int hist_level,
                                      int precision, bool include_overflows) const {
    os.precision(precision);
    if (hist_level > 2) os << "# ";
    else os << "  ";
    if (hist_level > 0 && hist_i == 0) {
      if (hist_i != -1 && hist_level > 2) os << "hist " << hist_i;
      os << "rank " << hists_[0][hist_i].rank()
         << " hist, " << hists_[0][hist_i].size() << " total bins including overflows\n";
    }
    if (hist_level > 2) {
      os << "# bin_multi_index : bin_value bin_variance\n";
      auto hist(combined_hist(hist_i));
      for (auto && x : bh::indexed(hist, include_overflows ? bh::coverage::all : bh::coverage::inner)) {
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
    ar & num_threads_ & nhists()
       & error_bound_ & track_covariance_ & error_bound_include_overflows_;

    for (unsigned hist_i = 0; hist_i < nhists(); hist_i++)
      ar & combined_hist(hist_i);
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int /* file_version */) {
    std::size_t nh;
    ar & num_threads_ & nh
       & error_bound_ & track_covariance_ & error_bound_include_overflows_;

    // initialize with a specific number of histograms
    init(nh);

    // for each hist, load it into thread 0
    for (unsigned hist_i = 0; hist_i < nh; hist_i++)
      ar & hists_[0][hist_i];
  }

}; // EECHistBase

} // namespace hist
} // namespace eec

#endif // EEC_HIST_HH