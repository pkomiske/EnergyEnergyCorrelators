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
 *   _    _ _____  _____ _______ __ _____  
 *  | |  | |_   _|/ ____|__   __/_ |  __ \ 
 *  | |__| | | | | (___    | |   | | |  | |
 *  |  __  | | |  \___ \   | |   | | |  | |
 *  | |  | |_| |_ ____) |  | |   | | |__| |
 *  |_|  |_|_____|_____/   |_|   |_|_____/ 
 */

#ifndef EEC_HIST1D_HH
#define EEC_HIST1D_HH

#include "EECHistBase.hh"

namespace eec {
namespace hist {

#ifndef SWIG_PREPROCESSOR

// EECHistTraits for EECHist1D
template<class T>
struct EECHistTraits<EECHist1D<T>> {
  typedef T Transform;
  typedef bh::axis::regular<double, Transform> Axis;

  static constexpr unsigned rank = 1;

  typedef struct HistFactory {
    static auto make_hist(const Axis & axis) {
      return bh::make_histogram_with(bh::weight_storage(), axis);
    }
    static auto make_simple_hist(const Axis & axis) {
      return bh::make_histogram_with(simple_weight_storage(), axis);
    }
    static auto make_covariance_hist(const Axis & axis) {
      return bh::make_histogram_with(simple_weight_storage(), axis, axis);
    }
  } HistFactory;

  typedef decltype(HistFactory::make_hist(Axis())) WeightedHist;
  typedef decltype(HistFactory::make_simple_hist(Axis())) SimpleWeightedHist;
  typedef decltype(HistFactory::make_covariance_hist(Axis())) CovarianceHist;
};

#endif // SWIG_PREPROCESSOR

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

  EECHist1D(unsigned nbins, double axis_min, double axis_max,
            int num_threads = 1,
            bool track_covariance = true,
            bool variance_bound = true, 
            bool variance_bound_include_overflows = true) :
    Base(num_threads, track_covariance, variance_bound, variance_bound_include_overflows),
    nbins_(nbins), axis_min_(axis_min), axis_max_(axis_max)
  {
    this->init(1);
  }
  virtual ~EECHist1D() = default;

#ifndef SWIG_PREPROCESSOR
  void reset_axes() {
    nbins_ = this->nbins();
    axis_min_ = this->axis().value(0);
    axis_max_ = this->axis().value(nbins_);
  }

  auto make_hist() const { return Traits::HistFactory::make_hist(axis0()); }
  auto make_simple_hist() const { return Traits::HistFactory::make_simple_hist(axis0()); }
  auto make_covariance_hist() const { return Traits::HistFactory::make_covariance_hist(axis0()); }
#endif

protected:

  std::string axes_description() const { return name_transform<Tr>(); }

private:

  Axis axis0() const { return Axis(nbins_, axis_min_, axis_max_); }

#ifdef BOOST_SERIALIZATION_ACCESS_HPP
  friend class boost::serialization::access;
#endif

  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & nbins_ & axis_min_ & axis_max_;
    ar & boost::serialization::base_object<Base>(*this);
  }

}; // EECHist1D

} // namespace hist
} // namespace eec

#endif // EEC_HIST1D_HH
