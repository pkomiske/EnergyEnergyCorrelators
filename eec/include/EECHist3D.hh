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
 *   _    _ _____  _____ _______ ____  _____  
 *  | |  | |_   _|/ ____|__   __|___ \|  __ \ 
 *  | |__| | | | | (___    | |    __) | |  | |
 *  |  __  | | |  \___ \   | |   |__ <| |  | |
 *  | |  | |_| |_ ____) |  | |   ___) | |__| |
 *  |_|  |_|_____|_____/   |_|  |____/|_____/     
 */

#ifndef EEC_HIST3D_HH
#define EEC_HIST3D_HH

#include <array>

#include "EECHistBase.hh"

namespace eec {
namespace hist {

#ifndef SWIG_PREPROCESSOR

// EECHistTraits for EECHist3D
template<class T0, class T1, class T2>
struct EECHistTraits<EECHist3D<T0, T1, T2>> {
  typedef T0 Transform0;
  typedef T1 Transform1;
  typedef T2 Transform2;
  typedef bh::axis::regular<double, Transform0> Axis0;
  typedef bh::axis::regular<double, Transform1> Axis1;
  typedef bh::axis::regular<double, Transform2> Axis2;

  static constexpr unsigned rank = 3;

  typedef struct HistFactory {
    static auto make_hist(const Axis0 & axis0, const Axis1 & axis1, const Axis2 & axis2) {
      return bh::make_histogram_with(bh::weight_storage(), axis0, axis1, axis2);
    }
    static auto make_simple_hist(const Axis0 & axis0, const Axis1 & axis1, const Axis2 & axis2) {
      return bh::make_histogram_with(simple_weight_storage(), axis0, axis1, axis2);
    }
    static auto make_covariance_hist(const Axis0 & axis0, const Axis1 & axis1, const Axis2 & axis2) {
      return bh::make_histogram_with(simple_weight_storage(), axis0, axis1, axis2, axis0, axis1, axis2);
    }
  } HistFactory;

  typedef decltype(HistFactory::make_hist(Axis0(), Axis1(), Axis2())) WeightedHist;
  typedef decltype(HistFactory::make_simple_hist(Axis0(), Axis1(), Axis2())) SimpleWeightedHist;
  typedef decltype(HistFactory::make_covariance_hist(Axis0(), Axis1(), Axis2())) CovarianceHist;
};

#endif // SWIG_PREPROCESSOR

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
            int num_threads = 1,
            bool track_covariance = false,
            bool variance_bound = true,
            bool variance_bound_include_overflows = true) :
    Base(num_threads, track_covariance, variance_bound, variance_bound_include_overflows),
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

  auto make_hist() const { return Traits::HistFactory::make_hist(axis0(), axis1(), axis2()); }
  auto make_simple_hist() const { return Traits::HistFactory::make_simple_hist(axis0(), axis1(), axis2()); }
  auto make_covariance_hist() const { return Traits::HistFactory::make_covariance_hist(axis0(), axis1(), axis2()); }
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

  Axis0 axis0() const { return Axis0(nbins_[0], axis_mins_[0], axis_maxs_[0]); }
  Axis1 axis1() const { return Axis1(nbins_[1], axis_mins_[1], axis_maxs_[1]); }
  Axis2 axis2() const { return Axis2(nbins_[2], axis_mins_[2], axis_maxs_[2]); }

#ifdef BOOST_SERIALIZATION_ACCESS_HPP
  friend class boost::serialization::access;
#endif

  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */) {
    ar & nbins_ & axis_mins_ & axis_maxs_;
    ar & boost::serialization::base_object<Base>(*this);
  }

}; // EECHist3D

} // namespace hist
} // namespace eec

#endif // EEC_HIST3D_HH
