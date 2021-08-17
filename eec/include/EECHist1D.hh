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

BEGIN_EEC_NAMESPACE
namespace hist {



// EECHistTraits for EECHist1D
template<class Transform>
struct EECHistTraits<EECHist1D<Transform>> {

  static constexpr unsigned rank = 1;

  typedef bh::axis::regular<double, Transform> Axis;

  typedef std::array<unsigned, rank> NBins;
  typedef std::array<std::array<double, 2>, rank> AxesRange;

#ifndef SWIG_PREPROCESSOR
  static auto make_hist(const NBins & nbins, const AxesRange & axes_range) {
    return bh::make_histogram_with(bh::weight_storage(),
                                   Axis(nbins[0], axes_range[0][0], axes_range[0][1]));
  }
  static auto make_simple_hist(const NBins & nbins, const AxesRange & axes_range) {
    return bh::make_histogram_with(simple_weight_storage(),
                                   Axis(nbins[0], axes_range[0][0], axes_range[0][1]));
  }
  static auto make_covariance_hist(const NBins & nbins, const AxesRange & axes_range) {
    return bh::make_histogram_with(simple_weight_storage(),
                                   Axis(nbins[0], axes_range[0][0], axes_range[0][1]),
                                   Axis(nbins[0], axes_range[0][0], axes_range[0][1]));
  }
#endif // !SWIG_PREPROCESSOR

  static std::string axes_description() { return name_transform<Transform>(); }

};

//-----------------------------------------------------------------------------
// 1D histogram class
//-----------------------------------------------------------------------------

template<class Transform>
class EECHist1D : public EECHistBase<EECHist1D<Transform>> {
public:

#ifndef SWIG_PREPROCESSOR

  // inherited constructor
  using EECHistBase<EECHist1D<Transform>>::EECHistBase;

#endif

  virtual ~EECHist1D() = default;

private:

  #ifdef BOOST_SERIALIZATION_ACCESS_HPP
    friend class boost::serialization::access;
  #endif

  #ifdef EEC_SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      std::cout << "EECHist1D::serialize, version " << version << std::endl;
      if (version == 0)
        ar & this->nbins_[0] & this->axes_range_[0][0] & this->axes_range_[0][1];
      
      ar & boost::serialization::base_object<EECHistBase<EECHist1D<Transform>>>(*this);
      std::cout << "EECHist1D::done" << std::endl;
    }
  #endif

}; // EECHist1D

// aliases
using EECHist1DId = EECHist1D<axis::id>;
using EECHist1DLog = EECHist1D<axis::log>;

} // namespace hist
END_EEC_NAMESPACE

#if !defined(SWIG_PREPROCESSOR) && defined(EEC_SERIALIZATION)
  BOOST_CLASS_VERSION(EEC_NAMESPACE::hist::EECHist1DId, 1)
  BOOST_CLASS_VERSION(EEC_NAMESPACE::hist::EECHist1DLog, 1)
  EEC_HISTBASE_SERIALIZATION(EEC_NAMESPACE::hist::EECHist1DId)
  EEC_HISTBASE_SERIALIZATION(EEC_NAMESPACE::hist::EECHist1DLog)
#endif

#endif // EEC_HIST1D_HH
