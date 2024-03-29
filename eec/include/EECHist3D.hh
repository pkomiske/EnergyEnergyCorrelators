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

#include "EECHistBase.hh"

BEGIN_EEC_NAMESPACE
namespace hist {

// EECHistTraits for EECHist3D
template<class Tr0, class Tr1, class Tr2>
struct EECHistTraits<EECHist3D<Tr0, Tr1, Tr2>> {

  static constexpr unsigned rank = 3;

  typedef bh::axis::regular<double, Tr0> Axis0;
  typedef bh::axis::regular<double, Tr1> Axis1;
  typedef bh::axis::regular<double, Tr2> Axis2;

  typedef std::array<unsigned, rank> NBins;
  typedef std::array<std::array<double, 2>, rank> AxesRange;

#ifndef SWIG_PREPROCESSOR
  static auto make_hist(const NBins & nbins, const AxesRange & axes_range) {
    return bh::make_histogram_with(bh::weight_storage(),
                                   Axis0(nbins[0], axes_range[0][0], axes_range[0][1]),
                                   Axis1(nbins[1], axes_range[1][0], axes_range[1][1]),
                                   Axis2(nbins[2], axes_range[2][0], axes_range[2][1]));
  }
  static auto make_simple_hist(const NBins & nbins, const AxesRange & axes_range) {
    return bh::make_histogram_with(simple_weight_storage(),
                                   Axis0(nbins[0], axes_range[0][0], axes_range[0][1]),
                                   Axis1(nbins[1], axes_range[1][0], axes_range[1][1]),
                                   Axis2(nbins[2], axes_range[2][0], axes_range[2][1]));
  }
  static auto make_covariance_hist(const NBins & nbins, const AxesRange & axes_range) {
    return bh::make_histogram_with(simple_weight_storage(),
                                   Axis0(nbins[0], axes_range[0][0], axes_range[0][1]),
                                   Axis1(nbins[1], axes_range[1][0], axes_range[1][1]),
                                   Axis2(nbins[2], axes_range[2][0], axes_range[2][1]),
                                   Axis0(nbins[0], axes_range[0][0], axes_range[0][1]),
                                   Axis1(nbins[1], axes_range[1][0], axes_range[1][1]),
                                   Axis2(nbins[2], axes_range[2][0], axes_range[2][1]));
  }
#endif // !SWIG_PREPROCESSOR

};

//-----------------------------------------------------------------------------
// 3D histogram class
//-----------------------------------------------------------------------------

template<class Tr0, class Tr1, class Tr2>
class EECHist3D : public EECHistBase<EECHist3D<Tr0, Tr1, Tr2>> {
public:

#ifndef SWIG_PREPROCESSOR

  // inherited constructor
  using EECHistBase<EECHist3D<Tr0, Tr1, Tr2>>::EECHistBase;

#endif

  virtual ~EECHist3D() = default;

  static std::string hist_name() { return "EECHist3D"; }
  static std::string axes_description() {
    std::ostringstream oss;
    oss << name_transform(Tr0()) << ", "
        << name_transform(Tr1()) << ", "
        << name_transform(Tr2());
    return oss.str();
  }

private:

  #ifdef BOOST_SERIALIZATION_ACCESS_HPP
    friend class boost::serialization::access;
    BOOST_SERIALIZATION_SPLIT_MEMBER()
  #endif

  #ifdef EEC_SERIALIZATION
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const {
      ar & boost::serialization::base_object<EECHistBase<EECHist3D<Tr0, Tr1, Tr2>>>(*this);
    }

    template<class Archive>
    void load(Archive & ar, const unsigned int version) {
      if (version == 0) {
        std::array<double, 3> axis_mins, axis_maxs;
        ar & this->nbins_ & axis_mins & axis_maxs;
        for (unsigned i = 0; i < 3; i++)
          this->axes_range_[i] = {axis_mins[i], axis_maxs[i]};
      }

      ar & boost::serialization::base_object<EECHistBase<EECHist3D<Tr0, Tr1, Tr2>>>(*this);
    }
  #endif // EEC_SERIALIZATION

}; // EECHist3D

// aliases
using EECHist3DIdIdId = EECHist3D<axis::id, axis::id, axis::id>;
using EECHist3DLogIdId = EECHist3D<axis::log, axis::id, axis::id>;
using EECHist3DIdLogId = EECHist3D<axis::id, axis::log, axis::id>;
using EECHist3DLogLogId = EECHist3D<axis::log, axis::log, axis::id>;

} // namespace hist
END_EEC_NAMESPACE

#define EEC_HIST3D_SERIALIZATION(T) \
  

#if !defined(SWIG_PREPROCESSOR) && defined(EEC_SERIALIZATION)
  EEC_HIST_SERIALIZATION(EECHist3DIdIdId, 1)
  EEC_HIST_SERIALIZATION(EECHist3DLogIdId, 1)
  EEC_HIST_SERIALIZATION(EECHist3DIdLogId, 1)
  EEC_HIST_SERIALIZATION(EECHist3DLogLogId, 1)
#endif

#endif // EEC_HIST3D_HH
