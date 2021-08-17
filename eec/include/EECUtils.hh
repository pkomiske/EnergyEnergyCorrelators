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

#include <string>

// OpenMP for multithreading
#ifdef _OPENMP
# include <omp.h>
#endif

// serialization code based on boost serialization
#ifdef EEC_SERIALIZATION
# include <boost/archive/binary_iarchive.hpp>
# include <boost/archive/binary_oarchive.hpp>
# include <boost/archive/text_iarchive.hpp>
# include <boost/archive/text_oarchive.hpp>
# include <boost/serialization/base_object.hpp>
# include <boost/serialization/array.hpp>
# include <boost/serialization/map.hpp>
# include <boost/serialization/string.hpp>
# include <boost/serialization/vector.hpp>
# include <boost/serialization/version.hpp>

// compression based on boost iostreams
# ifdef EEC_COMPRESSION
#  include <boost/iostreams/filtering_stream.hpp>
#  include <boost/iostreams/filter/zlib.hpp>
# endif
#endif // EEC_SERIALIZATION

// FastJet PseudoJet
#ifdef EEC_USE_PYFJCORE
# ifndef EVENTGEOMETRY_USE_PYFJCORE
#  define EVENTGEOMETRY_USE_PYFJCORE
# endif
#else
# include "fastjet/PseudoJet.hh"
#endif

// EventGeometry (for events and pairwise distances)
#include "EventGeometry.hh"

// include Wasserstein package in the proper namespace
#ifndef BEGIN_EEC_NAMESPACE
# define BEGIN_EEC_NAMESPACE namespace fastjet { namespace contrib { namespace eec {
# define END_EEC_NAMESPACE } } }
# define EEC_NAMESPACE fastjet::contrib::eec
#endif

BEGIN_EEC_NAMESPACE

// forward declarations
class EECBase;

//-----------------------------------------------------------------------------
// enums
//-----------------------------------------------------------------------------

// a standardized enum to select a ParticleWeight
enum class ParticleWeight : int {
  Default=-1, // TransverseMomentum in this package
  TransverseMomentum=0,
  Energy=1,
  TransverseEnergy=2,
  Momentum=3
};

// a standardized enum to select a PairwiseDistance
enum class PairwiseDistance : int {
  Default=-1, // DeltaR in this package
  DeltaR=0,
  HadronicDot=1,
  EEDot=2,
  HadronicDotMassive=3,
  EEDotMassless=4,
  EEArcLength=5,
  EEArcLengthMassive=6,
  EECosTheta=7,
  EECosThetaMassive=8
};

enum class ArchiveFormat { Text=0, Binary=1 };
enum class CompressionMode { Auto=0, Plain=1, Zlib=2 };

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
// Pairwise distances not found in EventGeometry
//-----------------------------------------------------------------------------

// EventGeometry-style pairwise distance yielding (cos(theta)/R)^beta
// note: this doesn't usually satisfy the triangle inequality
template<typename Value>
class EECosTheta : public eventgeometry::PairwiseDistanceBaseNonReduced<EECosTheta<Value>, std::vector<PseudoJet>, Value> {
public:
  typedef PseudoJet Particle;
  typedef std::vector<PseudoJet>::const_iterator ParticleIterator;

  EECosTheta(Value R, Value beta) :
    eventgeometry::PairwiseDistanceBaseNonReduced<EECosTheta, std::vector<PseudoJet>, Value>(R, beta)
  {}
  static std::string name() { return "EECosTheta"; }
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    return fastjet::cos_theta(p0, p1);
  }
}; // EECosTheta

// EventGeometry-style pairwise distance yielding (cos(theta)/R)^beta
// note: this doesn't usually satisfy the triangle inequality
template<typename Value>
class EECosThetaMassive : public eventgeometry::PairwiseDistanceBaseNonReduced<EECosThetaMassive<Value>, std::vector<PseudoJet>, Value> {
public:
  typedef PseudoJet Particle;
  typedef std::vector<PseudoJet>::const_iterator ParticleIterator;

  EECosThetaMassive(Value R, Value beta) :
    eventgeometry::PairwiseDistanceBaseNonReduced<EECosThetaMassive, std::vector<PseudoJet>, Value>(R, beta)
  {}
  static std::string name() { return "EECosThetaMassive"; }
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::min(1.0, std::max(-1.0, (p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.E()*p1.E())));
  }
}; // EECosTheta

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
inline void set_archive_format(ArchiveFormat a) {
  if (int(a) < 0 || int(a) > 1)
    throw std::invalid_argument("invalid archive format");

  archform_ = a;
}
inline void set_compression_mode(CompressionMode c) {

  if (int(c) < 0 || int(c) > 2)
    throw std::invalid_argument("invalid compression mode");

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

#ifndef SWIG_PREPROCESSOR

  // gets thread num if OpenMP is enabled, otherwise returns 0
  inline int get_thread_num() {
  #ifdef _OPENMP
    return omp_get_thread_num();
  #else
    return 0;
  #endif
  }

#endif // SWIG_PREPROCESSOR

inline std::string particle_weight_name(ParticleWeight pw) {
  switch (pw) {
    case ParticleWeight::Default:
    case ParticleWeight::TransverseMomentum:
      return "TransverseMomentum";
      break;

    case ParticleWeight::Energy:
      return "Energy";
      break;

    case ParticleWeight::TransverseEnergy:
      return "TransverseEnergy";
      break;

    case ParticleWeight::Momentum:
      return "Momentum";
      break;

    default:
      return "Unknown";
  }
}

inline std::string pairwise_distance_name(PairwiseDistance pd) {
  switch (pd) {
    case PairwiseDistance::Default:
    case PairwiseDistance::DeltaR:
      return "DeltaR";
      break;

    case PairwiseDistance::HadronicDot:
      return "HadronicDot";
      break;

    case PairwiseDistance::EEDot:
      return "EEDot";
      break;

    case PairwiseDistance::HadronicDotMassive:
      return "HadronicDotMassive";
      break;

    case PairwiseDistance::EEDotMassless:
      return "EEDotMassless";
      break;

    case PairwiseDistance::EEArcLength:
      return "EEArcLength";
      break;

    case PairwiseDistance::EEArcLengthMassive:
      return "EEArcLengthMassive";
      break;

    case PairwiseDistance::EECosTheta:
      return "EECosTheta";
      break;

    case PairwiseDistance::EECosThetaMassive:
      return "EECosThetaMassive";
      break;

    default:
      return "Unknown";
  }
}

END_EEC_NAMESPACE

#endif // EEC_UTILS_HH
