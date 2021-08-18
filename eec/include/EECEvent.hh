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
 *   ________      ________ _   _ _______
 *  |  ____\ \    / /  ____| \ | |__   __|
 *  | |__   \ \  / /| |__  |  \| |  | |
 *  |  __|   \ \/ / |  __| | . ` |  | |
 *  | |____   \  /  | |____| |\  |  | |
 *  |______|   \/   |______|_| \_|  |_|
 */

#ifndef EEC_EVENT_HH
#define EEC_EVENT_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "EECConfig.hh"

BEGIN_EEC_NAMESPACE

//-----------------------------------------------------------------------------
// Particle weights (similar to EventGeometry)
//-----------------------------------------------------------------------------

#ifndef SWIG_PREPROCESSOR

struct TransverseMomentum {
  static double weight(const PseudoJet & pj) { return pj.pt(); }
};
struct TransverseEnergy {
  static double weight(const PseudoJet & pj) { return pj.Et(); }
};
struct Energy {
  static double weight(const PseudoJet & pj) { return pj.E(); }
};
struct Momentum {
  static double weight(const PseudoJet & pj) { return pj.modp(); }
};

//-----------------------------------------------------------------------------
// Pairwise distances (similar to EventGeometry)
//-----------------------------------------------------------------------------

// Hadronic Delta_R measure with proper checking for phi
struct DeltaR {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    double dphiabs(std::fabs(p0.phi() - p1.phi()));
    double dy(p0.rap() - p1.rap()), dphi(dphiabs > PI ? TWOPI - dphiabs : dphiabs);
    return std::sqrt(dy*dy + dphi*dphi);
  }
};

// Dot product measure normalized with transverse momenta
struct HadronicDot {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::sqrt(std::max(2*fastjet::dot_product(p0, p1) / std::sqrt(p0.pt2()*p1.pt2()), 0.0));
  }  
};

// Dot product measure normalized by energy
struct EEDot {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::sqrt(std::max(2*fastjet::dot_product(p0, p1) / (p0.E()*p1.E()), 0.0));
  }
};

// Massive dot product measure normalized with transverse energies
struct HadronicDotMassive {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::sqrt(std::max(2*fastjet::dot_product(p0, p1) / std::sqrt(p0.Et2()*p1.Et2()), 0.0));
  }
};

// Massive dot product measure normalized with energies
struct EEDotMassless {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::sqrt(std::max(2*fastjet::dot_product(p0, p1) / std::sqrt(p0.modp2()*p1.modp2()), 0.0));
  }
};

// Arc length between momentum vectors
struct EEArcLength {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return fastjet::theta(p0, p1);
  }
};

// Arc length between momentum vectors, normalized by the energy
struct EEArcLengthMassive {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::acos(std::min(1.0, std::max(-1.0, (p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.E()*p1.E()))));
  }
};

// note: this doesn't usually satisfy the triangle inequality
struct EECosTheta {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return fastjet::cos_theta(p0, p1);
  }
};

// note: this doesn't usually satisfy the triangle inequality
struct EECosThetaMassive {
  static double distance(const PseudoJet & p0, const PseudoJet & p1) {
    return std::min(1.0, std::max(-1.0, (p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.E()*p1.E())));
  }
};

#endif // !SWIG_PREPROCESSOR

//-----------------------------------------------------------------------------
// Class for representing an event to the EEC computation
//-----------------------------------------------------------------------------

class EECEvent {
private:

  enum LazyInitType { None, Array, Custom };

  // actual event data
  std::vector<std::vector<double>> weights_;
  std::vector<double> dists_;
  double event_weight_;
  unsigned mult_;

  // internal data
  LazyInitType lazy_init_type_;
  std::array<const double *, 3> ptrs_;

public:

  // default constructor
  EECEvent() : event_weight_(0), mult_(0), lazy_init_type_(None) {}

  // construct from vector of pseudojets and vector of charges
  EECEvent(const EECConfig & config,
           double event_weight,
           const std::vector<PseudoJet> & pjs,
           const std::vector<double> & charges) :
    event_weight_(event_weight),
    mult_(pjs.size()),
    lazy_init_type_(None)
  {
    // process weights
    std::vector<double> raw_weights;
    switch (config.particle_weight) {
      case ParticleWeight::TransverseMomentum:
        raw_weights = get_raw_weights<TransverseMomentum>(pjs);
        break;

      case ParticleWeight::Energy:
        raw_weights = get_raw_weights<Energy>(pjs);
        break;

      case ParticleWeight::TransverseEnergy:
        raw_weights = get_raw_weights<TransverseEnergy>(pjs);
        break;

      case ParticleWeight::Momentum:
        raw_weights = get_raw_weights<Momentum>(pjs);
        break;

      default:
        throw std::invalid_argument("invalid particle weight");
    }
    process_weights(config, raw_weights);

    // process charges
    if (config.use_charges && charges.size() != mult())
      throw std::invalid_argument("length of charges must match number of particles");

    if (!config.use_charges && charges.size() != 0)
      throw std::invalid_argument("charges present when not being used");

    process_charges(config, charges.data());

    // process distances
    switch (config.pairwise_distance) {
      case PairwiseDistance::DeltaR:
        fill_distances<DeltaR>(config, pjs);
        break;

      case PairwiseDistance::HadronicDot:
        fill_distances<HadronicDot>(config, pjs);
        break;

      case PairwiseDistance::EEDot:
        fill_distances<EEDot>(config, pjs);
        break;

      case PairwiseDistance::HadronicDotMassive:
        fill_distances<HadronicDotMassive>(config, pjs);
        break;

      case PairwiseDistance::EEDotMassless:
        fill_distances<EEDotMassless>(config, pjs);
        break;

      case PairwiseDistance::EEArcLength:
        fill_distances<EEArcLength>(config, pjs);
        break;

      case PairwiseDistance::EEArcLengthMassive:
        fill_distances<EEArcLengthMassive>(config, pjs);
        break;

      case PairwiseDistance::EECosTheta:
        fill_distances<EECosTheta>(config, pjs);
        break;

      case PairwiseDistance::EECosThetaMassive:
        fill_distances<EECosThetaMassive>(config, pjs);
        break;

      default:
        throw std::invalid_argument("invalid pairwise distance");
    }

    // check degen
    check_degeneracy(config);
  }

#ifndef SWIG

  // construct from explicit raw weights, distances, and charges (as vectors)
  EECEvent(const EECConfig & config,
           double event_weight,
           const std::vector<double> & raw_weights,
           const std::vector<double> & dists,
           const std::vector<double> & charges) :
    event_weight_(event_weight),
    lazy_init_type_(None)
  {
    if (dists.size() != raw_weights.size()*raw_weights.size())
      throw std::invalid_argument("dists must be size mult*mult");

    if (config.use_charges && raw_weights.size() != charges.size())
      throw std::invalid_argument("lenght of charges must match the length of weights");

    mult_ = raw_weights.size();
    array_init(config, raw_weights.data(), dists.data(), charges.data());
  }

// SWIG is defined
#else

  // construct from pointer to array of particles using euclidean distance between the coordinates
  EECEvent(bool use_charges, double event_weight,
           const double * event_ptr, unsigned mult, unsigned nfeatures) :
    event_weight_(event_weight),
    mult_(mult),
    lazy_init_type_(Array),
    ptrs_{event_ptr, nullptr, nullptr}
  {
    if (use_charges && nfeatures != 4)
      throw std::invalid_argument("nfeatures should be 4 when using charges");
    else if (nfeatures != 3)
      std::invalid_argument("nfeatures should be greater than 3 if not using charges");
  }

  // construct from explicit raw weights, distances, and charges (as pointers)
  EECEvent(bool use_charges, double event_weight,
           const double * raw_weights, unsigned weights_mult,
           const double * charges, unsigned charges_mult,
           const double * dists, unsigned d0, unsigned d1) :
    event_weight_(event_weight),
    mult_(weights_mult),
    lazy_init_type_(Custom),
    ptrs_{raw_weights, dists, charges}
  {
    if (use_charges && charges_mult != 0)
      throw std::invalid_argument("charges present when not being used");

    if (charges_mult != 0 && weights_mult != charges_mult)
      throw std::invalid_argument("weights and charges should be the same length");

    if (d0 != d1 || weights_mult != d0)
      throw std::invalid_argument("dists should be size (mult, mult)");
  }

#endif // SWIG

  virtual ~EECEvent() = default;

  // access functions
  const std::vector<std::vector<double>> & weights() const { return weights_; }
  const std::vector<double> & dists() const { return dists_; }
  double event_weight() const { return event_weight_; }
  unsigned mult() const { return mult_; }

private:

  // function to extract weights from vector of PseudoJets
  template<class ParticleWeight>
  static std::vector<double> get_raw_weights(const std::vector<PseudoJet> & pjs) {

    std::vector<double> raw_weights;
    raw_weights.reserve(pjs.size());

    for (const PseudoJet & pj : pjs)
      raw_weights.push_back(ParticleWeight::weight(pj));

    return raw_weights;
  }

  // set weights according to raw_weights and weight_powers
  void process_weights(const EECConfig & config,
                       std::vector<double> & raw_weights) {

    // normalize weights if requested
    if (config.norm) {
      double weight_total(0);
      for (double w : raw_weights)
        weight_total += w;

      for (double & w : raw_weights)
        w /= weight_total;
    }

    // set internal weights according to raw weights and weight_powers
    weights_.resize(config.weight_powers.size());
    for (unsigned i = 0; i < config.weight_powers.size(); i++) {
      weights_[i].resize(mult());

      // set weights[i][j] to weight[j]^weight_power[i]
      if (config.weight_powers[i] == 1)
        std::copy(raw_weights.begin(), raw_weights.end(), weights_[i].begin());
      else
        for (unsigned j = 0; j < mult(); j++)
          weights_[i][j] = std::pow(raw_weights[j], config.weight_powers[i]);
    }
  }

  // multiply weights[i][j] by charge[j]^charge_power[i]
  void process_charges(const EECConfig & config, const double * charges) {
    if (config.use_charges)
      for (unsigned i = 0; i < config.charge_powers.size(); i++)
        if (config.charge_powers[i] != 0)
          for (unsigned j = 0; j < mult(); j++)
            weights_[i][j] *= std::pow(charges[j], config.charge_powers[i]);
  }

  // fill distances from vector of pseudojets
  template<class PairwiseDistance>
  void fill_distances(const EECConfig & config, const std::vector<PseudoJet> & pjs) {
    dists_.resize(mult()*mult());

    for (unsigned i = 0; i < mult(); i++) {
      unsigned ixm(i*mult());
      dists_[ixm + i] = 0;

      for (unsigned j = 0; j < i; j++)
        dists_[ixm + j] = dists_[j*mult() + i] = PairwiseDistance::distance(pjs[i], pjs[j]);
    }
  }

  // check for degeneracy among pairwise distances (should be a "measure zero" occurrence)
  // the existence of a degeneracy breaks assumptions made in the EEC computations
  void check_degeneracy(const EECConfig & config) const {
    if (!config.check_degen) return;

    std::unordered_set<double> dists_set;
    unsigned ndegen(0);
    for (unsigned i = 0; i < mult(); i++) {
      unsigned ixm(i*mult());
      for (unsigned j = 0; j < i; j++) {
        auto x(dists_set.insert(dists()[ixm + j]));
        if (!x.second) {
          if (ndegen++ == 0)
            std::cerr << "Begin Event\n";
          std::cerr << "  distance degeneracy encountered, particles " 
                    << i << " and " << j << ", distance is " << *x.first << std::endl;
        }
      }
    }

    if (ndegen > 0)
      std::cerr << "End Event\n";
  }

  void array_init(const EECConfig & config,
                  const double * raw_weights,
                  const double * dists,
                  const double * charges) {

    // process weights
    std::vector<double> raw_weights_vec(raw_weights, raw_weights + mult());
    process_weights(config, raw_weights_vec);

    // process charges
    process_charges(config, charges);

    // process dists according to R and beta
    dists_.resize(mult()*mult());
    std::copy(dists, dists + dists_.size(), dists_.begin());

    // check degen
    check_degeneracy(config);
  }

  // allows EECBase to access lazy_init
  friend class EECBase;

  // lazy initialization for some of the construction methods
  void lazy_init(const EECConfig & config) {

    // vector means already initialized
    if (lazy_init_type_ == None) return;

    if (lazy_init_type_ == Custom)
      array_init(config, ptrs_[0], ptrs_[1], ptrs_[2]);

    // lazy_init_type_ == Array
    else {
      const double * arr(ptrs_[0]);

      std::vector<double> raw_weights(mult());
      dists_.resize(mult()*mult());
      for (unsigned i = 0; i < mult(); i++) {
        unsigned ixm(i*mult()), ixnf(i*config.nfeatures);

        // store weight
        raw_weights[i] = arr[ixnf];

        // zero out diagonal
        dists_[ixm + i] = 0;

        double y_i(arr[ixnf + 1]), phi_i(arr[ixnf + 2]);
        for (unsigned j = 0; j < i; j++) {
          unsigned jxnf(j*config.nfeatures);
          double ydiff(y_i - arr[jxnf + 1]), phidiff(std::fabs(phi_i - arr[jxnf + 2]));
          if (phidiff > PI) phidiff = TWOPI - phidiff;

          dists_[ixm + j] = dists_[j*mult() + i] = std::sqrt(ydiff*ydiff + phidiff*phidiff);
        }
      }

      std::vector<double> charges;
      if (config.use_charges) {
        charges.resize(mult());
        for (unsigned i = 0; i < mult(); i++)
          charges[i] = arr[i*config.nfeatures + 3];
      }

      // process weights and charges
      process_weights(config, raw_weights);
      process_charges(config, charges.data());

      // check degen
      check_degeneracy(config);
    }
  }

}; // EECEvent

END_EEC_NAMESPACE

#endif // EEC_EVENT_HH
