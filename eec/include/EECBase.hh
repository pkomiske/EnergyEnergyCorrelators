// EnergyEnergyCorrelators - Evaluates EECs on particle physics events
// Copyright (C) 2020 Patrick T. Komiske III
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

#ifndef EEC_BASE_HH
#define EEC_BASE_HH

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace eec {

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------

const double REG = 1e-100;
const double PI = 3.14159265358979323846;
const double TWOPI = 6.28318530717958647693;

//-----------------------------------------------------------------------------
// Base class for all EEC computations
//-----------------------------------------------------------------------------

class EECBase {
protected:

  // the pairwise distances, pts, and powers of pt
  std::vector<double> dists_, pts_, charges_, orig_pt_powers_, pt_powers_;
  std::vector<unsigned> orig_ch_powers_, ch_powers_;

  // number of particles to correlate, features per particle, unique weights to compute
  // 3 by default, charge optional (pt, y, phi, [charge])
  unsigned N_, nfeatures_, nsym_;
  bool norm_, use_charges_, check_degen_, average_verts_;

  // vector of weights for each index
  std::vector<std::vector<double>> weights_;

  // name of method used for core computation
  std::string compname_;

  // the current weight of the event
  double weight_;

  // the current multiplicity of the event
  unsigned mult_;

  #ifdef __FASTJET_PSEUDOJET_HH__
  double (*pj_charge_)(const fastjet::PseudoJet &);
  #endif

public:

  EECBase(const std::vector<double> & pt_powers, const std::vector<unsigned> & ch_powers,
          unsigned N, bool norm, bool check_degen, bool average_verts) : 
    orig_pt_powers_(pt_powers), orig_ch_powers_(ch_powers),
    N_(N), nsym_(N_),
    norm_(norm), use_charges_(false), check_degen_(check_degen), average_verts_(average_verts),
    weights_(N_)
  {
    if (orig_pt_powers_.size() == 1)
      orig_pt_powers_ = std::vector<double>(N_, orig_pt_powers_[0]);
    if (orig_ch_powers_.size() == 1)
      orig_ch_powers_ = std::vector<unsigned>(N_, orig_ch_powers_[0]);

    if (orig_pt_powers_.size() != N_)
      throw std::invalid_argument("pt_powers must be a vector of size 1 or " + std::to_string(N_));
    if (orig_ch_powers_.size() != N_)
      throw std::invalid_argument("ch_powers must be a vector of size 1 or " + std::to_string(N_));

    // copy original powers, because these will be modified by how many symmetries there are
    pt_powers_ = orig_pt_powers_;
    ch_powers_ = orig_ch_powers_;

    // check for symmetries in the different cases
    switch (N_) {
      case 2: {
        if (pt_powers_[0] == pt_powers_[1] && ch_powers_[0] == ch_powers_[1]) {
          pt_powers_ = {pt_powers_[0]};
          ch_powers_ = {ch_powers_[0]};
          compname_ = "eec_ij_sym";
        }
        else {
          compname_ = "eec_no_sym";
          nsym_ = 0;
        }
        break;
      }

      case 3: {
        bool ptpowmatch01(pt_powers_[0] == pt_powers_[1]), chpowmatch01(ch_powers_[0] == ch_powers_[1]),
             ptpowmatch12(pt_powers_[1] == pt_powers_[2]), chpowmatch12(ch_powers_[1] == ch_powers_[2]),
             ptpowmatch02(pt_powers_[0] == pt_powers_[2]), chpowmatch02(ch_powers_[0] == ch_powers_[2]);
        if (ptpowmatch01 && ptpowmatch12 && chpowmatch01 && chpowmatch12) {
          pt_powers_ = {pt_powers_[0]};
          ch_powers_ = {ch_powers_[0]};
          compname_ = "eeec_ijk_sym";
        }
        else if (ptpowmatch01 && chpowmatch01 && !average_verts_) {
          pt_powers_ = {pt_powers_[0], pt_powers_[2]};
          ch_powers_ = {ch_powers_[0], ch_powers_[2]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else if (ptpowmatch12 && chpowmatch12 && !average_verts_) {
          pt_powers_ = {pt_powers_[1], pt_powers_[0]};
          ch_powers_ = {ch_powers_[1], ch_powers_[0]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else if (ptpowmatch02 && chpowmatch02 && !average_verts_) {
          pt_powers_ = {pt_powers_[2], pt_powers_[1]};
          ch_powers_ = {ch_powers_[2], ch_powers_[1]};
          compname_ = "eeec_ij_sym";
          nsym_ = 2;
        }
        else {
          compname_ = "eeec_no_sym";
          nsym_ = 0;
        }
        break;
      }

      case 4: {
        for (int i = 1; i < 4; i++) {
          if (pt_powers_[i] != pt_powers_[0] || ch_powers_[i] != ch_powers_[0])
            throw std::invalid_argument("N = 4 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeec_ijkl_sym";
        break;
      }

      case 5: {
        for (int i = 1; i < 5; i++) {
          if (pt_powers_[i] != pt_powers_[0] || ch_powers_[i] != ch_powers_[0])
            throw std::invalid_argument("N = 5 only supports the fully symmetric correlator currently");
        }
        compname_ = "eeec_ijklm_sym";
        break;
      }

      default:
        throw std::invalid_argument("N must be 2, 3, 4, or 5 currently");
    }

    // check for using charges at all
    for (int ch_power : ch_powers_)
      if (ch_power != 0)
        use_charges_ = true;
    nfeatures_ = (use_charges_ ? 4 : 3);

    // initialize this to null in case we have FastJet support
    #ifdef __FASTJET_PSEUDOJET_HH__
    pj_charge_ = nullptr;
    #endif   
  }

  virtual ~EECBase() {}

  virtual std::string description() const {
    std::ostringstream oss;
    oss << "  N - " << N_ << '\n'
        << "  norm - " << (norm_ ? "true" : "false") << '\n'
        << "  use_charges - " << (use_charges_ ? "true" : "false") << '\n'
        << "  nfeatures - " << nfeatures_ << '\n'
        << "  check_for_degeneracy - " << (check_degen_ ? "true" : "false") << '\n'
        << "  average_verts - " << (average_verts_ ? "true" : "false");

    // record pt and charge powers
    oss << '\n'
        << "  pt_powers - (" << pt_powers_[0];
    for (unsigned i = 1; i < orig_pt_powers_.size(); i++)
      oss << ", " << orig_pt_powers_[i];
    oss << ")\n"
        << "  ch_powers - (" << ch_powers_[0];
    for (unsigned i = 1; i < orig_ch_powers_.size(); i++)
      oss << ", " << orig_ch_powers_[i];
    oss << ")\n";

    return oss.str();
  }

  // compute on a vector of particles
  void compute(const std::vector<double> & particles, double weight = 1.0) {
    assert(particles.size() % nfeatures_ == 0);
    compute(particles.data(), particles.size()/nfeatures_, weight);
  }

  // compute on a C array of particles
  void compute(const double * particles, unsigned mult, double weight = 1.0) {

    // store event weight and multiplicity
    weight_ = weight;
    mult_ = mult;
    
    // compute pairwise distances and extract pts
    dists_.resize(mult_*mult_);
    pts_.resize(mult_);
    for (unsigned i = 0; i < mult_; i++) {
      unsigned ixm(i*mult_), ixnf(i*nfeatures_);

      // store pt
      pts_[i] = particles[ixnf];

      // zero out diagonal
      dists_[ixm + i] = 0;

      double y_i(particles[ixnf + 1]), phi_i(particles[ixnf + 2]);
      for (unsigned j = 0; j < i; j++) {
        unsigned jxnf(j*nfeatures_);
        double ydiff(y_i - particles[jxnf + 1]), 
               phidiffabs(std::fabs(phi_i - particles[jxnf + 2]));

        // ensure that the phi difference is properly handled
        if (phidiffabs > PI) phidiffabs -= TWOPI;
        dists_[ixm + j] = dists_[j * mult_ + i] = std::sqrt(ydiff*ydiff + phidiffabs*phidiffabs);
      }
    }

    // store charges
    if (use_charges_) {
      charges_.resize(mult_);
      for (unsigned i = 0; i < mult_; i++)
        charges_[i] = particles[i*nfeatures_ + 3];
    }

    // check for degeneracy
    if (check_degen_) {
      std::unordered_set<double> dists_set;
      for (unsigned i = 0; i < mult_; i++) {
        unsigned ixm(i*mult_);
        for (unsigned j = 0; j < i; j++) {
          auto x(dists_set.insert(dists_[ixm + j]));
          if (!x.second) {
            std::cerr << "distance degeneracy encountered, particles " << i << " and " << j << ", distance is " << *x.first << std::endl;
            //throw std::runtime_error("distance degeneracy encountered, particles " 
            //                         + std::to_string(i) + " and " + std::to_string(j) + ", distance is " + std::to_string(*x.first));
          }
        }
      }
    }

    // run actual computation
    else {

      // fill weights with powers of pts and charges
      set_weights();

      // delegate EEC computation to subclass
      compute_eec();
    }
  }

  // fastjet support
  #ifdef __FASTJET_PSEUDOJET_HH__
  void set_pseudojet_charge_func(double (*pj_charge)(const fastjet::PseudoJet &)) {
    pj_charge_ = pj_charge;
  }

  void compute(const std::vector<fastjet::PseudoJet> & pjs, const double weight = 1.0) {

    weight_ = weight;
    mult_ = pjs.size();

    for (unsigned i = 0; i < mult_; i++) {
      unsigned ixm(i*mult_);
      pts_[i] = pjs[i].pt();
      dists_[ixm + i] = 0;

      for (unsigned j = 0; j < i; j++)
        dists_[ixm + j] = dists_[j * mult_ + i] = pjs[i].delta_R(pjs[j]);
    }

    // store charges if provided a function to do so
    if (pj_charge_ != nullptr) {
      charges_.resize(pjs.size());
      for (unsigned i = 0; i < mult_; i++)
        charges_[i] = pj_charge_(pjs[i]);  
    }

    // fill weights with powers of pts and charges
    set_weights();

    // delegate EEC computation to subclass
    compute_eec();
  }
  #endif // __FASTJET_PSEUDOJET_HH__

protected:

  // method that carries out the specific EEC computation
  virtual void compute_eec() = 0;

  // get weights as powers of pts and charges
  void set_weights() {

    // normalize pts
    if (norm_) {
      double pttot(0);
      for (double pt : pts_) pttot += pt;
      for (double & pt : pts_) pt /= pttot;
    }

    for (unsigned i = 0, npowers = pt_powers_.size(); i < npowers; i++) {
      std::vector<double> & weights(weights_[i]);
      weights.resize(mult_);

      // set weights to pt^ptpower
      double pt_power(pt_powers_[i]);
      if (pt_power == 1)
        for (unsigned j = 0; j < mult_; j++)
            weights[j] = pts_[j];
      else if (pt_power == 2)
        for (unsigned j = 0; j < mult_; j++)
            weights[j] = pts_[j]*pts_[j];
      else if (pt_power == 0.5)
        for (unsigned j = 0; j < mult_; j++)
            weights[j] = std::sqrt(pts_[j]);
      else
        for (unsigned j = 0; j < mult_; j++)
            weights[j] = std::pow(pts_[j], pt_power);

      // include effect of charge
      if (use_charges_) {
        double ch_power(ch_powers_[i]);
        if (ch_power == 0) {}
        else if (ch_power == 1)
          for (unsigned j = 0; j < mult_; j++)
            weights[j] *= charges_[j];
        else if (ch_power == 2)
          for (unsigned j = 0; j < mult_; j++)
            weights[j] *= charges_[j]*charges_[j];
        else
          for (unsigned j = 0; j < mult_; j++)
            weights[j] *= std::pow(charges_[j], ch_power);
      }
    }
  }

}; // EECBase

} // namespace eec

#endif // EEC_BASE_HH