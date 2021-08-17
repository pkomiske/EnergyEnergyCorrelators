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
 *    _____ ____  _   _ ______ _____ _____
 *   / ____/ __ \| \ | |  ____|_   _/ ____|
 *  | |   | |  | |  \| | |__    | || |  __
 *  | |   | |  | | . ` |  __|   | || | |_ |
 *  | |___| |__| | |\  | |     _| || |__| |
 *   \_____\____/|_| \_|_|    |_____\_____|
 */

#ifndef EEC_CONFIG_HH
#define EEC_CONFIG_HH

#include <vector>

#include "EECUtils.hh"

BEGIN_EEC_NAMESPACE

//-----------------------------------------------------------------------------
// Class that stores the configuration for an EEC computation
//-----------------------------------------------------------------------------

struct EECConfig {

  // exponents for the particle weights and charges
  std::vector<double> weight_powers;
  std::vector<unsigned> charge_powers;

  // computation specifications
  unsigned N, nfeatures;
  int num_threads, omp_chunksize;
  long print_every;
  bool norm, use_charges, check_degen, average_verts;
  double R, beta;

  // in case of FastJet events
  ParticleWeight particle_weight;
  PairwiseDistance pairwise_distance;

  // constructor stores all fields in particular order
  EECConfig(unsigned N = 2,
            bool norm = true,
            const std::vector<double> & weight_powers = {1},
            const std::vector<unsigned> & charge_powers = {0},
            ParticleWeight particle_weight = ParticleWeight::Default,
            PairwiseDistance pairwise_distance = PairwiseDistance::Default,
            int num_threads = -1,
            int omp_chunksize = 10,
            long print_every = -10,
            bool check_degen = false,
            bool average_verts = false,
            double R = 1,
            double beta = 1) :
    nfeatures(3), use_charges(false)
  {
    this->N = N;
    this->norm = norm;
    this->weight_powers = weight_powers;
    this->charge_powers = charge_powers;
    this->particle_weight = particle_weight;
    this->pairwise_distance = pairwise_distance;
    this->num_threads = num_threads;
    this->omp_chunksize = omp_chunksize;
    this->print_every = print_every;
    this->check_degen = check_degen;
    this->average_verts = average_verts;
    this->R = R;
    this->beta = beta;
  }

private:

  //////////////////////
  // BOOST SERIALIZATION
  //////////////////////

#ifdef BOOST_SERIALIZATION_ACCESS_HPP
  friend class boost::serialization::access;
#endif

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & weight_powers & charge_powers
       & N & nfeatures & num_threads & omp_chunksize & print_every
       & norm & use_charges & check_degen & average_verts
       & R & beta
       & particle_weight & pairwise_distance;
  }

}; // EECConfig

END_EEC_NAMESPACE

#endif // EEC_CONFIG_HH
