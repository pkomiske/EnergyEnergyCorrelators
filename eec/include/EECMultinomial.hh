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
 *   __  __ _    _ _   _______ _____ _   _  ____  __  __ _____          _      
 *  |  \/  | |  | | | |__   __|_   _| \ | |/ __ \|  \/  |_   _|   /\   | |     
 *  | \  / | |  | | |    | |    | | |  \| | |  | | \  / | | |    /  \  | |     
 *  | |\/| | |  | | |    | |    | | | . ` | |  | | |\/| | | |   / /\ \ | |     
 *  | |  | | |__| | |____| |   _| |_| |\  | |__| | |  | |_| |_ / ____ \| |____ 
 *  |_|  |_|\____/|______|_|  |_____|_| \_|\____/|_|  |_|_____/_/    \_\______|
 */

#ifndef EEC_MULTINOMIAL_HH
#define EEC_MULTINOMIAL_HH

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "EECUtils.hh"

// Check for 64 bit compilation
#if defined(__GNUC__) || defined(__clang__)
#  if defined(__x86_64__) || defined(__aarch64__)
#    define ENV64BIT
#  else
#    define ENV32BIT
#  endif
#elif defined(_MSC_VER)
#  if defined(_WIN64)
#    define ENV64BIT
#  else
#    define ENV32BIT
#  endif
#else
#  define ENV32BIT
#endif

BEGIN_EEC_NAMESPACE

const std::array<unsigned, 13> FACTORIALS {
  1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600
};

#ifdef ENV64BIT
  const std::array<std::size_t, 21> FACTORIALS_LONG {
    1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,
    6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000,
    6402373705728000, 121645100408832000, 2432902008176640000
  };
#else
  const std::array<unsigned, 13> FACTORIALS_LONG = FACTORIALS;
#endif

// multinomial factor on sorted indices
template<std::size_t N>
unsigned multinomial(const std::array<unsigned, N> & inds) noexcept {

  unsigned denom(1), count(1);
  for (unsigned i = 1; i < N; i++) {
    if (inds[i] == inds[i-1]) count++;
    else {
      denom *= FACTORIALS[count];
      count = 1;
    }
  }
  denom *= FACTORIALS[count];

  return FACTORIALS[N]/denom;
}

inline unsigned multinomial_vector(const std::vector<unsigned> & inds) noexcept {

  unsigned denom(1), count(1);
  for (unsigned i = 1; i < inds.size(); i++) {
    if (inds[i] == inds[i-1]) count++;
    else {
      denom *= FACTORIALS[count];
      count = 1;
    }
  }
  denom *= FACTORIALS[count];

  return FACTORIALS[inds.size()]/denom;
}

template<unsigned N_>
struct Multinomial {

  Multinomial() : Nfactorial_(FACTORIALS[N_]) {
    static_assert(1 <= N_ && N_ < 13, "N must be positive and less than 13");
  }

  // get access to N via a public function
  static constexpr unsigned N() noexcept { return N_; }

  // set index at position 0 < i < N-1
  template<unsigned i>
  void set_index(unsigned ind) noexcept {
  #ifndef SWIG
    static_assert(i > 0 && i < N_-1, "index i must be less than N-1 and greater than 0");
  #endif
    _set_index<i>(ind);
  }

  // 0th index is special
  void set_index_0(unsigned ind) noexcept {
    inds_[0] = ind;
    counts_[0] = denoms_[0] = 1;
  }

  // set index at position N-1
  void set_index_final(unsigned ind) noexcept {
    _set_index<N()-1>(ind);

    // handle final degeneracy factor
    if (counts_[N()-1] > 1)
      denoms_[N()-1] *= FACTORIALS[counts_[N()-1]];
  }

  unsigned value() const noexcept {

    // if we are entirely degenerate (most cases) return N!
    if (denoms_[N()-1] == 1) return Nfactorial_;

    // return N!/(denom)
    else return Nfactorial_/denoms_[N()-1];
  }

private:

  std::array<unsigned, N_> inds_, counts_, denoms_;
  unsigned Nfactorial_;

  template<unsigned i>
  void _set_index(unsigned ind) noexcept {
    inds_[i] = ind;
    counts_[i] = counts_[i-1];
    denoms_[i] = denoms_[i-1];
    if (ind == inds_[i-1]) counts_[i]++;
    else {
      denoms_[i] *= FACTORIALS[counts_[i]];
      counts_[i] = 1;
    }
  }

}; // Multinomial

// designed to work with an arbitrarily-sized vector
struct DynamicMultinomial {

  DynamicMultinomial(unsigned N) :
    N_(N), Nm1_(N_-1), Nfactorial_(FACTORIALS_LONG[N_]),
    inds_(N_), counts_(N_), denoms_(N_)
  {
    if (N_ == 0 || N_ >= FACTORIALS_LONG.size())
      throw std::invalid_argument("N must be positive and less than " +
                                  std::to_string(FACTORIALS_LONG.size()));
  }

  unsigned N() const noexcept { return N_; }

  // set index at position 0 < i < N-1
  void set_index(unsigned i, unsigned ind) noexcept {
    if (i == 0) {
      inds_[0] = ind;
      counts_[0] = denoms_[0] = 1;
    }
    else {
      _set_index(i, ind);
      if (i == Nm1_ && counts_[Nm1_] > 1)  
        denoms_[Nm1_] *= FACTORIALS_LONG[counts_[Nm1_]];
    }
  }

  std::size_t value() const noexcept {

    // if we are entirely degenerate (most cases) return N!
    if (denoms_[Nm1_] == 1) return Nfactorial_;

    // return N!/(denom)
    else return Nfactorial_/denoms_[Nm1_];
  }

private:

  unsigned N_, Nm1_;
  std::size_t Nfactorial_;
  std::vector<unsigned> inds_, counts_;
  std::vector<std::size_t> denoms_;

  void _set_index(unsigned i, unsigned ind) noexcept {
    unsigned im1(i-1);
    inds_[i] = ind;
    counts_[i] = counts_[im1];
    denoms_[i] = denoms_[im1];
    if (ind == inds_[im1]) counts_[i]++;
    else {
      denoms_[i] *= FACTORIALS_LONG[counts_[i]];
      counts_[i] = 1;
    }
  }

}; // DynamicMultinomial

END_EEC_NAMESPACE

#endif // EEC_MULTINOMIAL_HH
