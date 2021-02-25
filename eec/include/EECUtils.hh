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

/*  ______ ______ _____ 
 * |  ____|  ____/ ____|
 * | |__  | |__ | |     
 * |  __| |  __|| |     
 * | |____| |___| |____ 
 * |______|______\_____|
 *  _    _ _______ _____ _       _____ 
 * | |  | |__   __|_   _| |     / ____|
 * | |  | |  | |    | | | |    | (___  
 * | |  | |  | |    | | | |     \___ \ 
 * | |__| |  | |   _| |_| |____ ____) |
 *  \____/   |_|  |_____|______|_____/ 
 */

#ifndef EEC_UTILS_HH
#define EEC_UTILS_HH

// OpenMP for multithreading
#ifdef _OPENMP
#include <omp.h>
#endif

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------

const double REG = 1e-100;
const double PI = 3.14159265358979323846;
const double TWOPI = 6.28318530717958647693;

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

// determine the number of threads to use
inline int determine_num_threads(int num_threads) {
#ifdef _OPENMP
  if (num_threads == -1 || num_threads > omp_get_max_threads())
    return omp_get_max_threads();
  return num_threads;
#else
  return 1;
#endif
}

inline int get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

#endif // EEC_UTILS_HH