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
 */

#ifndef EEC_HH
#define EEC_HH

#include "EECLongestSide.hh"
#include "EECTriangleOPE.hh"

// namespace for EEC code
namespace eec {

//-----------------------------------------------------------------------------
// EECLongestSide typedefs
//-----------------------------------------------------------------------------

typedef EECLongestSide<hist::axis::id> EECLongestSideId;
typedef EECLongestSide<hist::axis::log> EECLongestSideLog;

//-----------------------------------------------------------------------------
// EECTriangleOPE typedefs
//-----------------------------------------------------------------------------

typedef EECTriangleOPE<hist::axis::id, hist::axis::id, hist::axis::id> EECTriangleOPEIdIdId;
typedef EECTriangleOPE<hist::axis::log, hist::axis::id, hist::axis::id> EECTriangleOPELogIdId;
typedef EECTriangleOPE<hist::axis::id, hist::axis::log, hist::axis::id> EECTriangleOPEIdLogId;
typedef EECTriangleOPE<hist::axis::log, hist::axis::log, hist::axis::id> EECTriangleOPELogLogId;

} // namespace eec

#endif // EEC_HH
