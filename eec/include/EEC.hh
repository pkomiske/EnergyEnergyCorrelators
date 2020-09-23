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

#ifndef EEC_HH
#define EEC_HH

#include "EECLongestSide.hh"
#include "EECTriangleOPE.hh"

// namespace for EEC code
namespace eec {

//-----------------------------------------------------------------------------
// EECLongestSide typedefs
//-----------------------------------------------------------------------------

typedef EECLongestSide<> EECLongestSide_id;
typedef EECLongestSide<bh::axis::transform::log> EECLongestSide_log;

//-----------------------------------------------------------------------------
// EECTriangleOPE typedefs
//-----------------------------------------------------------------------------

typedef EECTriangleOPE<> EECTriangleOPE_id_id_id;
typedef EECTriangleOPE<bh::axis::transform::log> EECTriangleOPE_log_id_id;
typedef EECTriangleOPE<bh::axis::transform::id, bh::axis::transform::log> EECTriangleOPE_id_log_id;
typedef EECTriangleOPE<bh::axis::transform::log, bh::axis::transform::log> EECTriangleOPE_log_log_id;

} // namespace eec

#endif // EEC_HH
