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

BEGIN_EEC_NAMESPACE

//-----------------------------------------------------------------------------
// EECLongestSide aliases
//-----------------------------------------------------------------------------

using EECLongestSideId = EECLongestSide<hist::axis::id>;
using EECLongestSideLog = EECLongestSide<hist::axis::log>;

//-----------------------------------------------------------------------------
// EECTriangleOPE aliases
//-----------------------------------------------------------------------------

using EECTriangleOPEIdIdId = EECTriangleOPE<hist::axis::id, hist::axis::id, hist::axis::id>;
using EECTriangleOPELogIdId = EECTriangleOPE<hist::axis::log, hist::axis::id, hist::axis::id>;
using EECTriangleOPEIdLogId = EECTriangleOPE<hist::axis::id, hist::axis::log, hist::axis::id>;
using EECTriangleOPELogLogId = EECTriangleOPE<hist::axis::log, hist::axis::log, hist::axis::id>;

END_EEC_NAMESPACE

#endif // EEC_HH
