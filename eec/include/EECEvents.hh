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
 *   ________      ________ _   _ _______ _____ 
 *  |  ____\ \    / /  ____| \ | |__   __/ ____|
 *  | |__   \ \  / /| |__  |  \| |  | | | (___  
 *  |  __|   \ \/ / |  __| | . ` |  | |  \___ \ 
 *  | |____   \  /  | |____| |\  |  | |  ____) |
 *  |______|   \/   |______|_| \_|  |_| |_____/ 
 */

#ifndef EEC_EVENTS_HH
#define EEC_EVENTS_HH

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace eec {

//-----------------------------------------------------------------------------
// Class to help store multiple events for multithreaded computation
//-----------------------------------------------------------------------------

class EECEvents {
private:

  std::vector<const double *> events_;
  std::vector<unsigned> mults_;
  std::vector<double> weights_;

  unsigned nfeatures_;

public:

  EECEvents(std::size_t nev = 0, unsigned nfeatures = 0) : nfeatures_(nfeatures) {
    events_.reserve(nev);
    mults_.reserve(nev);
    weights_.reserve(nev);
  }

  // access functions
  const std::vector<const double *> & events() const { return events_; }
  const std::vector<unsigned> & mults() const { return mults_; }
  const std::vector<double> & weights() const { return weights_; }

  // add event 
  void append(const double * event_ptr, unsigned mult, unsigned nfeatures,  double weight) {
    if (nfeatures_ > 0 && nfeatures != nfeatures_) {
      std::ostringstream oss;
      oss << "event has " << nfeatures << " features per particle, expected "
          << nfeatures_ << " features per particle";
      throw std::invalid_argument(oss.str());
    }

    events_.push_back(event_ptr);
    mults_.push_back(mult);
    weights_.push_back(weight);
  }

  void append(const double * event_ptr, unsigned mult) {
    events_.push_back(event_ptr);
    mults_.push_back(mult);
  }

}; // EECEvents

} // namespace eec

#endif // EEC_EVENTS_HH
