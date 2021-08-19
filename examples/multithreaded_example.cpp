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

// EnergyEnergyCorrelators library
#include "EEC.hh"

// classes and functions for reading/preparing events
#include "ExampleUtils.hh"

// without these lines, `eec::` should be prefixed with `fastjet::contrib`
using namespace fastjet;
using namespace fastjet::contrib;

template<class T>
void run_eec_comp(T & eec, EventProducer * evp) {

  // turns off some extra histograms that EEC can calculate to avoid printing too much below
  eec.set_track_covariance(false);
  eec.set_variance_bound(false);

  // this prints a nicely formatted description of the EEC
  std::cout << eec.description() << std::endl;

  // loop over events
  evp->reset();
  while (evp->next())

    // evp->particles() is just a vector of PseudoJets representing the particles
    // .push_back() internally stores the particles and delays computation
    eec.push_back(evp->particles());

  // run multithreaded computation
  eec.batch_compute();

  // sum EEC (this should be number of events if each event has total weight 1)
  std::cout << std::setprecision(8)
            << "Number of events: " << eec.event_counter() << '\n'
            << "Sum of " << eec.compname() << ": " << eec.sum() << '\n'
            << '\n';

  // output histogram(s)
  std::cout << eec;
}

int main(int argc, char** argv) {

  // load events
  EventProducer * evp(load_events(argc, argv));
  if (evp == nullptr)
    return 1;

  // create an EECLongestSide
  //   arguments used here: (N, nbins, {axis_min, axis_max})
  eec::EECLongestSideLog eec_longestside(2, 75, {1e-5, 1});

  // create an EECTriangleOPE
  //   arguments used here: ({nbins_xL, nbins_xi, nbins_phi}, {{axis_xL_range, axis_xi_range, axis_phi_range}})
  //   note: the extra braces around the axes_range argument are required by std::array
  //   number of bins is kept small here so not too much text gets printed
  eec::EECTriangleOPELogLogId eec_triangleope({5, 5, 3}, {{{1e-5, 1.}, {1e-5, 1.}, {0., eec::PI/2}}});

  // demonstrate basic usage for each EEC
  run_eec_comp(eec_longestside, evp);
  run_eec_comp(eec_triangleope, evp);

  return 0;
}
