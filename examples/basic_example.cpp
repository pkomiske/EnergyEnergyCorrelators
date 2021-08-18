//------------------------------------------------------------------------
// This file is part of Wasserstein, a C++ library with a Python wrapper
// that computes the Wasserstein/EMD distance. If you use it for academic
// research, please cite or acknowledge the following works:
//
//   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
//       https://doi.org/10.1103/PhysRevLett.123.041801
//   - Komiske, Metodiev, Thaler (2020) arXiv:2004.04159
//       https://doi.org/10.1007/JHEP07%282020%29006
//   - Boneel, van de Panne, Paris, Heidrich (2011)
//       https://doi.org/10.1145/2070781.2024192
//   - LEMON graph library https://lemon.cs.elte.hu/trac/lemon
//
// Copyright (C) 2019-2021 Patrick T. Komiske III
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
//------------------------------------------------------------------------

// EnergyEnergyCorrelators library
#include "EEC.hh"

// classes and functions for reading/preparing events
#include "ExampleUtils.hh"

// without these lines, `eec::` should be prefixed with `fastjet::contrib`
using namespace fastjet;
using namespace fastjet::contrib;

template<class T>
void run_eec_comp(T & eec, EventProducer * evp) {

  eec.set_track_covariance(false);
  eec.set_variance_bound(false);

  std::cout << eec.description() << std::endl;

  // loop over events
  evp->reset();
  while (evp->next())

    // evp->particles() is just a vector of PseudoJets representing the particles
    eec.compute(evp->particles());

  // outputs histogram(s)
  std::cout << eec;
}

int main(int argc, char** argv) {

  // load events
  EventProducer * evp(load_events(argc, argv));
  if (evp == nullptr)
    return 1;

  // specify EECs
  eec::EECLongestSideLog eec_longestside(2, 75, {1e-5, 1});
  eec::EECTriangleOPELogLogId eec_triangleope({10, 10, 5}, {{{1e-5, 1.}, {1e-5, 1.}, {0., eec::PI/2}}});

  run_eec_comp(eec_longestside, evp);
  run_eec_comp(eec_triangleope, evp);

  return 0;
}
