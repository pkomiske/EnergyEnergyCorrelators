# EnergyEnergyCorrelators - Evaluates EECs on particle physics events
# Copyright (C) 2020-2021 Patrick T. Komiske III
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import eec
import energyflow as ef
import matplotlib.pyplot as plt
import numpy as np

# load some particle test data
# events will be an array of 2d arrays with columns (pt, y, phi, charge)
events, y = ef.qg_jets.load(num_data=10000, pad=False)
for i in range(len(events)):
    events[i][:,3] = ef.pids2chrgs(events[i][:,3])

colors = {2: 'tab:blue', 3: 'tab:green', 4: 'tab:red'}
errorbar_opts = {
   'fmt': 'o',
   'lw': 1.5,
   'capsize': 1.5,
   'capthick': 1,
   'markersize': 1.5,
}

for N in [2, 3, 4]:

    # EECLongestSide instance
    #   note: `axis` keyword argument accepted, defaults to 'log' (other option is 'id')
    eec_ls = eec.EECLongestSide(N, 75, axis_range=(1e-5, 1))
    print(eec_ls)

    # multicore compute
    eec_ls(events)

    # scale eec for plot
    eec_ls.scale(1/eec_ls.sum())

    # get bins
    midbins, bins = eec_ls.bin_centers(), eec_ls.bin_edges()
    binwidths = np.log(bins[1:]) - np.log(bins[:-1])

    # get EEC hist and (rough) error estimate
    # argument 0 means get histogram 0 (there can be multiple if using asymmetric vertex powers)
    # argument False mean ignore the overflow bins
    hist, errs = eec_ls.get_hist_errs(0, False)

    plt.errorbar(midbins, hist/binwidths,
                 xerr=(midbins - bins[:-1], bins[1:] - midbins),
                 yerr=errs/binwidths,
                 color=colors[N],
                 label='N = {}'.format(N),
                 **errorbar_opts)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1e-5, 1)
plt.ylim(1e-7, 1)

plt.xlabel('xL')
plt.ylabel('(Normalized) Cross Section')

plt.legend(loc='lower center', frameon=False)

plt.show()
