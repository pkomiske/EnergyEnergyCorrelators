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
import numpy as np

# load some particle test data
# events will be an array of 2d arrays with columns (pt, y, phi, charge)
events, y = ef.qg_jets.load(num_data=500, pad=False)
for i in range(len(events)):
    events[i][:,3] = ef.pids2chrgs(events[i][:,3])

# EECLongestSide instance
#   note: `axis` keyword argument accepted, defaults to 'log' (other option is 'id')
eec_ls = eec.EECLongestSide(2, 750, axis_range=(1e-5, 1))
print(eec_ls)

# multicore compute
eec_ls(events)

# EECTriangleOPE instance
#   note: `axes` keyword argument accepted, defaults to ('log', 'log', 'id')
eec_ope = eec.EECTriangleOPE((150, 150, 150), axes_range=[(1e-4, 1), (1e-4, 1), (0, np.pi/2)])
print(eec_ope)

# multicore compute
eec_ope(events)

# print some results
print('EECLongestSide sum:', eec_ls.sum())
print('EECTriangleOPE sum:', eec_ope.sum())
print('Number of events:', len(events))
print()

# put all data into a Python dictionary
eec_ls_dict = eec_ls.as_dict()
eec_ope_dist = eec_ope.as_dict()

print('EEC dict keys', eec_ls_dict.keys())

# demonstrate rebinning (combines every 10 bins into 1)
eec_ls.reduce(eec.rebin(10))
eec_ope.reduce([eec.rebin(5), eec.rebin(10), eec.rebin(15)])

print('EECLongestSide nbins:', eec_ls.nbins())
print('EECTriangleOPE nbins:', eec_ope.nbins(0), eec_ope.nbins(1), eec_ope.nbins(2))
