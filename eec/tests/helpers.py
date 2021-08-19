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

from collections import Counter
import itertools
import math
from operator import itemgetter

import numpy as np
from scipy.spatial.distance import cdist

import eec

def epsilon_diff(X, Y, epsilon=1e-14):
    return np.all(np.abs(X - Y) < epsilon)

def epsilon_percent(X, Y, epsilon=1e-14):
    return np.all(2*np.abs(X - Y)/(np.abs(X) + np.abs(Y) + 1e-100) < epsilon)

def epsilon_either(X, Y, eps_diff, eps_percent):
    return epsilon_diff(X, Y, eps_diff) or epsilon_percent(X, Y, eps_percent)

# function for getting histograms from observable values
def calc_eec_hist_on_event(vals, bins, weights):
    
    # compute histogram with errors
    hist = np.histogram(vals, bins=bins, weights=weights)[0]
    errs2 = hist*hist
        
    return hist, errs2

class SlowEECBase(object):

    def __init__(self, N, nbins, bin_ranges, axes, norm, weight_powers, charge_powers, overflows=True):
        self.N = N
        self.norm = norm
        self.weight_powers = N*[weight_powers] if isinstance(weight_powers, (int, float)) else weight_powers
        self.charge_powers = N*[charge_powers] if isinstance(charge_powers, (int, float)) else charge_powers

        self.bins = []
        assert len(nbins) == len(bin_ranges)
        assert len(axes) == len(nbins)
        for n,bin_range,axis in zip(nbins, bin_ranges, axes):
            if axis == 'id':
                bs = np.linspace(*bin_range, n+1)
            elif axis == 'log':
                bs = np.exp(np.linspace(*np.log(bin_range), n+1))
            else:
                raise ValueError('bad axis type')

            if overflows:
                bs = np.concatenate(([0.], bs))
                bs = np.concatenate((bs, [np.inf]))

            self.bins.append(bs)

        s = tuple(len(b) - 1 for b in self.bins)
        self.hist, self.errs2 = np.zeros(s), np.zeros(s)

    def __call__(self, events, weights):
        for event,weight in zip(events, weights):

            pts = event[:,0]/event[:,0].sum() if self.norm else event[:,0]
            dists = cdist(event[:,1:3], event[:,1:3])
            charges = event[:,3] if event.shape[1] == 4 else np.ones_like(pts)

            hist, errs2 = self._compute_func(pts, dists, charges, weight)
            self.hist += hist
            self.errs2 += errs2

    @property
    def errs(self):
        return np.sqrt(self.errs2)

class SlowEECLongestSideSym(SlowEECBase):

    inds, combs, factors = {}, {}, {}

    def construct_inds_factors(self, nmax, npec=3):
        inds, combs, factors = [None], [None], [None]

        if npec in self.inds and nmax < len(self.inds[npec]):
            return
        
        for n in range(1, nmax + 1):
            npecfact = math.factorial(npec)
            inds_n, combs_n, factors_n = [], [], []
            
            for comb in itertools.combinations_with_replacement(range(n), npec):
                
                symfactor = 1
                for c in Counter(comb).values():
                    symfactor *= math.factorial(c)
                    
                factors_n.append(npecfact//symfactor)
                assert eec.multinomial(comb) == factors_n[-1]
                inds_n.append(tuple(itertools.combinations(comb, 2)))
                combs_n.append(comb)
            
            inds_n = np.ascontiguousarray(np.asarray(inds_n).transpose((2,0,1)))
            inds.append((inds_n[0], inds_n[1]))
            
            factors.append(np.asarray(factors_n))
            combs.append(np.asarray(combs_n))
            
        self.inds[npec] = inds
        self.combs[npec] = combs
        self.factors[npec] = factors

    def _compute_func(self, pts, dists, charges, weight):

        # handle zero length events
        mult = len(pts)
        if mult == 0:
            return 0., 0.

        # handle charges
        weights = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        
        # form weights
        weights = weight * self.factors[self.N][mult] * np.prod(weights[self.combs[self.N][mult]], axis=1)
        
        # get pairwise dists and maximum length of all triangles
        max_dists = np.max(dists[self.inds[self.N][mult]], axis=1)
        
        # get hist
        return calc_eec_hist_on_event(max_dists, self.bins[0], weights)
    
class SuperSlowEECLongestSideSym(SlowEECBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.N == 2:
            self._compute_func = self._compute_eec
        elif self.N == 3:
            self._compute_func = self._compute_eeec
        elif self.N == 4:
            self._compute_func = self._compute_eeeec
        else:
            raise ValueError('Invalid N')

    def _compute_eec(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]

        hist_vals, hist_weights = [], []
        for i in range(len(pts)):
            for j in range(i+1):
                hist_vals.append(dists[i,j])
                hist_weights.append(weight*weights0[i]*weights1[j]*(1. if i==j else 2.))

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], np.asarray(hist_weights))

    def _compute_eeec(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        hist_vals, hist_weights = [], []
        for i in range(len(pts)):
            weight_i = weight * weights0[i]
            for j in range(i+1):
                ij_match = (i == j)
                weight_ij = weight_i * weights1[j]
                for k in range(j+1):
                    jk_match = (j == k)
                    hist_vals.append(max(dists[i,j], dists[i,k], dists[j,k]))
                    hist_weights.append(weight_ij * weights2[k] * (1. if ij_match and jk_match else (3. if ij_match or jk_match else 6.)))

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], np.asarray(hist_weights))

    def _compute_eeeec(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]
        weights3 = pts**self.weight_powers[3] * charges**self.charge_powers[3]

        hist_vals, hist_weights = [], []
        for i in range(len(pts)):
            weight_i = weight * weights0[i]

            for j in range(i+1):
                weight_ij = weight_i * weights1[j]
                
                for k in range(j+1):
                    weight_ijk = weight_ij * weights2[k]
                    dist_ijk_max = max(dists[i,j], dists[i,k], dists[j,k])
                    
                    for l in range(k+1):
                        hist_vals.append(max(dist_ijk_max, dists[l,i], dists[l,j], dists[l,k]))
                        hist_weights.append(weight_ijk * weights3[l] * eec.multinomial((i, j, k, l)))

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], np.asarray(hist_weights))

class SuperSlowEECLongestSide(SlowEECBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.N == 2:
            self._compute_func = self._compute_eec
        elif self.N == 3:
            self._compute_func = self._compute_eeec
        else:
            raise ValueError('Invalid N')

    def _compute_eec(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]

        mult = len(pts)
        hist_vals = [dists[i,j] for i in range(mult) for j in range(mult)]
        hist_weights = [weights0[i] * weights1[j] for i in range(mult) for j in range(mult)]

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], weight * np.asarray(hist_weights))

    def _compute_eeec(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        mult = len(pts)
        hist_vals = [max(dists[i,j], dists[i,k], dists[j,k]) for i in range(mult) for j in range(mult) for k in range(mult)]
        hist_weights = [weights0[i] * weights1[j] * weights2[k] for i in range(mult) for j in range(mult) for k in range(mult)]

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], weight * np.asarray(hist_weights))

class SuperSlowEECLongestSideAsymN3(SlowEECBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.N == 3

        match01 = (self.weight_powers[0] == self.weight_powers[1]) and (self.charge_powers[0] == self.charge_powers[1])
        match02 = (self.weight_powers[0] == self.weight_powers[2]) and (self.charge_powers[0] == self.charge_powers[2])
        match12 = (self.weight_powers[1] == self.weight_powers[2]) and (self.charge_powers[1] == self.charge_powers[2])
        assert not (match01 and match02), 'This EEC is for asymmetric computation'
        self._compute_func = self._compute_eeec_ij_sym
        nh = 2
        if match01:
            pass
        elif match02:
            self.weight_powers = [self.weight_powers[0], self.weight_powers[2], self.weight_powers[1]]
            self.charge_powers = [self.charge_powers[0], self.charge_powers[2], self.charge_powers[1]]
        elif match12:
            self.weight_powers = [self.weight_powers[1], self.weight_powers[2], self.weight_powers[0]]
            self.charge_powers = [self.charge_powers[1], self.charge_powers[2], self.charge_powers[0]]
        else:
            self._compute_func = self._compute_eeec_no_sym
            nh = 3

        self.hist = np.zeros(((nh,) + self.hist.shape))
        self.errs2 = np.zeros(((nh,) + self.errs2.shape))

    def _compute_eeec_ij_sym(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        mult = len(pts)
        hist_vals, hist_weights = [[], []], [[], []]
        for i in range(mult):
            weight_i = weights0[i]
            for j in range(mult):
                d_ij = (dists[i,j], 0)
                weight_ij = weight_i * weights1[j]
                ij_match = i == j
                for k in range(mult):
                    d_ik = (dists[i,k], 1)
                    d_jk = (dists[j,k], 1)
                    weight_ijk = weight_ij * weights2[k]
                    jk_match = j == k
                    max_dist = max(d_ij, d_ik, d_jk, key=itemgetter(0))

                    hist_vals[max_dist[1]].append(max_dist[0])
                    hist_weights[max_dist[1]].append(weight_ijk)

                    # all degenerate fill everything
                    if ij_match and jk_match:
                        hist_vals[1 - max_dist[1]].append(max_dist[0])
                        hist_weights[1 - max_dist[1]].append(weight_ijk)

        hist0, errs0 = calc_eec_hist_on_event(np.asarray(hist_vals[0]), self.bins[0], weight * np.asarray(hist_weights[0]))
        hist1, errs1 = calc_eec_hist_on_event(np.asarray(hist_vals[1]), self.bins[0], weight * np.asarray(hist_weights[1]))

        return [hist0, hist1], [errs0, errs1]

    def _compute_eeec_no_sym(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        mult = len(pts)
        hist_vals, hist_weights = [[], [], []], [[], [], []]
        for i in range(mult):
            weight_i = weights0[i]
            for j in range(mult):
                d_ij = (dists[i,j], 0)
                weight_ij = weight_i * weights1[j]
                ij_match = i == j
                for k in range(mult):
                    d_ik = (dists[i,k], 2)
                    d_jk = (dists[j,k], 1)
                    weight_ijk = weight_ij * weights2[k]
                    ik_match = i == k
                    jk_match = j == k
                    max_dist = max(d_ij, d_ik, d_jk, key=itemgetter(0))

                    # no degeneracy
                    if not (ij_match or ik_match or jk_match):
                        hist_vals[max_dist[1]].append(max_dist[0])
                        hist_weights[max_dist[1]].append(weight_ijk)

                    # all degenerate
                    elif ij_match and jk_match:
                        hist_vals[0].append(max_dist[0])
                        hist_vals[1].append(max_dist[0])
                        hist_vals[2].append(max_dist[0])
                        hist_weights[0].append(weight_ijk)
                        hist_weights[1].append(weight_ijk)
                        hist_weights[2].append(weight_ijk)

                    elif ij_match:
                        hist_vals[1].append(max_dist[0])
                        hist_vals[2].append(max_dist[0])
                        hist_weights[1].append(weight_ijk)
                        hist_weights[2].append(weight_ijk)

                    elif jk_match:
                        hist_vals[0].append(max_dist[0])
                        hist_vals[2].append(max_dist[0])
                        hist_weights[0].append(weight_ijk)
                        hist_weights[2].append(weight_ijk)

                    elif ik_match:
                        hist_vals[1].append(max_dist[0])
                        hist_vals[0].append(max_dist[0])
                        hist_weights[1].append(weight_ijk)
                        hist_weights[0].append(weight_ijk)

                    else:
                        assert False

        hist0, errs0 = calc_eec_hist_on_event(np.asarray(hist_vals[0]), self.bins[0], weight * np.asarray(hist_weights[0]))
        hist1, errs1 = calc_eec_hist_on_event(np.asarray(hist_vals[1]), self.bins[0], weight * np.asarray(hist_weights[1]))
        hist2, errs2 = calc_eec_hist_on_event(np.asarray(hist_vals[2]), self.bins[0], weight * np.asarray(hist_weights[2]))

        return [hist0, hist1, hist2], [errs0, errs1, errs2]

class SuperSlowEECTriangleOPE(SlowEECBase):
    
    def __init__(self, *args, **kwargs):
        average_verts = kwargs.pop('average_verts', False)
        super().__init__(*args, **kwargs)
        assert self.N == 3

        match01 = (self.weight_powers[0] == self.weight_powers[1]) and (self.charge_powers[0] == self.charge_powers[1])
        match02 = (self.weight_powers[0] == self.weight_powers[2]) and (self.charge_powers[0] == self.charge_powers[2])
        match12 = (self.weight_powers[1] == self.weight_powers[2]) and (self.charge_powers[1] == self.charge_powers[2])
        if (match01 and match02) or average_verts:
            self._compute_func = self._compute_eeec_ijk_sym
            nh = 1
        elif match01:
            self._compute_func = self._compute_eeec_ij_sym
            nh = 3
        elif match02:
            self.weight_powers = [self.weight_powers[0], self.weight_powers[2], self.weight_powers[1]]
            self.charge_powers = [self.charge_powers[0], self.charge_powers[2], self.charge_powers[1]]
            self._compute_func = self._compute_eeec_ij_sym
            nh = 3
        elif match12:
            self.weight_powers = [self.weight_powers[1], self.weight_powers[2], self.weight_powers[0]]
            self.charge_powers = [self.charge_powers[1], self.charge_powers[2], self.charge_powers[0]]
            self._compute_func = self._compute_eeec_ij_sym
            nh = 3
        else:
            self._compute_func = self._compute_eeec_no_sym
            nh = 6

        self.hist = np.zeros(((nh,) + self.hist.shape))
        self.errs2 = np.zeros(((nh,) + self.errs2.shape))

    def calc_eec_hist(self, dists, weights):
        xi = dists[:,0]/(dists[:,1] + 1e-100)
        diff = dists[:,2] - dists[:,1]
        phi = np.arcsin(np.sqrt(np.abs(1 - diff*diff/(dists[:,0]*dists[:,0] + 1e-100))))

        hist = np.histogramdd([dists[:,2], xi, phi], bins=self.bins, weights=weights)[0]
        errs2 = hist*hist

        return hist, errs2

    def _compute_eeec_ijk_sym(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        mult = len(pts)
        hist_vals, hist_weights = [], []
        for i in range(mult):
            weight_i = weights0[i]
            for j in range(mult):
                d_ij = dists[i,j]
                weight_ij = weight_i * weights1[j]
                for k in range(mult):
                    hist_vals.append(sorted([d_ij, dists[i,k], dists[j,k]]))
                    hist_weights.append(weight_ij * weights2[k])

        return self.calc_eec_hist(np.asarray(hist_vals), weight * np.asarray(hist_weights))

    def _compute_eeec_ij_sym(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        mult = len(pts)
        hist_vals, hist_weights = [[], [], []], [[], [], []]
        for i in range(mult):
            weight_i = weights0[i]
            for j in range(mult):
                d_ij = (dists[i,j], 0)
                weight_ij = weight_i * weights1[j]
                ij_match = i == j
                for k in range(mult):
                    d_ik = (dists[i,k], 1)
                    d_jk = (dists[j,k], 2)
                    weight_ijk = weight_ij * weights2[k]
                    jk_match = j == k
                    ik_match = i == k
                    sorted_dists = sorted([d_ij, d_ik, d_jk], key=itemgetter(0))
                    dist_vals = [x[0] for x in sorted_dists]

                    if not (ij_match or ik_match or jk_match):
                        hist_is = [(0 if sorted_dists[0][1] == 0 else 
                                  (1 if sorted_dists[1][1] == 0 else
                                  (2 if sorted_dists[2][1] == 0 else 3)))]
                    elif ij_match and jk_match:
                        hist_is = range(3)
                    elif ij_match:
                        hist_is = [0]
                    elif jk_match or ik_match:
                        hist_is = [1, 2]
                    else:   
                        assert False

                    for hist_i in hist_is:
                        hist_vals[hist_i].append(dist_vals)
                        hist_weights[hist_i].append(weight_ijk)

        hist0, errs0 = self.calc_eec_hist(np.asarray(hist_vals[0]), weight * np.asarray(hist_weights[0]))
        hist1, errs1 = self.calc_eec_hist(np.asarray(hist_vals[1]), weight * np.asarray(hist_weights[1]))
        hist2, errs2 = self.calc_eec_hist(np.asarray(hist_vals[2]), weight * np.asarray(hist_weights[2]))

        return [hist0, hist1, hist2], [errs0, errs1, errs2]

    def _compute_eeec_no_sym(self, pts, dists, charges, weight):
        weights0 = pts**self.weight_powers[0] * charges**self.charge_powers[0]
        weights1 = pts**self.weight_powers[1] * charges**self.charge_powers[1]
        weights2 = pts**self.weight_powers[2] * charges**self.charge_powers[2]

        mult = len(pts)
        hist_vals, hist_weights = [[], [], [], [], [], []], [[], [], [], [], [], []]
        for i in range(mult):
            weight_i = weights0[i]
            for j in range(mult):
                d_ij = (dists[i,j], 0)
                weight_ij = weight_i * weights1[j]
                ij_match = i == j
                for k in range(mult):
                    d_ik = (dists[i,k], 1)
                    d_jk = (dists[j,k], 2)
                    weight_ijk = weight_ij * weights2[k]
                    jk_match = j == k
                    ik_match = i == k
                    sorted_dists = sorted([d_ij, d_ik, d_jk], key=itemgetter(0))
                    dist_vals = [x[0] for x in sorted_dists]

                    # no degeneracy
                    if not (ij_match or ik_match or jk_match):
                        if sorted_dists[0][1] == 0:
                            if sorted_dists[1][1] == 1:
                                hist_is = [0]
                            else:
                                hist_is = [1]
                        elif sorted_dists[1][1] == 0:
                            if sorted_dists[0][1] == 1:
                                hist_is = [2]
                            else:
                                hist_is = [3]
                        else:
                            if sorted_dists[0][1] == 1:
                                hist_is = [4]
                            else:
                                hist_is = [5]

                    # all degenerate
                    elif ij_match and jk_match:
                        hist_is = range(6)
                    elif ij_match:
                        hist_is = [0, 1]
                    elif jk_match:
                        hist_is = [3, 5]
                    elif ik_match:
                        hist_is = [2, 4]
                    else:
                        assert False

                    for hist_i in hist_is:
                        hist_vals[hist_i].append(dist_vals)
                        hist_weights[hist_i].append(weight_ijk)

        hists, errs = [], []
        for i in range(6):
            h, e = self.calc_eec_hist(np.asarray(hist_vals[i]), weight * np.asarray(hist_weights[i]))
            hists.append(h)
            errs.append(e)

        return hists, errs