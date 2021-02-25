from collections import Counter
import itertools
import math
from operator import itemgetter
import pickle
import tempfile

import energyflow as ef
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from eec import eec
from eec import *

# load some reasonable test data
events, y = ef.qg_jets.load(num_data=500, pad=False)
for i in range(len(events)):
    events[i][:,3] = ef.pids2chrgs(events[i][:,3])

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

    def __init__(self, N, nbins, bin_ranges, axes, norm, pt_powers, ch_powers, overflows=True):
        self.N = N
        self.norm = norm
        self.pt_powers = N*[pt_powers] if isinstance(pt_powers, (int, float)) else pt_powers
        self.ch_powers = N*[ch_powers] if isinstance(ch_powers, (int, float)) else ch_powers

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
        weights = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        
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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]

        hist_vals, hist_weights = [], []
        for i in range(len(pts)):
            for j in range(i+1):
                hist_vals.append(dists[i,j])
                hist_weights.append(weight*weights0[i]*weights1[j]*(1. if i==j else 2.))

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], np.asarray(hist_weights))

    def _compute_eeec(self, pts, dists, charges, weight):
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]
        weights3 = pts**self.pt_powers[3] * charges**self.ch_powers[3]

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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]

        mult = len(pts)
        hist_vals = [dists[i,j] for i in range(mult) for j in range(mult)]
        hist_weights = [weights0[i] * weights1[j] for i in range(mult) for j in range(mult)]

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], weight * np.asarray(hist_weights))

    def _compute_eeec(self, pts, dists, charges, weight):
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

        mult = len(pts)
        hist_vals = [max(dists[i,j], dists[i,k], dists[j,k]) for i in range(mult) for j in range(mult) for k in range(mult)]
        hist_weights = [weights0[i] * weights1[j] * weights2[k] for i in range(mult) for j in range(mult) for k in range(mult)]

        return calc_eec_hist_on_event(np.asarray(hist_vals), self.bins[0], weight * np.asarray(hist_weights))

class SuperSlowEECLongestSideAsymN3(SlowEECBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.N == 3

        match01 = (self.pt_powers[0] == self.pt_powers[1]) and (self.ch_powers[0] == self.ch_powers[1])
        match02 = (self.pt_powers[0] == self.pt_powers[2]) and (self.ch_powers[0] == self.ch_powers[2])
        match12 = (self.pt_powers[1] == self.pt_powers[2]) and (self.ch_powers[1] == self.ch_powers[2])
        assert not (match01 and match02), 'This EEC is for asymmetric computation'
        self._compute_func = self._compute_eeec_ij_sym
        nh = 2
        if match01:
            pass
        elif match02:
            self.pt_powers = [self.pt_powers[0], self.pt_powers[2], self.pt_powers[1]]
            self.ch_powers = [self.ch_powers[0], self.ch_powers[2], self.ch_powers[1]]
        elif match12:
            self.pt_powers = [self.pt_powers[1], self.pt_powers[2], self.pt_powers[0]]
            self.ch_powers = [self.ch_powers[1], self.ch_powers[2], self.ch_powers[0]]
        else:
            self._compute_func = self._compute_eeec_no_sym
            nh = 3

        self.hist = np.zeros(((nh,) + self.hist.shape))
        self.errs2 = np.zeros(((nh,) + self.errs2.shape))

    def _compute_eeec_ij_sym(self, pts, dists, charges, weight):
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

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

        match01 = (self.pt_powers[0] == self.pt_powers[1]) and (self.ch_powers[0] == self.ch_powers[1])
        match02 = (self.pt_powers[0] == self.pt_powers[2]) and (self.ch_powers[0] == self.ch_powers[2])
        match12 = (self.pt_powers[1] == self.pt_powers[2]) and (self.ch_powers[1] == self.ch_powers[2])
        if (match01 and match02) or average_verts:
            self._compute_func = self._compute_eeec_ijk_sym
            nh = 1
        elif match01:
            self._compute_func = self._compute_eeec_ij_sym
            nh = 3
        elif match02:
            self.pt_powers = [self.pt_powers[0], self.pt_powers[2], self.pt_powers[1]]
            self.ch_powers = [self.ch_powers[0], self.ch_powers[2], self.ch_powers[1]]
            self._compute_func = self._compute_eeec_ij_sym
            nh = 3
        elif match12:
            self.pt_powers = [self.pt_powers[1], self.pt_powers[2], self.pt_powers[0]]
            self.ch_powers = [self.ch_powers[1], self.ch_powers[2], self.ch_powers[0]]
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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

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
        weights0 = pts**self.pt_powers[0] * charges**self.ch_powers[0]
        weights1 = pts**self.pt_powers[1] * charges**self.ch_powers[1]
        weights2 = pts**self.pt_powers[2] * charges**self.ch_powers[2]

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

@pytest.mark.multinomial
@pytest.mark.parametrize('nparticles', [1, 2, 5, 10, 20])
@pytest.mark.parametrize('N', [2, 3, 4])
def test_multinomial(N, nparticles):
    dm = eec.DynamicMultinomial(N)

    if N == 2:
        m2 = eec.Multinomial2()
        for i in range(nparticles):
            dm.set_index(0, i)
            m2.set_index_0(i)
            for j in range(i+1):
                dm.set_index(1, j)
                m2.set_index_final(j)
                assert dm.value() == m2.value()
                assert dm.value() == (1. if i==j else 2.)

    elif N == 3:
        m3 = eec.Multinomial3()
        for i in range(nparticles):
            dm.set_index(0, i)
            m3.set_index_0(i)
            for j in range(i+1):
                ij_match = (i == j)
                dm.set_index(1, j)
                m3.set_index_1(j)
                for k in range(j+1):
                    jk_match = (j == k)
                    dm.set_index(2, k)
                    m3.set_index_final(k)
                    assert dm.value() == m3.value()
                    assert dm.value() == (1. if ij_match and jk_match else (3. if ij_match or jk_match else 6.))

    elif N == 4:
        m4 = eec.Multinomial4()
        for i in range(nparticles):
            dm.set_index(0, i)
            m4.set_index_0(i)
            for j in range(i+1):
                dm.set_index(1, j)
                m4.set_index_1(j)
                for k in range(j+1):
                    dm.set_index(2, k)
                    m4.set_index_2(k)
                    for l in range(k+1):
                        dm.set_index(3, l)
                        m4.set_index_final(l)
                        assert dm.value() == m4.value()
                        assert dm.value() == eec.multinomial((i, j, k, l))

@pytest.mark.pycompare
@pytest.mark.parametrize('nparticles', [0, 1, 2, 4, 8])
@pytest.mark.parametrize('ch_powers', [0, 1, 2][:1])
@pytest.mark.parametrize('pt_powers', [1, 2][:1])
@pytest.mark.parametrize('nbins', [1, 15])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4])
def test_pycompare_longestside(N, axis, nbins, pt_powers, ch_powers, nparticles):

    super_slow_eec = SuperSlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles] for event in events[:200]]
    weights = 2*np.random.rand(len(events))

    super_slow_eec(local_events, weights)
    slow_eec(local_events, weights)

    assert epsilon_percent(super_slow_eec.hist, slow_eec.hist, 1e-12)
    assert epsilon_percent(super_slow_eec.errs, slow_eec.errs, 1e-6)

@pytest.mark.longestside
@pytest.mark.sym
@pytest.mark.parametrize('nparticles', [0, 1, 2, 4, 8, 16])
@pytest.mark.parametrize('ch_powers', [0, 1, 2])
@pytest.mark.parametrize('pt_powers', [1, 2])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('use_general_eNc', [False, True])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4, 5, 6])
def test_longestside_sym(N, axis, use_general_eNc, num_threads, pt_powers, ch_powers, nparticles):
    if nparticles > 8 and N >= 5:
        pytest.skip()

    nbins = 15
    eec = EECLongestSide(N, nbins, axis=axis, axis_range=(1e-5, 1), pt_powers=(pt_powers,), ch_powers=(ch_powers,),
                         print_every=0, num_threads=num_threads, use_general_eNc=use_general_eNc)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles] for event in events]
    weights = 2*np.random.rand(len(events))

    eec(local_events, weights)
    slow_eec(local_events, weights)

    hist, errs = eec.get_hist_errs()
    assert epsilon_either(hist, slow_eec.hist, 10**-12, 1e-14)
    assert epsilon_either(errs, slow_eec.errs, 10**-6, 1e-7)

@pytest.mark.longestside
@pytest.mark.asym
@pytest.mark.parametrize('nparticles', [1, 2, 5, 10])
@pytest.mark.parametrize('ch_powers', [(0,0), (1,1), (2,2), (0,1), (1,0), (2,0), (0,2)])
@pytest.mark.parametrize('pt_powers', [(1,1), (1,2), (2,1), (2,2), (0,1), (1,0)])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('axis', ['log', 'id'])
def test_longestside_asym_N2_average_verts(axis, num_threads, pt_powers, ch_powers, nparticles):

    eec = EECLongestSide(2, 15, axis=axis, axis_range=(1e-5, 1), pt_powers=pt_powers, ch_powers=ch_powers,
                         print_every=0, num_threads=num_threads)
    super_slow_eec = SuperSlowEECLongestSide(2, (15,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)

    local_events = [event[-nparticles:] for event in events]
    weights = 2*np.random.rand(len(events))

    eec(local_events, weights)
    super_slow_eec(local_events, weights)

    hist, errs = eec.get_hist_errs()
    assert epsilon_percent(hist, super_slow_eec.hist, 1e-10)
    assert epsilon_percent(errs, super_slow_eec.errs, 10**-5.5)

@pytest.mark.longestside
@pytest.mark.asym
@pytest.mark.parametrize('nparticles', [1, 2, 10])
@pytest.mark.parametrize('average_verts', [True, False])
@pytest.mark.parametrize('ch_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('pt_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('axis', ['log', 'id'])
def test_longestside_asym_N3(axis, num_threads, pt_powers, ch_powers, average_verts, nparticles):

    eec = EECLongestSide(3, 15, axis=axis, axis_range=(1e-5, 1), pt_powers=pt_powers, ch_powers=ch_powers,
                         print_every=0, num_threads=num_threads, average_verts=average_verts)
    super_slow_eec = (SuperSlowEECLongestSide(3, (15,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
                      if average_verts or (len(set(pt_powers)) == 1 and len(set(ch_powers)) == 1) else
                      SuperSlowEECLongestSideAsymN3(3, (15,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers))

    nev = 100
    local_events = [event[-nparticles:] for event in events[:nev]]
    weights = 2*np.random.rand(len(events))[:nev]

    eec(local_events, weights)
    super_slow_eec(local_events, weights)

    if average_verts or len(super_slow_eec.hist.shape) == 1:
        hist, errs = eec.get_hist_errs()
        assert epsilon_percent(hist, super_slow_eec.hist, 1e-10)
        assert epsilon_percent(errs, super_slow_eec.errs, 10**-5.5)
    else:
        for hist_i in range(super_slow_eec.hist.shape[0]):
            hist, errs = eec.get_hist_errs(hist_i)
            assert epsilon_either(hist, super_slow_eec.hist[hist_i], 1e-12, 1e-14), hist_i
            assert epsilon_either(errs, super_slow_eec.errs[hist_i], 1e-6, 1e-7), hist_i

@pytest.mark.triangleope
@pytest.mark.parametrize('nparticles', [1, 4, 10])
@pytest.mark.parametrize('average_verts', [True, False])
@pytest.mark.parametrize('ch_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('pt_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('axes', [('log', 'log', 'id'), ('id', 'id', 'id'), ('log', 'id', 'id'), ('id', 'log', 'id')])
def test_triangleope(axes, num_threads, pt_powers, ch_powers, average_verts, nparticles):

    bin_ranges = [(1e-5, 1), (1e-5, 1), (0, np.pi/2)]
    eec = EECTriangleOPE(nbins=(15, 15, 15), axes=axes, axis_ranges=bin_ranges,
                         pt_powers=pt_powers, ch_powers=ch_powers,
                         print_every=0, num_threads=num_threads, average_verts=average_verts)
    super_slow_eec = SuperSlowEECTriangleOPE(3, (15, 15, 15), bin_ranges, axes, True, pt_powers, ch_powers,
                                             average_verts=average_verts)

    nev = 100
    local_events = [event[-nparticles:] for event in events[:nev]]
    weights = 2*np.random.rand(len(events))[:nev]

    eec(local_events, weights)
    super_slow_eec(local_events, weights)

    for hist_i in range(super_slow_eec.hist.shape[0]):
        hist, errs = eec.get_hist_errs(hist_i)
        print(np.max(np.abs(hist - super_slow_eec.hist[hist_i])), hist_i, super_slow_eec.hist.shape[0])
        assert epsilon_either(hist, super_slow_eec.hist[hist_i], 1e-10, 1e-14), hist_i
        assert epsilon_either(errs, super_slow_eec.errs[hist_i], 1e-6, 1e-7), hist_i

@pytest.mark.longestside
@pytest.mark.pickle
@pytest.mark.parametrize('nparticles', [2, 4, 8])
@pytest.mark.parametrize('ch_powers', [0, 1, 2])
@pytest.mark.parametrize('pt_powers', [1, 2])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4])
def test_pickling_longestside(N, axis, num_threads, pt_powers, ch_powers, nparticles):

    nbins = 15
    eec = EECLongestSide(N, nbins, axis=axis, axis_range=(1e-5, 1), pt_powers=(pt_powers,), ch_powers=(ch_powers,),
                         print_every=0, num_threads=num_threads)

    local_events = [event[:nparticles] for event in events]
    weights = 2*np.random.rand(len(events))

    eec(local_events, weights)

    with tempfile.TemporaryFile() as f:
        pickle.dump(eec, f)
        f.seek(0)
        eec_loaded = pickle.load(f)

    for i in range(2):
        assert np.all(eec_loaded.get_hist_vars()[i] == eec.get_hist_vars()[i])

@pytest.mark.triangleope
@pytest.mark.pickle
@pytest.mark.parametrize('nparticles', [2, 4, 8])
@pytest.mark.parametrize('ch_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('pt_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('axes', [('log', 'log', 'id'), ('id', 'id', 'id'), ('log', 'id', 'id'), ('id', 'log', 'id')])
def test_pickling_triangleope(axes, num_threads, pt_powers, ch_powers, nparticles):
    
    bin_ranges = [(1e-5, 1), (1e-5, 1), (0, np.pi/2)]
    eec = EECTriangleOPE(nbins=(15, 15, 15), axes=axes, axis_ranges=bin_ranges,
                         pt_powers=pt_powers, ch_powers=ch_powers,
                         print_every=0, num_threads=num_threads)

    nev = 100
    local_events = [event[-nparticles:] for event in events[:nev]]
    weights = 2*np.random.rand(len(events))[:nev]

    eec(local_events, weights)

    with tempfile.TemporaryFile() as f:
        pickle.dump(eec, f)
        f.seek(0)
        eec_loaded = pickle.load(f)

    for i in range(2):
        assert np.all(eec_loaded.get_hist_vars()[i] == eec.get_hist_vars()[i])
