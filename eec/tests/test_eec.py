from collections import Counter
import itertools
import math

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

def epsilon_diff(X, Y, epsilon=10**-14):
    return np.all(np.abs(X - Y) < epsilon)

def epsilon_percent(X, Y, epsilon=10**-14):
    return np.all(2*np.abs(X - Y)/(np.abs(X) + np.abs(Y) + 1e-100) < epsilon)

# function for getting histograms from observable values
def calc_eec_hist_on_event(vals, bins, weights):
    
    # compute histogram with errors
    hist = np.histogram(vals, bins=bins, weights=weights)[0]
    errs2 = hist**2
        
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
                if bs[0] != 0.:
                    bs = np.concatenate(([0.], bs))
                if bs[-1] != np.inf:
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
@pytest.mark.parametrize('nparticles', [0, 1, 2, 5, 10][2:])
@pytest.mark.parametrize('ch_powers', [0, 1, 2][:1])
@pytest.mark.parametrize('pt_powers', [1, 2][:1])
@pytest.mark.parametrize('nbins', [1, 15])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4])
def test_pycompare_longestside(N, axis, nbins, pt_powers, ch_powers, nparticles):

    super_slow_eec = SuperSlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles,:(3 if ch_powers==0 else 4)] for event in events[:200]]
    weights = 2*np.random.rand(len(events))

    super_slow_eec(local_events, weights)
    slow_eec(local_events, weights)

    assert epsilon_percent(super_slow_eec.hist, slow_eec.hist, 10**-12)
    assert epsilon_percent(super_slow_eec.errs, slow_eec.errs, 10**-6)

@pytest.mark.longestside
@pytest.mark.sym
@pytest.mark.parametrize('nparticles', [0, 1, 2, 5, 10])
@pytest.mark.parametrize('ch_powers', [0, 1, 2])
@pytest.mark.parametrize('pt_powers', [1, 2])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('nbins', [1, 15])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4, 5, 6])
def test_longestside_sym(N, axis, nbins, num_threads, pt_powers, ch_powers, nparticles):
    if nparticles > 5 and N > 5:
        pytest.skip()

    eec = EECLongestSide(N, nbins, axis=axis, axis_range=(1e-5, 1), pt_powers=(pt_powers,), ch_powers=(ch_powers,),
                         print_every=0, num_threads=num_threads)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles,:(3 if ch_powers==0 else 4)] for event in events]
    weights = 2*np.random.rand(len(events))

    eec(local_events, weights)
    slow_eec(local_events, weights)

    hist, errs = eec.get_hist_errs()
    assert epsilon_percent(hist, slow_eec.hist, 10**-9.5)
    assert epsilon_percent(errs, slow_eec.errs, 10**-5.5)

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

    local_events = [event[:nparticles] for event in events]
    weights = 2*np.random.rand(len(events))

    eec(local_events, weights)
    super_slow_eec(local_events, weights)

    hist, errs = eec.get_hist_errs()
    assert epsilon_percent(hist, super_slow_eec.hist, 10**-10)
    assert epsilon_percent(errs, super_slow_eec.errs, 10**-5.5)

@pytest.mark.longestside
@pytest.mark.asym
@pytest.mark.parametrize('nparticles', [1, 2, 10])
@pytest.mark.parametrize('ch_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('pt_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('axis', ['log', 'id'])
def test_longestside_asym_N3_average_verts(axis, num_threads, pt_powers, ch_powers, nparticles):

    eec = EECLongestSide(3, 15, axis=axis, axis_range=(1e-5, 1), pt_powers=pt_powers, ch_powers=ch_powers,
                         print_every=0, num_threads=num_threads, average_verts=True)
    super_slow_eec = SuperSlowEECLongestSide(3, (15,), ((1e-5, 1),), (axis,), True, pt_powers, ch_powers)

    nev = 100
    local_events = [event[:nparticles] for event in events[:nev]]
    weights = 2*np.random.rand(len(events))[:nev]

    eec(local_events, weights)
    super_slow_eec(local_events, weights)

    hist, errs = eec.get_hist_errs()
    assert epsilon_percent(hist, super_slow_eec.hist, 10**-10)
    assert epsilon_percent(errs, super_slow_eec.errs, 10**-5.5)    

