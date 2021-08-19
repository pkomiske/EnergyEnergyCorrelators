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

import pickle
import tempfile

import energyflow as ef
import numpy as np
import pytest

from helpers import *

# load some reasonable test data
events, y = ef.qg_jets.load(num_data=500, pad=False)
for i in range(len(events)):
    events[i][:,3] = ef.pids2chrgs(events[i][:,3])

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
@pytest.mark.parametrize('charge_powers', [0, 1, 2])
@pytest.mark.parametrize('weight_powers', [1, 2])
@pytest.mark.parametrize('nbins', [1, 15])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4])
def test_pycompare_longestside(N, axis, nbins, weight_powers, charge_powers, nparticles):

    super_slow_eec = SuperSlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles] for event in events[:200]]
    weights = 2*np.random.rand(len(local_events))

    super_slow_eec(local_events, weights)
    slow_eec(local_events, weights)

    assert epsilon_percent(super_slow_eec.hist, slow_eec.hist, 1e-12)
    assert epsilon_percent(super_slow_eec.errs, slow_eec.errs, 1e-6)

@pytest.mark.longestside
@pytest.mark.sym
@pytest.mark.parametrize('nparticles', [0, 1, 2, 4, 8, 16])
@pytest.mark.parametrize('charge_powers', [0, 1, 2])
@pytest.mark.parametrize('weight_powers', [1, 2])
@pytest.mark.parametrize('num_threads', [1, 3])
@pytest.mark.parametrize('use_general_eNc', [False, True])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4, 5, 6])
def test_longestside_sym(N, axis, use_general_eNc, num_threads, weight_powers, charge_powers, nparticles):
    if nparticles > 8 and N >= 5:
        pytest.skip()

    nbins = 15
    e = eec.EECLongestSide(N, nbins, axis=axis, axis_range=(1e-5, 1.0), weight_powers=(weight_powers,), charge_powers=(charge_powers,),
                              print_every=0, num_threads=num_threads, use_general_eNc=use_general_eNc)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles] for event in events]
    weights = 2*np.random.rand(len(local_events))

    e(local_events, event_weights=weights)
    slow_eec(local_events, weights)

    hist, errs = e.get_hist_errs()
    assert epsilon_either(hist, slow_eec.hist, 10**-12, 1e-14)
    assert epsilon_either(errs, slow_eec.errs, 10**-6, 1e-7)

@pytest.mark.longestside
@pytest.mark.asym
@pytest.mark.parametrize('nparticles', [1, 2, 5, 10])
@pytest.mark.parametrize('charge_powers', [(0,0), (1,1), (2,2), (0,1), (1,0), (2,0), (0,2)])
@pytest.mark.parametrize('weight_powers', [(1,1), (1,2), (2,1), (2,2), (0,1), (1,0)])
@pytest.mark.parametrize('num_threads', [1, 3])
@pytest.mark.parametrize('axis', ['log', 'id'])
def test_longestside_asym_N2_average_verts(axis, num_threads, weight_powers, charge_powers, nparticles):

    e = eec.EECLongestSide(2, 15, axis=axis, axis_range=(1e-5, 1), weight_powers=weight_powers, charge_powers=charge_powers,
                           print_every=0, num_threads=num_threads)
    super_slow_eec = SuperSlowEECLongestSide(2, (15,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers)

    local_events = [event[-nparticles:] for event in events]
    weights = 2*np.random.rand(len(local_events))

    e(local_events, event_weights=weights)
    super_slow_eec(local_events, weights)

    hist, errs = e.get_hist_errs()
    assert epsilon_either(hist, super_slow_eec.hist, 10**-11, 1e-13)
    assert epsilon_either(errs, super_slow_eec.errs, 10**-6, 1e-6)

@pytest.mark.longestside
@pytest.mark.asym
@pytest.mark.parametrize('nparticles', [1, 2, 10])
@pytest.mark.parametrize('average_verts', [True, False])
@pytest.mark.parametrize('charge_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('weight_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, 3])
@pytest.mark.parametrize('axis', ['log', 'id'])
def test_longestside_asym_N3(axis, num_threads, weight_powers, charge_powers, average_verts, nparticles):

    e = eec.EECLongestSide(3, 15, axis=axis, axis_range=(1e-5, 1), weight_powers=weight_powers, charge_powers=charge_powers,
                           print_every=0, num_threads=num_threads, average_verts=average_verts)
    super_slow_eec = (SuperSlowEECLongestSide(3, (15,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers)
                      if average_verts or (len(set(weight_powers)) == 1 and len(set(charge_powers)) == 1) else
                      SuperSlowEECLongestSideAsymN3(3, (15,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers))

    nev = 100
    local_events = [event[-nparticles:] for event in events[:nev]]
    weights = 2*np.random.rand(len(local_events))

    e(local_events, event_weights=weights)
    super_slow_eec(local_events, weights)

    if average_verts or len(super_slow_eec.hist.shape) == 1:
        hist, errs = e.get_hist_errs()
        assert epsilon_either(hist, super_slow_eec.hist, 10**-11, 1e-13)
        assert epsilon_either(errs, super_slow_eec.errs, 10**-6, 1e-6)
    else:
        for hist_i in range(super_slow_eec.hist.shape[0]):
            hist, errs = e.get_hist_errs(hist_i)
            assert epsilon_either(hist, super_slow_eec.hist[hist_i], 1e-12, 1e-13), hist_i
            assert epsilon_either(errs, super_slow_eec.errs[hist_i], 1e-6, 1e-6), hist_i

@pytest.mark.triangleope
@pytest.mark.parametrize('nparticles', [1, 4, 10])
@pytest.mark.parametrize('average_verts', [True, False])
@pytest.mark.parametrize('charge_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('weight_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, 3])
@pytest.mark.parametrize('axes', [('log', 'log', 'id'), ('id', 'id', 'id'), ('log', 'id', 'id'), ('id', 'log', 'id')])
def test_triangleope(axes, num_threads, weight_powers, charge_powers, average_verts, nparticles):

    bin_ranges = [(1e-5, 1), (1e-5, 1), (0, np.pi/2)]
    e = eec.EECTriangleOPE(nbins=(15, 15, 15), axes=axes, axes_range=bin_ranges,
                           weight_powers=weight_powers, charge_powers=charge_powers,
                           print_every=0, num_threads=num_threads, average_verts=average_verts)
    super_slow_eec = SuperSlowEECTriangleOPE(3, (15, 15, 15), bin_ranges, axes, True, weight_powers, charge_powers,
                                             average_verts=average_verts)

    nev = 100
    local_events = [event[-nparticles:] for event in events[:nev]]
    weights = 2*np.random.rand(len(local_events))

    e(local_events, event_weights=weights)
    super_slow_eec(local_events, weights)

    for hist_i in range(super_slow_eec.hist.shape[0]):
        hist, errs = e.get_hist_errs(hist_i)
        print(np.max(np.abs(hist - super_slow_eec.hist[hist_i])), hist_i, super_slow_eec.hist.shape[0])
        assert epsilon_either(hist, super_slow_eec.hist[hist_i], 1e-10, 1e-14), hist_i
        assert epsilon_either(errs, super_slow_eec.errs[hist_i], 1e-6, 1e-7), hist_i

@pytest.mark.longestside
@pytest.mark.pickle
@pytest.mark.parametrize('compmode', [eec.CompressionMode_Auto, eec.CompressionMode_Plain, eec.CompressionMode_Zlib])
@pytest.mark.parametrize('archform', [eec.ArchiveFormat_Text, eec.ArchiveFormat_Binary])
@pytest.mark.parametrize('nparticles', [2, 4, 8])
@pytest.mark.parametrize('charge_powers', [0, 1, 2])
@pytest.mark.parametrize('weight_powers', [1, 2])
@pytest.mark.parametrize('num_threads', [1, 3])
@pytest.mark.parametrize('axis', ['log', 'id'])
@pytest.mark.parametrize('N', [2, 3, 4])
def test_pickling_longestside(N, axis, num_threads, weight_powers, charge_powers, nparticles, archform, compmode):

    if eec.HAS_SERIALIZATION_SUPPORT:
        eec.set_archive_format(archform)
        eec.set_compression_mode(compmode)

    nbins = 15
    e = eec.EECLongestSide(N, nbins, axis=axis, axis_range=(1e-5, 1), weight_powers=(weight_powers,), charge_powers=(charge_powers,),
                           print_every=0, num_threads=num_threads)

    local_events = [event[:nparticles] for event in events]
    weights = 2*np.random.rand(len(local_events))

    e(local_events, event_weights=weights)

    if not eec.HAS_SERIALIZATION_SUPPORT:
        d = e.as_dict()
    else:
        with tempfile.TemporaryFile() as f:
            pickle.dump(e, f)
            f.seek(0)
            e_loaded = pickle.load(f)

        assert e == e_loaded, 'EECs did not match'

        for i in range(2):
            assert np.all(e_loaded.get_hist_vars()[i] == e.get_hist_vars()[i]), 'hists did not match'

@pytest.mark.triangleope
@pytest.mark.pickle
@pytest.mark.parametrize('compmode', [eec.CompressionMode_Auto, eec.CompressionMode_Plain, eec.CompressionMode_Zlib])
@pytest.mark.parametrize('archform', [eec.ArchiveFormat_Text, eec.ArchiveFormat_Binary])
@pytest.mark.parametrize('nparticles', [2, 4, 8])
@pytest.mark.parametrize('charge_powers', [(0,0,0), (1,1,1), (0,0,1), (0,1,0), (1,0,0)])
@pytest.mark.parametrize('weight_powers', [(1,1,1), (1,1,2), (1,2,1), (2,1,1)])
@pytest.mark.parametrize('num_threads', [1, 3])
@pytest.mark.parametrize('axes', [('log', 'log', 'id'), ('id', 'id', 'id'), ('log', 'id', 'id'), ('id', 'log', 'id')])
def test_pickling_triangleope(axes, num_threads, weight_powers, charge_powers, nparticles, archform, compmode):

    if eec.HAS_SERIALIZATION_SUPPORT:
        eec.set_archive_format(archform)
        eec.set_compression_mode(compmode)
    
    bin_ranges = [(1e-5, 1), (1e-5, 1), (0, np.pi/2)]
    e = eec.EECTriangleOPE(nbins=(15, 15, 15), axes=axes, axes_range=bin_ranges,
                           weight_powers=weight_powers, charge_powers=charge_powers,
                           print_every=0, num_threads=num_threads)

    nev = 100
    local_events = [event[-nparticles:] for event in events[:nev]]
    weights = 2*np.random.rand(len(local_events))

    e(local_events, event_weights=weights)

    if not eec.HAS_SERIALIZATION_SUPPORT:
        d = e.as_dict()
    else:
        with tempfile.TemporaryFile() as f:
            pickle.dump(e, f)
            f.seek(0)
            e_loaded = pickle.load(f)

        assert e == e_loaded, 'EECs did not match'

        for i in range(2):
            assert np.all(e_loaded.get_hist_vars()[i] == e.get_hist_vars()[i]), 'hists did not match'
