from test_eec import *

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

for N in [2, 3]:
	for axis in ['log']:
		for nbins in [1, 15]:
			for weight_powers in [1, 2]:
				for charge_powers in [0, 1]:
					for nparticles in [0, 1, 2, 4, 8]:
						test_pycompare_longestside(N, axis, nbins, weight_powers, charge_powers, nparticles)
