from test_eec import *

def test_longestside_sym(N, axis, use_general_eNc, num_threads, weight_powers, charge_powers, nparticles):
    if nparticles > 8 and N >= 5:
        pytest.skip()

    nbins = 15
    eec = EECLongestSide(N, nbins, axis=axis, axis_range=(1e-5, 1.0), weight_powers=(weight_powers,), charge_powers=(charge_powers,),
                         print_every=0, num_threads=num_threads, use_general_eNc=use_general_eNc)
    slow_eec = SlowEECLongestSideSym(N, (nbins,), ((1e-5, 1),), (axis,), True, weight_powers, charge_powers)
    slow_eec.construct_inds_factors(nparticles, N)

    local_events = [event[:nparticles] for event in events]
    weights = 2*np.random.rand(len(local_events))

    eec(local_events, event_weights=weights)
    slow_eec(local_events, weights)

    hist, errs = eec.get_hist_errs()
    print(np.max(np.abs(hist - slow_eec.hist)), np.max(np.abs(errs - slow_eec.errs)))
    assert epsilon_either(hist, slow_eec.hist, 10**-12, 1e-14)
    assert epsilon_either(errs, slow_eec.errs, 10**-6, 1e-7)

for N in [2, 3, 4]:
    for axis in ['log', 'id']:
        for use_general_eNc in [False, True]:
            for num_threads in [1, -1]:
                for weight_powers in [1, 2]:
                    for charge_powers in [0, 1, 2]:
                        for nparticles in [0, 1, 2, 4, 8, 16]:
                            test_longestside_sym(N, axis, use_general_eNc, num_threads, weight_powers, charge_powers, nparticles)