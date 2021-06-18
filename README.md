# EnergyEnergyCorrelators

[![build-wheels](https://github.com/pkomiske/EnergyEnergyCorrelators/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/pkomiske/EnergyEnergyCorrelators/actions)

[![PyPI version](https://badge.fury.io/py/EnergyEnergyCorrelators.svg)](https://pypi.org/project/EnergyEnergyCorrelators/)
[![python versions](https://img.shields.io/pypi/pyversions/EnergyEnergyCorrelators)](https://pypi.org/project/EnergyEnergyCorrelators/)

This library can be used to compute various Energy-Energy Correlators (EECs) [[1, 2, 3](#references)] on collections of particles. The core computations are carried out efficiently in C++ utilizing the [BOOST Histogram package](https://www.boost.org/doc/libs/1_76_0/libs/histogram/doc/html/index.html) (a copy of which is distributed with this library). Note that a C++14-compatible compiler is required by the BOOST Histogram package. The C++ interface is header-only to facilitate easy integration into any existing framework, including the [FastJet library](http://fastjet.fr/). A convenient Python interface is also provided via SWIG.

## Documentation

This README is currently the documentation for the EEC library. In the future, this package may be incorporated into other projects, such as [EnergyFlow](https://energyflow.network) or [FastJet contrib](https://fastjet.hepforge.org/contrib/).

The EEC library provides a variety of types of energy-energy correlators that can be computed, utilizing a flexible design sctructure that facilitates adding new computations easily. Each computation is represented by its own C++ class, which derives from the common `EECBase` class that contains common functionality such as passing in particles and extracting the histogrammed result.

Since the value of any EEC on a given event is a distribution, a histogram must be declared in advance that will be filled event-by-event. The computation classes are templated to allow for user specification of axes transformations. The template arguments should be models of the [BOOST Histogram Transform concept](https://www.boost.org/doc/libs/1_76_0/libs/histogram/doc/html/histogram/concepts.html#histogram.concepts.Transform), such as `boost::histogram::axis::transform::id` (the identity transform, bins will be linearly spaced) or `boost::histgraom::axis::transform::log` (to get logarithmically spaced bins).

The EEC library can raise the transverse momentum of each vertex of the EEC to an arbitrary power (only an exponent of 1 is infrared and collinear safe). Additionally, charge-dependent EECs can be computed by passing in particle charges and specifying an integer exponent for the charge of each vertex of the EEC. Tests have been written which cover the majority of the Python package.

Current EEC computations are described below:

- [EECLongestSide](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECLongestSide.hh): Computes the N-point EEC distribution binned according to the longest side (largest angle) between the N particles. Supported values of N are 2 through a maximum of 12 (on 32-bit architectures) or 20 (on 64-bit architectures) . Since the resulting distribution is one-dimensional, there is a single template argument (that defaults to the identity) specifying which type of axis should be used. The [constructor](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECLongestSide.hh#L157) takes four required arguments: the value of N, number of bins, axis minimum, axis maximum. Additionally, there are futher default arguments which are detailed below. `EECLongestSideId` and `EECLongestSideLog` are provided as typdefs of this class with the axis transform already included.

- [EECTriangleOPE](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECTriangleOPE.hh): This is a three-dimensional EEEC that uses coordinates that are particularly suited for studying the OPE limit where two of the three particles are close to collinear. There are three template arguments, corresponding to the `xL`, `xi`, and `phi` axes, respectively. The [constructor](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECTriangleOPE.hh#L118) takes nine arguments: for each axis (in the same order as the template arguments), the number of bins, axis minimum, and axis maximum. Additionally, the following typedefs are provided that include the axes transforms as specified: `EECTriangleOPEIdIdId`, `EECTriangleOPELogIdId`, `EECTriangleOPEIdLogId`, and `EECTriangleOPELogLogId`.

Optional arguments (those with defaults) to each EEC class are the following:

- `bool norm = true`: whether or not to divide the transverse momenta by their sum prior to computing the EEC.
- `vector<double> pt_powers = {1}`: exponent for the pT on each vertex of the EEC. Length must match the number of particles being correlated, or else be length 1 in which case the same power is used for all vertices.
- `vector<unsigned> ch_powers = {0}`: similar to `pt_powers` except that these are the exponents of the charges for each vertex. If any of these are non-zero, then particle charges are expected to be provided.
- `int num_threads = -1`: Number of threads to use for batch computations. A value of `-1` means use all available CPU cores.
- `long print_every = -10`: 
- `bool check_degen = false`: do no EEC computation but check if any particle distances are degenerate will each other in a given event.
- `bool average_verts = false`: do not separate the computation based on the asymmetry of the vertices, but instead average over all combinations (see below).
- `bool track_covariance = [computation dependent]`: This option is `true` for `EECLongestSide` and `false` for `EECTriangleOPE`. It indicates whether or not the covariance of the histogram bins should be stored. Note that this requires a histogram with double the number of axes, so it can be quite computationally expensive.
- `bool variance_bound = true`: Whether to calculate a crude upper bound on the variance of each histogram bin.
- `bool variance_bound_includes_overflows = true`: Whether or not the variance upper bound should include overflow bins when summing histograms.
- `bool use_general_eNc = false` (`EECLongestSide` only): The longest side EEC is hard-coded up to `N=8`, otherwise a general code is used that works for any `N`. This flag can be used to force that code to be used even for `N<=8`.

If `pt_powers` and `ch_powers` create distinguished vertices, then more than one histogram will be employed to calculate the EEC for each possibility (the `average_verts` option turns off this behavior). Currently, this is only relevant for N=3 computations: N=2 is automatically symmetric and the code for N>3 requires symmetric vertices. The `description` method of the class contains information about the different histograms created.

The resulting histogram values and variances can be accessed with the [`get_hist_vars`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECHistBase.hh#L352-L359) method, which accepts an index which defaults to 0 for which histogram to select (see above for how there can be multiple histograms per computation) and a boolean for whether or not to include the overflow bins, returning a pair of vectors of doubles, which are the flattened (C-style) histogram values and variances. There is also a [`get_hist_errs`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECHistBase.hh#L361-L368) method for taking the square root of the variances to get standard deviations. See the [`get_covariance`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECHistBase.hh#L371-L377) and [`get_variance_bound`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECHistBase.hh#L378-L384) methods for accessing this information. There are also `bin_edges` and `bin_centers` methods that return the bins (or their centers), which take an integer to specify an axis (if there is more than one). The histograms are printable using the [`hists_as_text`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECHistBase.hh#L320-L348) method.

## C++ Usage (Header-only)

The entire library can be used header-only; one only needs to include [`eev/include/EEC.hh`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EEC.hh) to access the full functionality. If you plan on using the EEC library with FastJet, ensure that `PseudoJet.hh` is included prior to including `EEC.hh`. This will expose an overloaded [`compute`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECBase.hh#L418-L449) method for each EEC computation that accepts a vector of `PseudoJet` objects. Otherwise, there is a `compute` method that takes a vector of doubles, which must be size `3M` where `M` is the number of particles, arranged as `pT1, rap1, phi1, pT2, rap2, phi2, ..., pTM, rapM, phiM`. If charge are to be used, then the vector of doubles is expected to be `4M` in length, arranged as `pT1, rap1, phi1, ch1, pT2, rap2, phi2, ch2, ..., pTM, rapM, phiM, chM`. See also the [`batch_compute`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECBase.hh#L272) method for parallelized computations.

## Python Usage

The EEC library also contains a SWIG-generated wrapper of the core C++ code. This is most easily used by installing via `pip`, e.g. `pip3 install EnergyEnergyCorrelators`. NumPy is the only required package. Note that a C++14-enabled compiler must be available for the compilation from source to succeed.

There is one Python class for each EEC computation. The templated arguments are dealt with by specifying the axis transforms as a tuple of strings. Currently, only `'id'` and `'log'` are supported, in the combinations for which there is a provided C++ typedef (see above). The arguments to the classes are straightforward, and mirror those of the C++ class constructor definition.

## References

[1] H. Chen, M. Luo,  I. Moult, T. Yang, X. Zhang, H. X. Zhu, _Three Point Energy Correlators in the Collinear Limit: Symmetries, Dualities and Analytic Results_, [[1912.11050](https://arxiv.org/abs/1912.11050)].

[2] H. Chen, I. Moult, X. Zhang, H. X. Zhu, _Rethinking Jets with Energy Correlators: Tracks, Resummation and Analytic Continuation_, [[2004.11381](https://arxiv.org/abs/2004.11381)].

[3] P. T. Komiske, I. Moult, J. Thaler, H. X. Zhu, _Analyzing N-Point Energy Correlators with CMS Open Data_, to appear soon.
