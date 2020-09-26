# EnergyEnergyCorrelators

This library can be used to compute various Energy-Energy Correlators (EECs) [[1, 2, 3](#references)] on collections of particles. The core computations are carried out efficiently in C++ utilizing the [BOOST Histogram package](https://www.boost.org/doc/libs/1_73_0/libs/histogram/doc/html/index.html) (a copy of which is distributed with this library). Note that a C++14-compatible compiler is required by the BOOST Histogram package. The C++ interface is header-only to facilitate easy integration into any existing framework, including the [FastJet library](http://fastjet.fr/). A convenient Python interface is also provided using Cython and multiprocessing.

## Documentation

This README is currently the documentation for the EEC library. In the future, this package may be incorporated into other projects, such as [EnergyFlow](https://energyflow.network) or [FastJet contrib](https://fastjet.hepforge.org/contrib/).

The EEC library provides a variety of types of energy-energy correlators that can be computed, utilizing a flexible design sctructure that facilitates adding new computations easily. Each computation is represented by its own C++ class, which derives from the common `EECBase` class that contains common functionality such as passing in particles and extracting the histogrammed result.

Since the value of any EEC on a given event is a distribution, a histogram must be declared in advance that will be filled event-by-event. The computation classes are templated to allow for user specification of axes transformations. The template arguments should be models of the [BOOST Histogram Transform concept](https://www.boost.org/doc/libs/1_73_0/libs/histogram/doc/html/histogram/concepts.html#histogram.concepts.Transform), such as `boost::histogram::axis::transform::id` (the identity transform, bins will be linearly spaced) or `boost::histgraom::axis::transform::log` (to get logarithmically spaced bins).

Since version `0.1.0`, the EEC library can raise the transverse momentum of each vertex of the EEC to an arbitrary power (only an exponent of 1 is infrared and collinear safe). Additionally, charge-dependent EECs can be computed by passing in particle charges and specifying an integer exponent for the charge of each vertex of the EEC.

Current EEC computations are described below:

- [EECLongestSide](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECLongestSide.hh#L58): Computes the N-point EEC distribution binned according to the longest side (largest angle) between the N particles. Supported values of N are 2, 3, 4, and 5 (larger values are simply computationally untenable). Since the resulting distribution is one-dimensional, there is a single template argument (that defaults to the identity) specifying which type of axis should be used. The [constructor](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECLongestSide.hh#L67) takes four required arguments: the number of bins, axis minimum, axis maximum, and value of N. Additionally, there are futher default arguments which are detailed below. `EECLongestSide_id` and `EECLongestSide_log` are provided as typdefs of this class with the axis transform already included.

- [EECTriangleOPE](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECTriangleOPE.hh#L54): This is a three-dimensional EEEC that uses coordinates that are particularly suited for studying the OPE limit where two of the three particles are close to collinear. There are three template arguments, corresponding to the `xL`, `xi`, and `phi` axes, respectively. The [constructor](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECTriangleOPE.hh#L61) takes nine arguments: for each axis (in the same order as the template arguments), the number of bins, axis minimum, and axis maximum. Additionally, the same  `EECTriangleOPE_id_id_id`, `EECTriangleOPE_log_id_id`, `EECTriangleOPE_id_log_id`, and `EECTriangleOPE_log_log_id` are provided as typedefs of this class with the axes transforms already specified.

Common arguments to each EEC class are the following:

- `bool norm`: whether or not to divide the transverse momenta by their sum prior to computing the EEC.
- `vector<double> pt_powers`: exponent for the pT on each vertex of the EEC. Length must match the number of particles being correlated, or else be length 1 in which case the same power is used for all vertices.
- `vector<unsigned int> ch_powers`: similar to `pt_powers` except that these are the exponents of the charges for each vertex. If any of these are non-zero, then particle charges are expected to be provided.
- `bool check_degen`: do no EEC computation but check if any particle distances are degenerate will each other in a given event.
- `bool average_verts`: do not separate the computation based on the asymmetry of the vertices, but instead average over all combinations (see below).

If `pt_powers` and `ch_powers` create distinguished vertices, then more than one histogram will be employed to calculate the EEC for each possibility (the `average_verts` option turns off this behavior). Currently, this is only relevant for N=3 computations: N=2 is automatically symmetric and N=4 and N=5 require symmetric vertices. The `description` method of the class contains information about the different histograms created.

The resulting histogram and corresponding errors can be accessed with the [`get_hist`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECHist.hh#L100-L105) method, which accepts a boolean for whether or not to include the overflow bins and an index which defaults to 0 for which histogram to select (see above for how there can be multiple histograms per computation) and returns a pair of vectors of doubles, which are the flattened  (C-style) histogram values and uncertainties. There are also `bin_edges` and `bin_centers` methods (specific to each computation class) that return the bins, which take an integer to specify an axis (if there is more than one).

Additionally, if the `EEC_HIST_FORMATTED_OUTPUT` macro is defined prior to the includion of `EEC.hh` (note that this requires that `boost/format.hpp` is available), then the histograms are printable to any output stream using the `output` method.

## C++ Usage (Header-only)

The entire library is contained in a single header file, [`eec/include/EEC.hh`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EEC.hh). If you plan on using the EEC library with FastJet, ensure that `PseudoJet.hh` is included prior to including `EEC.hh`. This will expose an overloaded [`compute`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECBase.hh#L278-L304) method for each EEC computation that accepts a vector of `PseudoJet` objects. Otherwise, there is a `compute` method that takes a vector of doubles, which must be size `3N` where `N` is the number of particles, arranged as `pT1, rap1, phi1, pT2, rap2, phi2, ..., pTN, rapN, phiN`.

## Python Usage

The EEC library also contains a Cython-based wrapper of the core C++ code. This is most easily used by installing via `pip`, e.g. `pip install eec`. NumPy is the only required package. Note that a C++14-enabled compiler must be available for the compilation from source to succeed.

There is one Python class for each EEC computation. The templated arguments are dealt with by specifying the axis transforms as a tuple of strings. Currently, only `'id'` and `'log'` are supported, in the combinations for which there is a provided C++ typedef (see above). The arguments to the classes are straightforward, and can be examined more closely in [core.pyx](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/core.pyx). There is also an [`eec`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/__init__.py#L50) method that can be used to parallelize computations on many events in Python.

## References

[1] H. Chen, M. Luo,  I. Moult, T. Yang, X. Zhang, H. X. Zhu, _Three Point Energy Correlators in the Collinear Limit: Symmetries, Dualities and Analytic Results_, [[1912.11050](https://arxiv.org/abs/1912.11050)].

[2] H. Chen, I. Moult, X. Zhang, H. X. Zhu, _Rethinking Jets with Energy Correlators: Tracks, Resummation and Analytic Continuation_, [[2004.11381](https://arxiv.org/abs/2004.11381)].

[3] L. Dixon, P. T. Komiske, I. Moult, J. Thaler, H. X. Zhu, _Analyzing N-Point Energy Correlators with CMS Open Data_, to appear soon.
