# EnergyEnergyCorrelators

This library can be used to compute various Energy-Energy Correlators (EECs) [[1, 2, 3](#references)] on collections of particles. The core computations are carried out efficiently in C++ utilizing the [BOOST Histogram package](https://www.boost.org/doc/libs/1_73_0/libs/histogram/doc/html/index.html). Note that a C++14-compatible compiler is required by the BOOST Histogram package. The C++ interface is header-only to facilitate easy integration into any existing framework, including the [FastJet library](http://fastjet.fr/). A convenient Python interface is also provided using Cython.

## Documentation

This README is currently the documentation for the EEC library. This may change in the future as it may be incorporated into other projects, such as [EnergyFlow](https://energyflow.network) or [FastJet contrib](https://fastjet.hepforge.org/contrib/).

The EEC library provides a variety of types of energy-energy correlators that can be computed, utilizing a flexible design sctructure that facilitates adding new computations easily. Each computation is represented by its own C++ class, which derives from the common `EECComputation` base class that contains common functionality such as passing in particles and extracting the histogrammed result.

Since the value of any EEC on a given event is a distribution, a histogram must be declared in advance that will be filled event-by-event. The computation classes are templated to allow for user specification of axes transformations. The template arguments should be models of the [BOOST Histogram Transform concept](https://www.boost.org/doc/libs/1_73_0/libs/histogram/doc/html/histogram/concepts.html#histogram.concepts.Transform), such as `boost::histogram::axis::transform::id` (the identity transform, bins will be linearly spaced) or `boost::histgraom::axis::transform::log` (to get logarithmically spaced bins).

The current EEC computations are described below:

- [EECLongestSide](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECComputations.hh#L378): Computes the N-point EEC distribution binned according to the longest side (largest angle) between the N particles. Supported values of N are 2, 3, 4, and 5 (larger values are simply computationally untenable). Since the resulting distribution is one-dimensional, there is a single template argument (that defaults to the identity) specifying which type of axis should be used. The [constructor](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECComputations.hh#L397) takes four arguments: the value of N, number of bins, axis minimum, and axis maximum. `EECLongestSide_id` and `EECLongestSide_log` are provided as typdefs of this class with the axis transform already included.

- [EECTriangleOPE](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECComputations.hh#L234): This is a three-dimensional EEEC that uses coordinates that are particularly suited for studying the OPE limit where two of the three particles are close to collinear. There are three template arguments, corresponding to the `xL`, `xi`, and `phi` axes, respectively. The [constructor](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECComputations.hh#L250) takes nine arguments: for each axis (in the same order as the template arguments), the number of bins, axis minimum, and axis maximum. `EECTriangleOPE_id_id_id`, `EECTriangleOPE_log_id_id`, `EECTriangleOPE_id_log_id`, and `EECTriangleOPE_log_log_id` are provided as typedefs of this class with the axes transforms already specified.

## C++ Usage (Header-only)

The entire library is contained in a single header file, [`eec/include/EECComputations.hh`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECComputations.hh). If you plan on using the EEC library with FastJet, ensure that `PseudoJet.hh` is included prior to including `EECComputations.hh`. This will expose an overloaded [`compute`](https://github.com/pkomiske/EnergyEnergyCorrelators/blob/master/eec/include/EECComputations.hh#L177-L195) method for each EEC computation that accepts a vector of `PseudoJet` objects. Otherwise, there is a `compute` method that takes a vector of `double`s, which must be size `3N` where `N` is the number of particles, arranged as `pT1, rap1, phi1, pT2, rap2, phi2, ..., pTN, rapN, phiN`.

## Python Usage

The EEC library also contains a Cython-based wrapper of the core C++ code. This is most easily used by installing via `pip`, e.g. `pip install eec`. Cython and NumPy are the only required dependencies. Note that a C++14-enabled compiler must be usable by Cython for the installation to succeed.

## References

[1] H. Chen, M. Luo,  I. Moult, T. Yang, X. Zhang, H. X. Zhu, _Three Point Energy Correlators in the Collinear Limit: Symmetries, Dualities and Analytic Results_, [[1912.11050](https://arxiv.org/abs/1912.11050)].

[2] H. Chen, I. Moult, X. Zhang, H. X. Zhu, _Rethinking Jets with Energy Correlators: Tracks, Resummation and Analytic Continuation_, [[2004.11381](https://arxiv.org/abs/2004.11381)].

[3] L. Dixon, P. T. Komiske, I. Moult, J. Thaler, H. X. Zhu, _Analyzing N-Point Energy Correlators with CMS Open Data_, to appear soon.
