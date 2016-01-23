entro-py
========

entro-py computes entropy for time series analysis. Currently implemented
algorithms are FuzzyEn and SampEn, both of which are described/compared in
[Chen et al. 2009][ChenEtAl-2009]. Implementation has been "inspired" by and
tested against this [MATLAB code][matlab].


Dependencies
------------

* [numpy][numpy]


TODO
----

Items in order of importance

* Implement cross entropy
* Pass in arbitrary function for similarity measurement
* Pass 2D array and calculate entropy along a given axis


[ChenEtAl-2009]: http://dx.doi.org/10.1016/j.medengphy.2008.04.005
[matlab]: http://www.mathworks.com/matlabcentral/fileexchange/50289-a-set-of-entropy-measures-for-temporal-series--1d-signals-
[numpy]: http://www.numpy.org/
